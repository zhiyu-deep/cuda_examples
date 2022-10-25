//
// Created by 63479 on 2022/9/1.
//

#include "Conv.h"
#include "ConvUtils.cuh"
#include "common.cuh"
#include "utils_assert.cuh"

__global__ void ConvNaiveImpl(
        const float *a, const float *weight, float *b,
        int n, int inputC, int h, int w, int outputC,
        int kernelX, int kernelY, int dilateX, int dilateY, int strideX, int strideY, int padX, int padY) {
    if (!CheckParamsLegal(h, kernelY, dilateY, padY, strideY) || !CheckParamsLegal(w, kernelX, dilateX, padX, strideX)) {
        assert(false);
    }
    // input卷积点数.
    int conv_rows = CalculateOutputLength(h, kernelY, dilateY, padY, strideY),
        conv_cols = CalculateOutputLength(w, kernelX, dilateX, padX, strideX);
    // attributes:
    int a_batch_stride = (h) * (w) * 4, a_channel_stride = a_batch_stride * n, a_h_stride = (w) * 4;
    // 当前线程: 展开后的output点.
    int outputX = blockIdx.x * 4 + threadIdx.x % 4,
        outputY = blockDim.x / 4 * blockIdx.y + threadIdx.x / 4;
    if (outputY < (n * conv_cols * conv_rows)) {
        // 当前线程: 在input中的卷积点.
        int output_index = outputY % (conv_cols * conv_rows), output_b = outputY / (conv_cols * conv_rows);
        int conv_X = output_index % conv_cols,
            conv_Y = output_index / conv_cols;
        // 当前线程: start点在padding矩阵中的坐标(以矩阵顶点为坐标轴).
        int start_in_pad_input_X = conv_X * strideX - padX,
            start_in_pad_input_Y = conv_Y * strideY - padY;
        const float *pad_a_ptr = a + output_b * a_batch_stride + start_in_pad_input_Y * a_h_stride + start_in_pad_input_X * 4;
        // 当前线程: 有效点.
        int start_offset_X = max(UP_DIVIDE(-start_in_pad_input_X, dilateX), 0),
            start_offset_Y = max(UP_DIVIDE(-start_in_pad_input_Y, dilateY), 0),
            end_offset_x = min(kernelX - 1, DOWN_DIVIDE(w - 1 - start_in_pad_input_X, dilateX)),
            end_offset_Y = min(kernelY - 1, DOWN_DIVIDE(h - 1 - start_in_pad_input_Y, dilateY));
        float a_value = 0., weight_value = 0., accumulate_value = 0.;
        for (int i = 0; i < kernelY; i++) {
            for (int j = 0; j < kernelX; j++) {
                for (int k = 0; k < UP_DIVIDE(inputC, 4); k++) {
                    for (int y = 0; y < inputC - k * 4; y++) {
                        if (i < start_offset_Y || i > end_offset_Y) {
                            a_value = 0.;
                        } else if (j < start_offset_X || j > end_offset_x) {
                            a_value = 0.;
                        } else {
                            a_value = pad_a_ptr[k * a_channel_stride + i * a_h_stride + j * 4 + y];
                        }
                        weight_value = weight[blockIdx.x * 4 * kernelX * kernelY * inputC + (i * kernelX * inputC + j * inputC + k * 4 + y) * 4 + threadIdx.x % 4];
//                        printf("there: %d, %f, %f.\n", threadIdx.x, a_value, weight_value);
                        accumulate_value += weight_value * a_value;
                    }
                }
            }
        }
        b[blockIdx.x * (n * conv_rows * conv_cols * 4) + outputY * 4 + threadIdx.x % 4] = accumulate_value;
    }
}

void ConvNaive(const float *a, const float *weight, float *b,
               int n, int inputC, int h, int w, int outputC,
               int kernelX, int kernelY, int dilateX, int dilateY, int strideX, int strideY, int padX, int padY) {
    int outputH = CalculateOutputLength(h, kernelY, dilateY, padY, strideY),
        outputW = CalculateOutputLength(w, kernelX, dilateX, padX, strideX);
    int planes = n * outputH * outputW;
    int block_size = 256;
    dim3 blocks(block_size), grids(UP_DIVIDE(outputC, 4), UP_DIVIDE(planes, block_size / 4));
    ConvNaiveImpl<<<grids, blocks>>>(a, weight, b, n, inputC, h, w, outputC, kernelX, kernelY, dilateX, dilateY, strideX, strideY, padX, padY);
}