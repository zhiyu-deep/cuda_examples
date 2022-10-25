//
// Created by 63479 on 2022/9/1.
//

#include "Conv.h"
#include "ConvUtils.cuh"
#include "common.cuh"
#include "utils_assert.cuh"

template<int RH, int RW>
__global__ void ConvOpt1Impl(
        const float *a, const float *weight, float *b,
        int n, int inputC, int h, int w, int outputC,
        int kernelX, int kernelY, int dilateX, int dilateY, int strideX, int strideY, int padX, int padY) {
    if (!CheckParamsLegal(h, kernelY, dilateY, padY, strideY) || !CheckParamsLegal(w, kernelX, dilateX, padX, strideX)) {
        assert(false)
    }

    int conv_rows = CalculateOutputLength(h, kernelY, dilateY, padY, strideY),
        conv_cols = CalculateOutputLength(w, kernelX, dilateX, padX, strideX);

    constexpr int deal_rows = RH * 32;

    __shared__ int points;
    extern __shared__ uint8_t share_mem_offsets[];
    int max_lines = UP_DIVIDE(deal_rows, conv_cols) + 1, max_points = max_lines;
    int *share_mem_info = (int*)share_mem_offsets;                                  // size = max_points * 4.
    float **share_mem_ptr = (float**)(share_mem_offsets + max_points * 4);          // size = max_points;

    // output中的索引信息.
    int outputYBegin = blockIdx.y * deal_rows,
        outputYEnd = min(outputYBegin + deal_rows - 1, n * conv_cols * conv_rows - 1),
        dealY = outputYEnd - outputYBegin + 1;
    // input卷积中的索引信息.
    int convYBegin = outputYBegin / conv_cols, convYEnd = outputYEnd / conv_cols,
        in_b_conv_X = outputYBegin % conv_cols,
        in_b_conv_points = min(dealY, conv_cols - in_b_conv_X),
        before_conv_points = 0,
        remain_conv_points = dealY - in_b_conv_points;
    // 处理每一组channel.
    for (int g = 0; g < UP_DIVIDE(inputC, 4); g++) {
        for (int k = 0; k < kernelY * kernelX; k++) {
            if (threadIdx.x == 0) {
                for (int convY = convYBegin; convY <= convYEnd;
                             convY++,
                             in_b_conv_X = 0,
                             before_conv_points += in_b_conv_points,
                             in_b_conv_points = min(remain_conv_points, conv_cols),
                             remain_conv_points -= in_b_conv_points) {
                    // 每个batch中的卷积信息.
                    // 每个卷积点的信息: in_b, in_b_conv_Y, in_b_conv_X.
                    // 卷积历史信息: before_conv_points, in_b_conv_points, remain_conv_points.
                    int in_b_conv_Y = convY % conv_rows, in_b = convY / conv_rows;

                    // 数据索引计算.
                    int y_offset = in_b_conv_Y * strideY - padY, x_offset = in_b_conv_X * strideX - padX;
                    int in_kernelY = k / kernelX, in_kernelX = k % kernelX;
                    if (y_offset + in_kernelY * dilateY >= 0 && y_offset + in_kernelY * dilateY <= h - 1) {
                        int begin = max(0, UP_DIVIDE(-x_offset, strideX)),
                            end = min(in_b_conv_points - 1, DOWN_DIVIDE(w - 1 - x_offset, strideX));
                        // 该点需要卷积.
                        if (end >= begin) {
                            (share_mem_info + convY * 4)[0] = 0;  // x offset.
                            (share_mem_info + convY * 4)[1] = before_conv_points;  // y offset.
                            (share_mem_info + convY * 4)[2] = (inputC - g * 4);  // x length.
                            (share_mem_info + convY * 4)[3] = (end - begin + 1);  // y length.
                            share_mem_ptr[convY] = (float *) (a + in_b * (h * w * 4) + y_offset * w * 4 + x_offset * 4 +
                                                              g * (n * h * w * 4));
                        }
                    }

                }
            }
        }
    }
}

void ConvOpt1(const float *a, const float *weight, float *b,
               int n, int inputC, int h, int w, int outputC,
               int kernelX, int kernelY, int dilateX, int dilateY, int strideX, int strideY, int padX, int padY) {
    int outputH = CalculateOutputLength(h, kernelY, dilateY, padY, strideY),
        outputW = CalculateOutputLength(w, kernelX, dilateX, padX, strideX);
    int planes = n * outputH * outputW;
    constexpr int RH = 4, RW =4;
    constexpr int block_size = 256;
    constexpr int deal_rows = 32 * RH;
    dim3 blocks(block_size), grids(UP_DIVIDE(outputC, 4) / (block_size / 32), UP_DIVIDE(planes, (RH * 32)));
    int max_lines = UP_DIVIDE(deal_rows, outputW) + 1, max_points = max_lines;
    int share_size = max_lines * 4 * sizeof(int) + max_lines * 1 * sizeof(float*);
    ConvOpt1Impl<RH, RW><<<grids, blocks, share_size>>>(a, weight, b, n, inputC, h, w, outputC, kernelX, kernelY, dilateX, dilateY, strideX, strideY, padX, padY);
}