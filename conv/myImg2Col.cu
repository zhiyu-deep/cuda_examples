//
// Created by 63479 on 2022/9/1.
//

#include "myImg2Col.cuh"

__global__ void MyImg2ColConvNaiveImpl(
        const float *a, const float *weight, float *b,
        int n, int c, int h, int w, int outputc, int outputh, int outputw,
        int kernelX, int kernelY, int dilateX, int dilateY, int strideX, int strideY, int padX, int padY) {
    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    // 点在output中位置.
    int outputXId = tId % outputw, outputNYId = tId / outputw, outputYId = outputNYId % outputh, outputNId = outputNYId / outputh;
    // 点在input中位置.
    int inputXId = outputXId * strideX, inputYId = outputYId * strideY;
    // 位于kernel中上界和下界.
    int validInputTopId = max(0, (padY - inputYId) / dilateY), validInputLeftId = max(0, (padX - inputXId) / dilateX);
    int validInputBottomId = min((padY + h - 1 - inputYId) / dilateY, kernelY), validInputRightId = min((padX + w - 1 - inputXId) / dilateX, kernelX);
    for (int i = validInputTopId; i <= validInputBottomId; i++) {
        for (int j = validInputLeftId; j <= validInputRightId; j++) {
            for (int k = 0; k < c; k++) {
                b[(i * kernelX + j) * c + k] = a[(inputYId + i * dilateY) * (w * c) + (inputXId + j * dilateX) * c + k];
            }
        }
    }
}

void MyImg2ColConvNaive(const float *a, const float *weight, float *b,
                        int n, int c, int h, int w, int outputc,
                        int kernelX, int kernelY, int dilateX, int dilateY, int strideX, int strideY, int padX, int padY) {
    // 0. prepare.
    int dilateKernelX = 1 + (kernelX - 1) * dilateX, dilateKernelY = 1 + (kernelY - 1) * dilateY;
    int outputh = (h + 2 * padY - dilateKernelY) / strideY + 1, outputw = (w + 2 * padX - dilateKernelX) / strideX + 1;
    // 1. 将数据转移到cache.
    dim3 block(256);
    dim3 grid(n * outputh * outputw / 256);
    MyImg2ColConvNaiveImpl<<<grid, block>>>(a, weight, b, n, c, h, w, outputc, outputh, outputw,
                                            kernelX, kernelY, dilateX, dilateY, strideX, strideY, padX, padY);
    // 2. cache 和 w 进行gemm.
}