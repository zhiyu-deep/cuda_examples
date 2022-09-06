//
// Created by 63479 on 2022/9/5.
//

#include "assert.h"
#include "transpose.cuh"

__global__ void TransposeImpl(float *a, float *b, int m, int n) {
    assert(m > 0 && n > 0);

    __shared__ float smem[32 * 32 + 32];

    int blockX = blockIdx.x * 32, blockY = blockIdx.y * 32;
    float data = 0.f;
    if (blockX + threadIdx.x < n && blockY + threadIdx.y < m) {
        data = a[blockY * n + blockX + threadIdx.y * n + threadIdx.x];
    }
    smem[threadIdx.y * 33 + threadIdx.x] = data;
    __syncthreads();

    blockX = blockIdx.y * 32, blockY = blockIdx.x * 32;

    if (blockX + threadIdx.x < m && blockY + threadIdx.y < n) {
        b[blockY * m + blockX + threadIdx.x + threadIdx.y * m] = smem[threadIdx.x * 33 + threadIdx.y];
    }
}

void Transpose(float *a, float *b, int m, int n) {
    dim3 block(32, 32);
    dim3 grid((n + 31) / 32, (m + 31) / 32);
    TransposeImpl<<<grid, block>>>(a, b, m, n);
}
