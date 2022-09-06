//
// Created by 63479 on 2022/9/5.
//

#include "stdio.h"
#include "MyGemmV3.cuh"
#include "transpose.cuh"

template<int TileH, int TileW, int TileK>
__global__ void MyGemmGlobalImplV3(float *a, float *b, float *c,
                                   int M, int N, int K) {
    // todo: 务必按照tileK padding.
    constexpr int AL1CacheY = TileH * 16 * 2, AL1CacheX = TileK, AL1CacheYThreads = AL1CacheY / 4, AL1CacheXThreads = AL1CacheX, AL1CacheXHaveThreads = 256 / AL1CacheYThreads;
    // blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y.
    int inWarpId = threadIdx.x % 32, warpId = threadIdx.x / 32;
    int inWarpIdX = inWarpId % 8, inWarpIdY = inWarpId / 8, warpIdX = warpId % 2, warpIdY = warpId / 2;
    int inBlockIdx = warpIdX * 8 + inWarpIdX, inBlockIdY = warpIdY * 4 + inWarpIdY;
    assert(TileK % 4 == 0);
    assert(TileK <= 8);
    assert(TileW % 4 == 0);
    assert(TileH == TileW);
    assert(TileH <= 32);

    float a_cache[TileH];
    float b_cache[TileW];
    float output_cache[2][2][TileH][TileW] = {0};

    __shared__ float a_smem[TileK * 16 * TileH * 2];
    __shared__ float b_smem[TileK * 16 * TileW * 2];

    // todo: 切换到L1 cache(block)(大cache + packs).
    // todo: 1. 定位到cache内容.
    // todo: block a ptr.
    float *a_ptr = a + blockIdx.y * (TileH * 16 * 2);
    // todo: block b ptr.
    float *b_ptr = b + blockIdx.x * (TileW * 16 * 2);
    // todo: block c ptr.
    float *c_ptr = c + blockIdx.y * (TileH * 16 * 2) * N + blockIdx.x * (16 * TileW * 2);

    int tIdY = threadIdx.x % AL1CacheYThreads;
#pragma unroll
    for (int k = 0; k < K; k+=TileK) {  // a已经是纵向分布.
        for (int tIdX = (threadIdx.x / AL1CacheYThreads); tIdX < AL1CacheXThreads; tIdX += AL1CacheXHaveThreads) {
            auto data = ((float4*)(a_ptr + k * M + tIdY * 4 + tIdX * M))[0];
            ((float4*)(a_smem + tIdY * 4 + tIdX * AL1CacheY))[0] = data;
            data = ((float4*)(b_ptr + k * N + tIdY * 4 + tIdX * N))[0];
            ((float4*)(b_smem + tIdX * AL1CacheY + tIdY * 4))[0] = data;
        }

        __syncthreads();
        // todo: register level(thread)
#pragma unroll
        for (int x = 0; x < 2; x++) {
#pragma unroll
            for (int y = 0; y < 2; y++) {
#pragma unroll
                for (int tk = 0; tk < TileK; tk++) {
#pragma unroll
                    for (int i = 0; i < TileH; i+=4) {
                        auto a_data = ((float4*)(a_smem + TileH * 16 * x + tk * AL1CacheY + inBlockIdY * TileH + i))[0];
                        ((float4*)(a_cache + i))[0] = a_data;
                    }
#pragma unroll
                    for (int j = 0; j < TileW; j+=4) {
                        auto b_data = ((float4*)(b_smem + TileW * 16 * y + tk * AL1CacheY + inBlockIdx * TileW + j))[0];
                        ((float4*)(b_cache + j))[0] = b_data;
                    }
#pragma unroll
                    for (int i = 0; i < TileH; i++) {
#pragma unroll
                        for (int j = 0; j < TileW; j++) {
                            output_cache[x][y][i][j] += a_cache[i] * b_cache[j];
                        }
                    }
                }
            }
        }
    }

    // todo: 让每个线程处理自己的.
#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
#pragma unroll
            for (int m = 0; m < TileH; m++) {
#pragma unroll
                for (int n = 0; n < TileW; n+=4) {
                    ((float4*)(c_ptr + i * TileH * 16 * N + j * TileW * 16 +
                    inBlockIdY * TileH * N + inBlockIdx * TileW +
                    m * N + n))[0] = ((float4*)(output_cache[i][j][m] + n))[0];
                }
            }
        }
    }
}

void MyGemmGlobalV3(float *a, float *b, float *c,
                    int M, int N, int K) {
    Transpose(a, a, M, K);

    constexpr int tileH = 4, tileW = 4, tileK = 4;
    dim3 block(256);
    dim3 grid(N / (16 * tileW * 2), M / (16 * tileH * 2));
    MyGemmGlobalImplV3<tileH, tileW, tileK><<<grid, block>>>(a, b, c, M, N, K);
}