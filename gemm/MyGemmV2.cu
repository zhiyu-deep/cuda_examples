//
// Created by 63479 on 2022/9/4.
//

#include "MyGemmV2.cuh"

template<int TileH, int TileW, int TileK>
__global__ void MyGemmGlobalImplV2(float *a, float *b, float *c,
                                   int M, int N, int K) {
    // blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y.
    int inWarpId = threadIdx.x % 32, warpId = threadIdx.x / 32;
    int inWarpIdX = inWarpId % 8, inWarpIdY = inWarpId / 8, warpIdX = warpId % 2, warpIdY = warpId / 2;
    int inBlockIdx = warpIdX * 8 + inWarpIdX, inBlockIdY = warpIdY * 4 + inWarpIdY;

    assert(TileK % 4 == 0);
    assert(TileW % 4 == 0);

    float a_cache[TileH][TileK];
    float b_cache[TileK][TileW];
    float output_cache[TileH][TileW] = {0};

    // todo: thread a ptr.
    float *a_ptr = a + blockIdx.y * (TileH * 16) * K + inBlockIdY * TileH * K;
    // todo: thread b ptr.
    float *b_ptr = b + blockIdx.x * (TileW * 16) + inBlockIdx * TileW;
    // todo: thread c ptr.
    float *c_ptr = c + blockIdx.y * (TileH * 16) * N + blockIdx.x * (16 * TileW) + inBlockIdY * TileH * N + inBlockIdx * TileW;
    // todo: part thread a ptr, part thread b ptr.
    for (int kId = 0; kId < K; kId += TileK, a_ptr += TileK, b_ptr += TileK * N) {
#pragma unroll
        for (int i = 0; i < TileH; i++) {
#pragma unroll
            for (int j = 0; j < TileK; j += 4) {
                auto data = ((float4*)(a_ptr + i * K + j))[0];
                a_cache[i][j] = data.x;
                a_cache[i][j + 1] = data.y;
                a_cache[i][j + 2] = data.z;
                a_cache[i][j + 3] = data.w;
            }
        }
#pragma unroll
        for (int i = 0; i < TileK; i++) {
            for (int j = 0; j < TileW; j += 4) {
                auto data = ((float4*)(b_ptr + i * N + j))[0];
                b_cache[i][j] = data.x;
                b_cache[i][j + 1] = data.y;
                b_cache[i][j + 2] = data.z;
                b_cache[i][j + 3] = data.w;
            }
        }
        // todo: 线程级别处理.
#pragma unroll
        for (int k = 0; k < TileK; k++) {
#pragma unroll
            for (int i = 0; i < TileH; i++) {
#pragma unroll
                for (int j = 0; j < TileW; j++) {
                    output_cache[i][j] += a_cache[i][k] * b_cache[k][j];
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < TileH; i++) {
#pragma unroll
        for (int j = 0; j < TileW; j+=4) {
            ((float4*)(c_ptr + i * N + j))[0] = ((float4*)(output_cache[i] + j))[0];
        }
    }
}

void MyGemmGlobalV2(float *a, float *b, float *c,
                    int M, int N, int K) {
    constexpr int tileH = 4, tileW = 4, tileK = 4;
    dim3 block(256);
    dim3 grid(N / (16 * tileW), M / (16 * tileH));
    MyGemmGlobalImplV2<tileH, tileW, tileK><<<grid, block>>>(a, b, c, M, N, K);
}