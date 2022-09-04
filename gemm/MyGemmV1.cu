//
// Created by 63479 on 2022/8/28.
//
#include "iostream"
#include "stdio.h"
#include "MyGemmV1.h"
#include "common.cuh"

template<int TileH, int TileW, int TileK>
__global__ void MyGemmGlobalImplV1(float *a, float *b, float *c,
                                   int M, int N, int K) {
    // blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y.
    int inWarpId = threadIdx.x % 32, warpId = threadIdx.x / 32;
    int inWarpIdX = inWarpId % 8, inWarpIdY = inWarpId / 8, warpIdX = warpId % 2, warpIdY = warpId / 2;
    int inBlockIdx = warpIdX * 8 + inWarpIdX, inBlockIdY = warpIdY * 4 + inWarpIdY;

    float a_cache[TileH][TileK];
    float b_cache[TileK][TileW];
    float output_cache[TileH][TileW] = {0};

    // todo: block a ptr.
    float *a_ptr = a + blockIdx.y * (TileH * 16) * K;
    // todo: block b ptr.
    float *b_ptr = b + blockIdx.x * (TileW * 16);
    // todo: block c ptr.
    float *c_ptr = c + blockIdx.y * (TileH * 16) * N + blockIdx.x * (16 * TileW) + inBlockIdY * TileH * N + inBlockIdx * TileW;
    for (int kId = 0; kId < K; kId += TileK, a_ptr += TileK, b_ptr += TileK * N) {
        // todo: 利用block中的id, 在block ptr中索引得到数据.
#pragma unroll
        for (int i = 0; i < TileH; i++) {
#pragma unroll
            for (int j = 0; j < TileK; j++) {
                a_cache[i][j] = a_ptr[(inBlockIdY * TileH + i) * K + j];
            }
        }
#pragma unroll
        for (int i = 0; i < TileK; i++) {
            for (int j = 0; j < TileW; j++) {
                b_cache[i][j] = b_ptr[i * N + inBlockIdx * TileW + j];
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
//    printf("%f, %f", a_cache[0][0], b_cache[0][0]);
#pragma unroll
    for (int i = 0; i < TileH; i++) {
#pragma unroll
        for (int j = 0; j < TileW; j++) {
            c_ptr[i * N + j] = output_cache[i][j];
        }
    }
}

void MyGemmGlobalV1(float *a, float *b, float *c,
                    int M, int N, int K) {
    // todo: 1. 明确形状的情况下, 完成参数的最优化, 完成resize操作(因为cpu并行的时候能知道tid, cuda只能在kernel内部计算tid).
    constexpr int tileH = 4, tileW = 4, tileK = 4;
    dim3 block(256);
    dim3 grid(N / (16 * tileW), M / (16 * tileH));
    MyGemmGlobalImplV1<tileH, tileW, tileK><<<grid, block>>>(a, b, c, M, N, K);
}