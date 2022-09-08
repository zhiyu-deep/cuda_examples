//
// Created by 63479 on 2022/9/5.
//

#include "MyGemmV5.cuh"
#include "MyGemmShareImpl.cuh"

void MyGemmGlobalV5Repeats(float *a, float *b, float *c,
                           int M, int N, int K) {
    constexpr int tileH = 4, tileW = 4, tileK = 4, repeatsY = 2, repeatsX = 2;
    dim3 block(256);
    dim3 grid(N / (16 * tileW * repeatsX), M / (16 * tileH * repeatsY));
    MyGemmGlobalImplV5<tileH, tileW, tileK, 2, 2><<<grid, block>>>(a, b, c, M, N, K);
}

void MyGemmGlobalV5NoRepeats(float *a, float *b, float *c,
                             int M, int N, int K) {
    constexpr int tileH = 4, tileW = 4, tileK = 4, repeatsY = 1, repeatsX = 1;
    dim3 block(256);
    dim3 grid(N / (16 * tileW * repeatsX), M / (16 * tileH * repeatsY));
    MyGemmGlobalImplV5<tileH, tileW, tileK, repeatsY, repeatsX><<<grid, block>>>(a, b, c, M, N, K);
}