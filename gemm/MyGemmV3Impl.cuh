//
// Created by 63479 on 2022/9/8.
//

#ifndef ARM64_TEST_MYGEMMV3IMPL_CUH
#define ARM64_TEST_MYGEMMV3IMPL_CUH


template<int TileH, int TileW, int TileK, int repeatsY, int repeatsX>
__global__ void MyGemmGlobalImplV3(float *a, float *b, float *c,
                                   int M, int N, int K) {
    // todo: 务必按照tileK padding.
    constexpr int AL1CacheY = TileH * 16 * repeatsY, AL1CacheX = TileK, AL1CacheYThreads = AL1CacheY / 4, AL1CacheXThreads = AL1CacheX, AL1CacheXHaveThreads = 256 / AL1CacheYThreads;
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
    float output_cache[repeatsY][repeatsX][TileH][TileW] = {0};

    __shared__ float a_smem[TileK * 16 * TileH * repeatsY];
    __shared__ float b_smem[TileK * 16 * TileW * repeatsX];

    // todo: 切换到L1 cache(block)(大cache + packs).
    // todo: 1. 定位到cache内容.
    // todo: block a ptr.
    float *a_ptr = a + blockIdx.y * (TileH * 16 * repeatsY);
    // todo: block b ptr.
    float *b_ptr = b + blockIdx.x * (TileW * 16 * repeatsX);
    // todo: block c ptr.
    float *c_ptr = c + blockIdx.y * (TileH * 16 * repeatsY) * N + blockIdx.x * (16 * TileW * repeatsX);

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
        for (int x = 0; x < repeatsY; x++) {
#pragma unroll
            for (int y = 0; y < repeatsX; y++) {
#pragma unroll
                for (int tk = 0; tk < TileK; tk++) {
#pragma unroll
                    for (int i = 0; i < TileH; i+=4) {
                        ((float4*)(a_cache + i))[0] = ((float4*)(a_smem + TileH * 16 * x + tk * AL1CacheY + inBlockIdY * TileH + i))[0];
                    }
#pragma unroll
                    for (int j = 0; j < TileW; j+=4) {
                        ((float4*)(b_cache + j))[0] = ((float4*)(b_smem + TileW * 16 * y + tk * AL1CacheY + inBlockIdx * TileW + j))[0];
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
    for (int i = 0; i < repeatsY; i++) {
#pragma unroll
        for (int j = 0; j < repeatsX; j++) {
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

#endif //ARM64_TEST_MYGEMMV3IMPL_CUH
