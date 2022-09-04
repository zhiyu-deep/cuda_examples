//
// Created by 63479 on 2022/8/10.
//

#include "scan_impl.h"
#include "stdio.h"

/**
 * @brief 线程级别scan.
 * @param t_val
 * @return
 */
__device__ float WarpScan(float t_val) {
    auto exchange = __shfl_sync(0xffffffff, t_val, 0, 2);
    if (threadIdx.x & 0x1) { t_val += exchange; }
    exchange = __shfl_sync(0xffffffff, t_val, 1, 4);
    if (threadIdx.x & 0x2) { t_val += exchange; }
    exchange = __shfl_sync(0xffffffff, t_val, 3, 8);
    if (threadIdx.x & 0x4) { t_val += exchange; }
    exchange = __shfl_sync(0xffffffff, t_val, 7, 16);
    if (threadIdx.x & 0x8) { t_val += exchange; }
    exchange = __shfl_sync(0xffffffff, t_val, 15, 32);
    if (threadIdx.x & 0x10) { t_val += exchange; }
    return t_val;
}

template<int warps>
__device__ float BlockScan(float t_val) {
    __shared__ float smem[warps * 32];

    // 1. 有数据的warp存储.
    smem[threadIdx.x] = t_val;
    __syncthreads();

    // 2. 每个warp委派线程去处理数据.
    int warp_id = threadIdx.x / 32;
    for (int i = 1; i < warps; i *= 2) {
        if (warp_id & i) {  // 1. 挑选warp.
            // 1. 定位到width.
            float exchange = smem[threadIdx.x / (i * 64) * (i * 64) + i * 32 - 1];
            t_val += exchange;
            smem[threadIdx.x] = t_val;
        }
        __syncthreads();
    }
    return t_val;
}

template<int warps>
__global__ void ScanBlockPackOneImpl(const float * __restrict__ input, float *output, int length) {
    input += blockDim.x * blockIdx.x;
    output += blockDim.x * blockIdx.x;

    // thread.
    auto t_ptr = input + threadIdx.x;
    float t_val = blockDim.x * blockIdx.x + threadIdx.x >= length ? 0.f : t_ptr[0];

    // warp.
    t_val = WarpScan(t_val);
    // block.
    t_val = BlockScan<warps>(t_val);
    // save.
    if (threadIdx.x < length) {
        output[threadIdx.x] = t_val;
    }
}

__global__ void GlobalScan(int i, float *output) {
    if (blockIdx.x & i) {
        auto exchange = output[blockIdx.x / (i * 2) * (i * 2) * (blockDim.x) + i * (blockDim.x) - 1];
        (output + blockIdx.x / (i * 2) * (i * 2) * (blockDim.x) + i * (blockDim.x))[threadIdx.x] += exchange;
    }
}

void Scan(const  float *input, float *output, int length) {
    if (length <= 32 * 32) {
        dim3 grid(1);
        if (32 >= length) {
            constexpr int warps = 1;
            dim3 block(32 * warps);
            ScanBlockPackOneImpl<warps><<<grid, block, sizeof(float) * 32 * warps>>>(input, output, length);
        }
#define Launch(warps)                        \
        else if (warps * 32 >= length) {     \
            dim3 block(32 * warps);          \
            ScanBlockPackOneImpl<warps><<<grid, block, sizeof(float) * 32 * warps>>>(input, output, length);  \
        }
        Launch(2)
        Launch(3)
        Launch(4)
        Launch(5)
        Launch(6)
        Launch(7)
        Launch(8)
        Launch(9)
        Launch(10)
        Launch(11)
        Launch(12)
        Launch(13)
        Launch(14)
        Launch(15)
        Launch(16)
        Launch(17)
        Launch(18)
        Launch(19)
        Launch(20)
        Launch(21)
        Launch(22)
        Launch(23)
        Launch(24)
        Launch(25)
        Launch(26)
        Launch(27)
        Launch(28)
        Launch(29)
        Launch(30)
        Launch(31)
        Launch(32)
    } else {
        constexpr int warps = 32;
        dim3 block(warps * 32);
        int blocks = (length + warps * 32 - 1) / (warps * 32);
        dim3 grid((length + warps * 32 - 1) / (warps * 32));
        ScanBlockPackOneImpl<warps><<<grid, block, sizeof(float) * 32 * warps>>>(input, output, length);
        for (int i = 1; i < blocks; i*=2) {
            GlobalScan<<<grid, blocks>>>(i, output);
        }
    }
    cudaDeviceSynchronize();
}
