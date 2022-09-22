//
// Created by 63479 on 2022/8/10.
//

#include "utils_policy.h"
#include "utils_io.cuh"
#include "scan_impl.h"
#include "utils_assert.cuh"

template<int length>
struct WarpScanPolicyIntern {
    struct Vec1 : public ChainPolicy<length % 2 == 0, Vec1, Vec1> {
        static constexpr int thread_nums = 16;
        static constexpr int thread_vec = 1;
        static constexpr int max_threads = 32;
    };

    struct Vec2 : public ChainPolicy<length % 2 == 0, Vec2, Vec1> {
        static constexpr int thread_nums = 16;
        static constexpr int thread_vec = 2;
        static constexpr int max_threads = 32;
    };

    struct Vec3 : public ChainPolicy<length % 3 == 0, Vec3, Vec2> {
        static constexpr int thread_nums = 16;
        static constexpr int thread_vec = 3;
        static constexpr int max_threads = 32;
    };

    struct Vec4 : public ChainPolicy<length % 4 == 0, Vec4, Vec3> {
        static constexpr int thread_nums = 16;
        static constexpr int thread_vec = 4;
        static constexpr int max_threads = 32;
    };

    static constexpr int max_length = 32 * 1 * 16;
    struct Vec : public ChainPolicy<length <= max_length, Vec4, InvalidPolicy> {};

    using PolicyAgent = Vec4;
};

/**
 * @brief
 * @param input
 * @param output
 */
template<int thread_nums, int thread_vec, int threads>
__global__ void WarpScan(const float *input, float *output, int length) {
    assert(length % thread_vec == 0);
    assert(thread_nums % thread_vec == 0);

    Vector<float, thread_nums, thread_vec> registers;
    registers << (input + threadIdx.x * thread_nums);

    // todo: thread scan, save to register.
#pragma unroll
    for (int i = 1; i < thread_nums; i++) {
        registers[i] += registers[i - 1];
    }

    // todo: warp scan.
    float exchange = registers[thread_nums - 1];
#pragma unroll
    for (int i = 2; i < 2 * threads; i *= 2) {
        exchange = __shfl_sync(0xffffffff, exchange, i / 2 - 1, i);
        if ((threadIdx.x % i) >= (i / 2)) {
#pragma unroll
            for (int j = 0; j < thread_nums; j++) {
                registers[j] += exchange;
            }
        }
        exchange = registers[thread_nums - 1];
    }

    // todo: write to global.
    registers >> (output + threadIdx.x * thread_nums);
}

void Scan(const  float *input, float *output, int length) {
    // warp scan.
    {
        constexpr int thread_nums = 16;
        constexpr int max_threads = 32;
        if (thread_nums * max_threads >= length) {
            if (length % 4 == 0) {
                dim3 block(length / thread_nums);
                dim3 grid(1);
                WarpScan<thread_nums, 4, max_threads><<<grid, block>>>(input, output, length);
            } else if (length % 3 == 0) {
                dim3 block(length / thread_nums);
                dim3 grid(1);
                WarpScan<thread_nums, 3, max_threads><<<grid, block>>>(input, output, length);
            } else if (length % 2 == 0) {
                dim3 block(length / thread_nums);
                dim3 grid(1);
                WarpScan<thread_nums, 2, max_threads><<<grid, block>>>(input, output, length);
            } else {
                dim3 block(length / thread_nums);
                dim3 grid(1);
                WarpScan<thread_nums, 1, max_threads><<<grid, block>>>(input, output, length);
            }
        }
    }
}
