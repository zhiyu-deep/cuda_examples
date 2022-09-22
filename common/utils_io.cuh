//
// Created by 63479 on 2022/9/21.
//

#ifndef ARM64_TEST_UTILS_IO_CUH
#define ARM64_TEST_UTILS_IO_CUH

#include "stdio.h"
#include "utils_type.h"
#include "utils_assert.cuh"

template<typename T, int length, int vec_length>
struct Vector {
    static constexpr int length_four_times  = conditional_num<length % 4 == 0, length / 4, (length + 3) / 4 - 1>::value;
    static constexpr int length_four_start  = 0;
    static constexpr int length_three_times = conditional_num<(length - length_four_times * 4) >= 3, 1, 0>::value;
    static constexpr int length_three_start = length_four_times * 4;
    static constexpr int length_two_times   = conditional_num<(length - length_four_times * 4 - length_three_times * 3) >= 2, 1, 0>::value;
    static constexpr int length_two_start   = length_four_times * 4 + length_three_times * 3;
    static constexpr int length_one_times   = length - length_four_times * 4 - length_three_times * 3 - length_two_times * 2;
    static constexpr int length_one_start   = length_four_times * 4 + length_three_times * 3 + length_two_times * 2;

    __device__ void operator<<(const T *ptr) {
#pragma unroll
        for (int start = length_four_start, i = 0; i < length_four_times; i++, start += 4) {
            ((float4*)(data + start))[i] = ((float4*)(ptr + start))[i];
        }
#pragma unroll
        for (int start = length_three_start, i = 0; i < length_three_times; i++, start += 3) {
            ((float3*)(data + start))[i] = ((float3*)(ptr + start))[i];
        }
#pragma unroll
        for (int start = length_two_start, i = 0; i < length_two_times; i++, start += 2) {
            ((float2*)(data + start))[i] = ((float2*)(ptr + start))[i];
        }
#pragma unroll
        for (int start = length_one_start, i = 0; i < length_one_times; i++, start += 1) {
            ((float1*)(data + start))[i] = ((float1*)(ptr + start))[i];
        }
    }

    __device__ void operator>>(T *ptr) {
#pragma unroll
        for (int start = length_four_start, i = 0; i < length_four_times; i++, start += 4) {
            ((float4*)(ptr + start))[i] = ((float4*)(data + start))[i];
        }
#pragma unroll
        for (int start = length_three_start, i = 0; i < length_three_times; i++, start += 3) {
            ((float3*)(ptr + start))[i] = ((float3*)(data + start))[i];
        }
#pragma unroll
        for (int start = length_two_start, i = 0; i < length_two_times; i++, start += 2) {
            ((float2*)(ptr + start))[i] = ((float2*)(data + start))[i];
        }
#pragma unroll
        for (int start = length_one_start, i = 0; i < length_one_times; i++, start += 1) {
            ((float1*)(ptr + start))[i] = ((float1*)(data + start))[i];
        }
    }

    __device__ T& operator[](int i) {
        return data[i];
    }

    __device__ const T& operator[](int i) const {
        return data[i];
    }

    T data[length];
};

template<typename T, int length>
struct Vector<T, length, 4> {
    static constexpr int times = length / 4;

    __device__ void operator<<(const T *ptr) {
#pragma unroll
        for (int i = 0; i < times; i++) {
            ((float4*)data)[i] = ((float4*)ptr)[i];
        }
    }

    __device__ void operator>>(T *ptr) {
#pragma unroll
        for (int i = 0; i < times; i++) {
            ((float4*)ptr)[i] = ((float4*)data)[i];
        }
    }

    __device__ T& operator[](int i) {
        return data[i];
    }

    __device__ const T& operator[](int i) const {
        return data[i];
    }

    T data[length];
};

template<typename T, int length>
struct Vector<T, length, 2> {
    static constexpr int times = length / 2;

    __device__ void operator<<(const T *ptr) {
#pragma unroll
        for (int i = 0; i < times; i++) {
            ((float2*)data)[i] = ((float2*)ptr)[i];
        }
    }

    __device__ void operator>>(T *ptr) {
#pragma unroll
        for (int i = 0; i < times; i++) {
            ((float2*)ptr)[i] = ((float2*)data)[i];
        }
    }

    __device__ T& operator[](int i) {
        return data[i];
    }

    __device__ const T& operator[](int i) const {
        return data[i];
    }

    T data[length];
};

template<typename T, int length>
struct Vector<T, length, 3> {
    static constexpr int times = length / 3;

    __device__ void operator<<(const T *ptr) {
#pragma unroll
        for (int i = 0; i < times; i++) {
            ((float3*)data)[i] = ((float3*)ptr)[i];
        }
    }

    __device__ void operator>>(T *ptr) {
#pragma unroll
        for (int i = 0; i < times; i++) {
            ((float3*)ptr)[i] = ((float3*)data)[i];
        }
    }

    __device__ T& operator[](int i) {
        return data[i];
    }

    __device__ const T& operator[](int i) const {
        return data[i];
    }

    T data[length];
};

#endif //ARM64_TEST_UTILS_IO_CUH
