//
// Created by 63479 on 2022/10/22.
//

#ifndef ARM64_TEST_CONVUTILS_CUH
#define ARM64_TEST_CONVUTILS_CUH

//__host__ int CalculateOutputLength(int l, int k, int d, int p, int s);

__forceinline__ __device__ __host__ int CalculateOutputLength(int l, int k, int d, int p, int s) {
    return ((l + 2 * p) - ((k - 1) * d + 1)) / s + 1;
}

__forceinline__ __device__ __host__ bool CheckParamsLegal(int l, int k, int d, int p, int s) {
    if (k < 1) {
        return false;
    }
    if ((1 + (k - 1) * d) <= p) {
        return false;
    }
    return true;
}

#endif //ARM64_TEST_CONVUTILS_CUH
