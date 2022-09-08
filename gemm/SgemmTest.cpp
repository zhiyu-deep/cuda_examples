//
// Created by 63479 on 2022/8/25.
//

#ifndef ARM64_TEST_SGEMMTEST_H
#define ARM64_TEST_SGEMMTEST_H

#include "register.h"
#include "common.cuh"
#include "transpose.cuh"

#include "MyGemmV1.h"
#include "MyGemmV2.cuh"
#include "MyGemmV3.cuh"
#include "MyGemmV4.cuh"
#include "MyGemmV5.cuh"
#include "MyGemmV6.cuh"
#include "nicolaswildeV1.cuh"
#include "nicolaswildeV2.cuh"
#include "nicolaswildeV3.cuh"

class GemmRefer {
public:
    GemmRefer(float **a, float **b, float **c, int m, int n, int k)
        : a_(a),
          b_(b),
          c_(c),
          m_(m),
          n_(n),
          k_(k) {}

    void operator() () const {
        for (int i = 0; i < m_; i++) {
            for (int j = 0; j < n_; j++) {
                (*c_)[i * n_ + j] = 0.f;
                for (int k = 0; k < k_; k++) {
                    (*c_)[i * n_ + j] += (*a_)[i * k_ + k] * (*b_)[k * n_ + j];
                }
            }
        }
    }

    float **a_, **b_, **c_;
    int m_, n_, k_;
};

class GemmTest : public Test {
public:
    explicit GemmTest(const std::string &name) : Test(name) {}

    virtual ErrorCode run() const {
        if (SGemmTest()) {
            return kNoError;
        }
        return kError;
    }

    virtual ~GemmTest() = default;

private:
    bool SGemmTest() const {
        int m = 1024, n = 896, k = 4;
        float *a, *b, *c, *d_a, *d_b, *d_c;
        float *refer_c;

        {
            // prepare input.
            auto callBackA = InputMallocAndCpy(&a, &d_a, m * k);
            auto callBackB = InputMallocAndCpy(&b, &d_b, n * k);
            // prepare output.
            auto callBack = OutputMallocAndDelayCpy(&c, &refer_c, &d_c, m * n);
            TestLaunchKernel(MyGemmGlobalV1(tile), MyGemmGlobalV1(d_a, d_b, d_c, m, n, k), GemmRefer(&a, &b, &refer_c, m, n, k)());
            WarmupKernel(MyGemmGlobalV1(d_a, d_b, d_c, m, n, k));
            ProfileKernel(MyGemmGlobalV1(tile), MyGemmGlobalV1(d_a, d_b, d_c, m, n, k), RepeatTimes);
        }
        {
            // prepare input.
            auto callBackA = InputMallocAndCpy(&a, &d_a, m * k);
            auto callBackB = InputMallocAndCpy(&b, &d_b, n * k);
            // prepare output.
            auto callBack = OutputMallocAndDelayCpy(&c, &refer_c, &d_c, m * n);
            TestLaunchKernel(MyGemmGlobalV2(tile + vec), MyGemmGlobalV2(d_a, d_b, d_c, m, n, k), GemmRefer(&a, &b, &refer_c, m, n, k)());
            WarmupKernel(MyGemmGlobalV2(d_a, d_b, d_c, m, n, k));
            ProfileKernel(MyGemmGlobalV2(tile + vec), MyGemmGlobalV2(d_a, d_b, d_c, m, n, k), RepeatTimes);
        }
        {
            // prepare input.
            auto callBackA = InputMallocAndCpy(&a, &d_a, m * k);
            auto callBackB = InputMallocAndCpy(&b, &d_b, n * k);

            float *d_d = nullptr;
            cudaMalloc(reinterpret_cast<void **>(&d_d), sizeof(float) * m * k);
            Transpose(d_a, d_d, m, k);

            // prepare output.
            auto callBack = OutputMallocAndDelayCpy(&c, &refer_c, &d_c, m * n);
            TestLaunchKernel(MyGemmGlobalV3(share)Repeats, MyGemmGlobalV3Repeats(d_d, d_b, d_c, m, n, k), GemmRefer(&a, &b, &refer_c, m, n, k)());
            WarmupKernel(MyGemmGlobalV3Repeats(d_d, d_b, d_c, m, n, k));
            ProfileKernel(MyGemmGlobalV3(share)Repeats, MyGemmGlobalV3Repeats(d_d, d_b, d_c, m, n, k), (RepeatTimes));

            cudaFree(d_d);
        }
        {
            // prepare input.
            auto callBackA = InputMallocAndCpy(&a, &d_a, m * k);
            auto callBackB = InputMallocAndCpy(&b, &d_b, n * k);

            float *d_d = nullptr;
            cudaMalloc(reinterpret_cast<void **>(&d_d), sizeof(float) * m * k);
            Transpose(d_a, d_d, m, k);

            // prepare output.
            auto callBack = OutputMallocAndDelayCpy(&c, &refer_c, &d_c, m * n);
            TestLaunchKernel(MyGemmGlobalV3(share)NoRepeats, MyGemmGlobalV3NoRepeats(d_d, d_b, d_c, m, n, k), GemmRefer(&a, &b, &refer_c, m, n, k)());
            WarmupKernel(MyGemmGlobalV3NoRepeats(d_d, d_b, d_c, m, n, k));
            ProfileKernel(MyGemmGlobalV3(share)NoRepeats, MyGemmGlobalV3NoRepeats(d_d, d_b, d_c, m, n, k), (RepeatTimes));

            cudaFree(d_d);
        }
        {
            // prepare input.
            auto callBackA = InputMallocAndCpy(&a, &d_a, m * k);
            auto callBackB = InputMallocAndCpy(&b, &d_b, n * k);

            float *d_d = nullptr;
            cudaMalloc(reinterpret_cast<void **>(&d_d), sizeof(float) * m * k);
            Transpose(d_a, d_d, m, k);

            // prepare output.
            auto callBack = OutputMallocAndDelayCpy(&c, &refer_c, &d_c, m * n);
            TestLaunchKernel(MyGemmGlobalV4(share + share prefetch)Repeats, MyGemmGlobalV4Repeats(d_d, d_b, d_c, m, n, k), GemmRefer(&a, &b, &refer_c, m, n, k)());
            WarmupKernel(MyGemmGlobalV4Repeats(d_d, d_b, d_c, m, n, k));
            ProfileKernel(MyGemmGlobalV4(share + share prefetch)Repeats, MyGemmGlobalV4Repeats(d_d, d_b, d_c, m, n, k), (RepeatTimes));

            cudaFree(d_d);
        }
        {
            // prepare input.
            auto callBackA = InputMallocAndCpy(&a, &d_a, m * k);
            auto callBackB = InputMallocAndCpy(&b, &d_b, n * k);

            float *d_d = nullptr;
            cudaMalloc(reinterpret_cast<void **>(&d_d), sizeof(float) * m * k);
            Transpose(d_a, d_d, m, k);

            // prepare output.
            auto callBack = OutputMallocAndDelayCpy(&c, &refer_c, &d_c, m * n);
            TestLaunchKernel(MyGemmGlobalV4(share + share prefetch)NoRepeats, MyGemmGlobalV4NoRepeats(d_d, d_b, d_c, m, n, k), GemmRefer(&a, &b, &refer_c, m, n, k)());
            WarmupKernel(MyGemmGlobalV4NoRepeats(d_d, d_b, d_c, m, n, k));
            ProfileKernel(MyGemmGlobalV4(share + share prefetch)NoRepeats, MyGemmGlobalV4NoRepeats(d_d, d_b, d_c, m, n, k), (RepeatTimes));

            cudaFree(d_d);
        }
        {
            // prepare input.
            auto callBackA = InputMallocAndCpy(&a, &d_a, m * k);
            auto callBackB = InputMallocAndCpy(&b, &d_b, n * k);

            float *d_d = nullptr;
            cudaMalloc(reinterpret_cast<void **>(&d_d), sizeof(float) * m * k);
            Transpose(d_a, d_d, m, k);

            // prepare output.
            auto callBack = OutputMallocAndDelayCpy(&c, &refer_c, &d_c, m * n);
            TestLaunchKernel(MyGemmGlobalV5(share + global prefetch)Repeats, MyGemmGlobalV5Repeats(d_d, d_b, d_c, m, n, k), GemmRefer(&a, &b, &refer_c, m, n, k)());
            WarmupKernel(MyGemmGlobalV5Repeats(d_d, d_b, d_c, m, n, k));
            ProfileKernel(MyGemmGlobalV5(share + global prefetch)Repeats, MyGemmGlobalV5Repeats(d_d, d_b, d_c, m, n, k), (RepeatTimes));

            cudaFree(d_d);
        }
        {
            // prepare input.
            auto callBackA = InputMallocAndCpy(&a, &d_a, m * k);
            auto callBackB = InputMallocAndCpy(&b, &d_b, n * k);

            float *d_d = nullptr;
            cudaMalloc(reinterpret_cast<void **>(&d_d), sizeof(float) * m * k);
            Transpose(d_a, d_d, m, k);

            // prepare output.
            auto callBack = OutputMallocAndDelayCpy(&c, &refer_c, &d_c, m * n);
            TestLaunchKernel(MyGemmGlobalV5(share + global prefetch)NoRepeats, MyGemmGlobalV5NoRepeats(d_d, d_b, d_c, m, n, k), GemmRefer(&a, &b, &refer_c, m, n, k)());
            WarmupKernel(MyGemmGlobalV5NoRepeats(d_d, d_b, d_c, m, n, k));
            ProfileKernel(MyGemmGlobalV5(share + global prefetch)NoRepeats, MyGemmGlobalV5NoRepeats(d_d, d_b, d_c, m, n, k), (RepeatTimes));

            cudaFree(d_d);
        }
        {
            // prepare input.
            auto callBackA = InputMallocAndCpy(&a, &d_a, m * k);
            auto callBackB = InputMallocAndCpy(&b, &d_b, n * k);

            float *d_d = nullptr;
            cudaMalloc(reinterpret_cast<void **>(&d_d), sizeof(float) * m * k);
            Transpose(d_a, d_d, m, k);

            // prepare output.
            auto callBack = OutputMallocAndDelayCpy(&c, &refer_c, &d_c, m * n);
            TestLaunchKernel(MyGemmGlobalV6(share + global prefetch + share prefetch)Repeats, MyGemmGlobalV6Repeats(d_d, d_b, d_c, m, n, k), GemmRefer(&a, &b, &refer_c, m, n, k)());
            WarmupKernel(MyGemmGlobalV6Repeats(d_d, d_b, d_c, m, n, k));
            ProfileKernel(MyGemmGlobalV6(share + global prefetch + share prefetch)Repeats, MyGemmGlobalV6Repeats(d_d, d_b, d_c, m, n, k), (RepeatTimes));

            cudaFree(d_d);
        }
        {
            // prepare input.
            auto callBackA = InputMallocAndCpy(&a, &d_a, m * k);
            auto callBackB = InputMallocAndCpy(&b, &d_b, n * k);

            float *d_d = nullptr;
            cudaMalloc(reinterpret_cast<void **>(&d_d), sizeof(float) * m * k);
            Transpose(d_a, d_d, m, k);

            // prepare output.
            auto callBack = OutputMallocAndDelayCpy(&c, &refer_c, &d_c, m * n);
            TestLaunchKernel(MyGemmGlobalV6(share + global prefetch + share prefetch)NoRepeats, MyGemmGlobalV6NoRepeats(d_d, d_b, d_c, m, n, k), GemmRefer(&a, &b, &refer_c, m, n, k)());
            WarmupKernel(MyGemmGlobalV6NoRepeats(d_d, d_b, d_c, m, n, k));
            ProfileKernel(MyGemmGlobalV6(share + global prefetch + share prefetch)NoRepeats, MyGemmGlobalV6NoRepeats(d_d, d_b, d_c, m, n, k), (RepeatTimes));

            cudaFree(d_d);
        }
        {
            // prepare input.
            auto callBackA = InputMallocAndCpy(&a, &d_a, m * k);
            auto callBackB = InputMallocAndCpy(&b, &d_b, n * k);
            // prepare output.
            auto callBack = OutputMallocAndDelayCpy(&c, &refer_c, &d_c, m * n);
            TestLaunchKernel(nicolaswildeV1, nicolaswildeV1(d_a, d_b, d_c, m, n, k), GemmRefer(&a, &b, &refer_c, m, n, k)());
            WarmupKernel(nicolaswildeV1(d_a, d_b, d_c, m, n, k));
            ProfileKernel(nicolaswildeV1, nicolaswildeV1(d_a, d_b, d_c, m, n, k), RepeatTimes);
        }
        {
            // prepare input.
            auto callBackA = InputMallocAndCpy(&a, &d_a, m * k);
            auto callBackB = InputMallocAndCpy(&b, &d_b, n * k);
            // prepare output.
            auto callBack = OutputMallocAndDelayCpy(&c, &refer_c, &d_c, m * n);
            TestLaunchKernel(nicolaswildeV2, nicolaswildeV2(d_a, d_b, d_c, m, n, k), GemmRefer(&a, &b, &refer_c, m, n, k)());
            WarmupKernel(nicolaswildeV2(d_a, d_b, d_c, m, n, k));
            ProfileKernel(nicolaswildeV2, nicolaswildeV2(d_a, d_b, d_c, m, n, k), RepeatTimes);
        }
        {
            // prepare input.
            auto callBackA = InputMallocAndCpy(&a, &d_a, m * k);
            auto callBackB = InputMallocAndCpy(&b, &d_b, n * k);
            // prepare output.
            auto callBack = OutputMallocAndDelayCpy(&c, &refer_c, &d_c, m * n);
            TestLaunchKernel(nicolaswildeV3, nicolaswildeV3(d_a, d_b, d_c, m, n, k), GemmRefer(&a, &b, &refer_c, m, n, k)());
            WarmupKernel(nicolaswildeV3(d_a, d_b, d_c, m, n, k));
            ProfileKernel(nicolaswildeV3, nicolaswildeV3(d_a, d_b, d_c, m, n, k), RepeatTimes);
        }
        return true;
    }
};

REGISTER_TEST(Gemm, GemmTest);

#endif //ARM64_TEST_SGEMMTEST_H
