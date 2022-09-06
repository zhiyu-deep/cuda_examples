//
// Created by 63479 on 2022/8/25.
//

#ifndef ARM64_TEST_SGEMMTEST_H
#define ARM64_TEST_SGEMMTEST_H

#include "register.h"
#include "common.cuh"

#include "MyGemmV1.h"
#include "MyGemmV2.cuh"
#include "MyGemmV3.cuh"
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
        int m = 1024, n = 1024, k = 32;
        float *a, *b, *c, *d_a, *d_b, *d_c;
        float *refer_c;

        {
            // prepare input.
            auto callBackA = InputMallocAndCpy(&a, &d_a, m * k);
            auto callBackB = InputMallocAndCpy(&b, &d_b, n * k);
            // prepare output.
            auto callBack = OutputMallocAndDelayCpy(&c, &refer_c, &d_c, m * n);
            TestLaunchKernel(MyGemmGlobalV1, MyGemmGlobalV1(d_a, d_b, d_c, m, n, k), GemmRefer(&a, &b, &refer_c, m, n, k)());
            WarmupKernel(MyGemmGlobalV1(d_a, d_b, d_c, m, n, k));
            ProfileKernel(MyGemmGlobalV1, MyGemmGlobalV1(d_a, d_b, d_c, m, n, k), RepeatTimes);
        }
        {
            // prepare input.
            auto callBackA = InputMallocAndCpy(&a, &d_a, m * k);
            auto callBackB = InputMallocAndCpy(&b, &d_b, n * k);
            // prepare output.
            auto callBack = OutputMallocAndDelayCpy(&c, &refer_c, &d_c, m * n);
            TestLaunchKernel(MyGemmGlobalV2, MyGemmGlobalV2(d_a, d_b, d_c, m, n, k), GemmRefer(&a, &b, &refer_c, m, n, k)());
            WarmupKernel(MyGemmGlobalV2(d_a, d_b, d_c, m, n, k));
            ProfileKernel(MyGemmGlobalV2, MyGemmGlobalV2(d_a, d_b, d_c, m, n, k), RepeatTimes);
        }
        {
            // prepare input.
            auto callBackA = InputMallocAndCpy(&a, &d_a, m * k);
            auto callBackB = InputMallocAndCpy(&b, &d_b, n * k);
            // prepare output.
            auto callBack = OutputMallocAndDelayCpy(&c, &refer_c, &d_c, m * n);
            TestLaunchKernel(MyGemmGlobalV3, MyGemmGlobalV3(d_a, d_b, d_c, m, n, k), GemmRefer(&a, &b, &refer_c, m, n, k)());
            WarmupKernel(MyGemmGlobalV3(d_a, d_b, d_c, m, n, k));
            ProfileKernel(MyGemmGlobalV3, MyGemmGlobalV3(d_a, d_b, d_c, m, n, k), (RepeatTimes + 1));
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
