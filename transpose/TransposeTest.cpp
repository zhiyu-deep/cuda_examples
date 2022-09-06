//
// Created by 63479 on 2022/9/5.
//

#include "common.cuh"
#include "register.h"
#include "transpose.cuh"

class TransposeRefer {
public:
    TransposeRefer(float **a, float **b, int m, int n)
            : a_(a),
              b_(b),
              m_(m),
              n_(n) {}

    void operator() () const {
        for (int i = 0; i < m_; i++) {
            for (int j = 0; j < n_; j++) {
                (*b_)[j * m_ + i] = (*a_)[i * n_ + j];
            }
        }
    }

    float **a_, **b_;
    int m_, n_;
};

class TransposeTest : public Test {
public:
    TransposeTest(const std::string &name) : Test(name) {}

    virtual ErrorCode run() const {
        if (transposeTest()) {
            return kNoError;
        }
        return kError;
    }

    virtual ~TransposeTest() = default;

private:
    bool transposeTest() const {
        float *a, *d_a;
        float *h_out, *refer_out, *d_out;

        for (int m = 1; m < 34; m++) {
            for (int n = 1; n < 34; n++) {
                auto callBackA = InputMallocAndCpy(&a, &d_a, m * n);
                auto callBackOut = OutputMallocAndDelayCpy(&h_out, &refer_out, &d_out, m * n);
                TestLaunchKernel("transpose", Transpose(d_a, d_out, m, n), TransposeRefer(&a, &refer_out, m, n)());
                WarmupKernel(Transpose(d_a, d_out, m, n));
                ProfileKernel("transpose", Transpose(d_a, d_out, m, n), RepeatTimes);
            }
        }
        return true;
    }
};

REGISTER_TEST(Transpose, TransposeTest);
