//
// Created by 63479 on 2022/11/13.
//
#include "common.cuh"
#include "register.h"
#include "example.cuh"

class WmmaRefer {
public:
    WmmaRefer(float **a, float **b, float **c, int m, int n, int k)
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

class WmmaExample : public Test {
public:
    WmmaExample(const std::string &name) : Test(name) {}

    virtual ErrorCode run() const {
        if (transposeTest()) {
            return kNoError;
        }
        return kError;
    }

    virtual ~WmmaExample() = default;

private:
    bool transposeTest() const {
        int m = 16, n = 16, k = 16;
        float *a, *b, *c, *d_a, *d_b, *d_c;
        float *refer_c;

        {
            // prepare input.
            auto callBackA = InputMallocAndCpy(&a, &d_a, m * k);
            auto callBackB = InputMallocAndCpy(&b, &d_b, n * k);
            // prepare output.
            auto callBack = OutputMallocAndDelayCpy(&c, &refer_c, &d_c, m * n);
            TestLaunchKernel(WmmaExample, wmma_ker(d_a, d_b, d_c), WmmaRefer(&a, &b, &refer_c, m, n, k)());
        }

        return true;
    }
};

REGISTER_TEST(Wmma, WmmaExample);