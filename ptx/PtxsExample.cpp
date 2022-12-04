//
// Created by 63479 on 2022/11/13.
//
#include "common.cuh"
#include "register.h"
#include "ptxs.h"

class PtxsGemmFLoat16Refer {
public:
    PtxsGemmFLoat16Refer(uint16_t **a, uint16_t **b, uint16_t **c, uint16_t **d, int m, int n, int k)
            : a_(a),
              b_(b),
              c_(c),
              d_(d),
              m_(m),
              n_(n),
              k_(k) {}

    void operator() () const {
        for (int i = 0; i < m_; i++) {
            for (int j = 0; j < n_; j++) {
                for (int k = 0; k < k_; k++) {
                    float a_f, b_f, c_f, d_f;
                    float32(&a_f, (*a_)[i * k_ + k]);
                    float32(&b_f, (*b_)[j * k_ + k]);
                    float32(&c_f, (*c_)[i * n_ + j]);
                    float32(&d_f, (*d_)[i * n_ + j]);
                    d_f += a_f * b_f;
                    uint16_t d_i;
                    float16(&d_i, d_f);
                    (*d_)[i * n_ + j] = d_i;
                }
                float c_f, d_f;
                float32(&c_f, (*c_)[i * n_ + j]);
                float32(&d_f, (*d_)[i * n_ + j]);
                d_f += c_f;
                uint16_t d_i;
                float16(&d_i, d_f);
                float result;
                float32(&result, d_i);
                (*d_)[i * n_ + j] = d_i;
            }
        }
    }

    uint16_t **a_, **b_, **c_, **d_;
    int m_, n_, k_;
};

class PtxsGemmFLoat32Refer {
public:
    PtxsGemmFLoat32Refer(uint16_t **a, uint16_t **b, float **c, float **d, int m, int n, int k)
            : a_(a),
              b_(b),
              c_(c),
              d_(d),
              m_(m),
              n_(n),
              k_(k) {}

    void operator() () const {
        for (int i = 0; i < m_; i++) {
            for (int j = 0; j < n_; j++) {
                for (int k = 0; k < k_; k++) {
                    float a_f, b_f, c_f, d_f = (*d_)[i * n_ + j];
                    float32(&a_f, (*a_)[i * k_ + k]);
                    float32(&b_f, (*b_)[j * k_ + k]);
                    d_f += a_f * b_f;
                    (*d_)[i * n_ + j] = d_f;
                }
                float c_f = (*c_)[i * n_ + j], d_f = (*d_)[i * n_ + j];
                d_f += c_f;
                (*d_)[i * n_ + j] = d_f;
            }
        }
    }

    uint16_t **a_, **b_;
    float **c_, **d_;
    int m_, n_, k_;
};

class PtxsGemmExample : public Test {
public:
    PtxsGemmExample(const std::string &name) : Test(name) {}

    virtual ErrorCode run() const {
        if (transposeTest()) {
            return kNoError;
        }
        return kError;
    }

    virtual ~PtxsGemmExample() = default;

private:
    bool transposeTest() const {
        int m = 16, n = 8, k = 8;

        {
            uint16_t *a, *b, *c, *d, *d_a, *d_b, *d_c, *d_d;
            uint16_t *refer_d;

            // prepare input.
            auto callBackA = InputMallocAndCpy(&a, &d_a, m * k);
            auto callBackB = InputMallocAndCpy(&b, &d_b, n * k);
            auto callBackC = InputMallocAndCpy(&c, &d_c, m * n);
            // prepare output.
            auto callBack = OutputMallocAndDelayCpy(&d, &refer_d, &d_d, m * n);

            TestLaunchKernel("PtxFLoat16Gemm", ptx_gemm_cpp(d_a, d_b, d_c, d_d), PtxsGemmFLoat16Refer(&a, &b, &c, &refer_d, m, n, k)());
        }

        {
            uint16_t *a, *b, *d_a, *d_b;
            float *c, *d, *d_c, *d_d;
            float *refer_d;

            // prepare input.
            auto callBackA = InputMallocAndCpy(&a, &d_a, m * k);
            auto callBackB = InputMallocAndCpy(&b, &d_b, n * k);
            auto callBackC = InputMallocAndCpy(&c, &d_c, m * n);
            // prepare output.
            auto callBack = OutputMallocAndDelayCpy(&d, &refer_d, &d_d, m * n);

            TestLaunchKernel("PtxFLoat32Gemm", ptx_gemm_cpp(d_a, d_b, d_c, d_d), PtxsGemmFLoat32Refer(&a, &b, &c, &refer_d, m, n, k)());
        }

        return true;
    }
};

REGISTER_TEST(PtxsGemm, PtxsGemmExample);