//
// Created by 63479 on 2022/9/22.
//

#include "register.h"
#include "common.cuh"


#include "scan_impl.h"

class ScanTestRefer {
public:
    ScanTestRefer(float **a, float **b, int m) : a_(a), b_(b), m_(m) {}

    void operator() () {
        if (m_ > 0) {
            (*b_)[0] = (*a_)[0];
        }
        for (int i = 1; i < m_; i++) {
            (*b_)[i] = (*b_)[i - 1] + (*a_)[i];
        }
    }

    float **a_;
    float **b_;
    int m_;
};

struct CustomMin {
   template <typename T>
   __device__ __forceinline__
   T operator()(const T &a, const T &b) const {
             return a + b;
   }
};

class ScanTest : public Test {
public:
    explicit ScanTest(const std::string &name) : Test(name) {}

    virtual ErrorCode run() const {
        {
            // 触发WarpScan.
            int m = 320;
            float *a, *b, *d_a, *d_b;
            float *refer_b;
            {
                // prepare input.
                auto callBackA = InputMallocAndCpy(&a, &d_a, m);
                // prepare output.
                auto callBack = OutputMallocAndDelayCpy(&b, &refer_b, &d_b, m);
                TestLaunchKernel(WarpScan, Scan(d_a, d_b, m), ScanTestRefer(&a, &refer_b, m)());
                WarmupKernel(Scan(d_a, d_b, m));
                ProfileKernel(WarpScan, Scan(d_a, d_b, m), RepeatTimes);
            }
            {
                CustomMin  add_op;
                int          init;
                void     *d_temp_storage = NULL;
                size_t   temp_storage_bytes = 0;
//                cub::DeviceReduce::Reduce(
//                        d_temp_storage, temp_storage_bytes,
//                        d_a, d_b, m, add_op, init);
            }
        }
        return kNoError;
    }

    ~ScanTest() = default;
};

REGISTER_TEST(Scan, ScanTest);