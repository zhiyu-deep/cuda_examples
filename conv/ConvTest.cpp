//
// Created by 63479 on 2022/8/25.
//

#ifndef ARM64_TEST_SGEMMTEST_H
#define ARM64_TEST_SGEMMTEST_H

#include "algorithm"

#include "register.h"
#include "common.cuh"
#include "ConvUtils.cuh"

#include "Conv.h"

class ConvNC4HW4Refer {
public:
    ConvNC4HW4Refer(float **a, float **weight, float **c,
                    int n, int inputC, int h, int w, int outputC,
                    int p_h, int p_w, int d_h, int d_w, int s_h, int s_w, int k_h, int k_w)
        : a_(a),
          weight_(weight),
          c_(c),
          n_(n),
          inputC_(inputC),
          h_(h),
          w_(w),
          outputC_(outputC),
          p_h_(p_h),
          p_w_(p_w),
          d_h_(d_h),
          d_w_(d_w),
          s_h_(s_h),
          s_w_(s_w),
          k_h_(k_h),
          k_w_(k_w) {}

    void operator() () const {
        int output_h = CalculateOutputLength(h_, k_h_, d_h_, p_h_, s_h_),
            output_w = CalculateOutputLength(w_, k_w_, d_w_, p_w_, s_w_);
        int batch_stride = (h_ + 2 * p_h_) * (w_ + 2 * p_w_) * 4, channel_stride = batch_stride * n_, h_stride = (w_ + 2 * p_w_) * 4;
        for (int g = 0; g < UP_DIVIDE(outputC_, 4); g++) {
            for (int i = 0; i < output_h * output_w * n_; i++) {
                for (int j = 0; j < 4; j++) {

                    int output_index = i % (output_h * output_w), output_b = i / (output_h * output_w);
                    int conv_x = output_index % output_w, conv_y = output_index / output_w;
                    int pad_x = conv_x * s_w_ - p_w_, pad_y = conv_y * s_h_ - p_h_;
                    const float *a_pad_ptr = (*a_) + output_b * batch_stride + pad_y * h_stride + pad_x * 4;
                    int start_offset_x = std::max(0, UP_DIVIDE(-pad_x, d_w_)),
                        start_offset_y = std::max(0, UP_DIVIDE(-pad_y, d_h_)),
                        end_offset_x   = std::min(k_w_ - 1, DOWN_DIVIDE(w_ - 1 - pad_x, d_w_)),
                        end_offset_y   = std::min(k_h_ - 1, DOWN_DIVIDE(h_ - 1 - pad_y, d_h_));
                    float a_value = 0., weight_value = 0., accumulate_value = 0.;
                    for (int n = 0; n < k_h_; n++) {
                        for (int k = 0; k < k_w_; k++) {
                            for (int m = 0; m < UP_DIVIDE(inputC_, 4); m++) {
                                for (int y = 0; y < inputC_ - m * 4; y++) {
                                    weight_value = (*weight_)[g * k_h_ * k_w_ * inputC_ * 4 + (n * k_w_ + k) * inputC_ * 4 + m * 4 * 4 + y * 4 + j];
                                    if (n < start_offset_y || n > end_offset_y) {
                                        a_value = 0.;
                                    } else if (k < start_offset_x || k > end_offset_x) {
                                        a_value = 0.;
                                    } else {
                                        a_value = a_pad_ptr[m * channel_stride + n * h_stride + k * 4 + y];
                                    }
//                                    std::cout << a_value << "," << weight_value << std::endl;
                                    accumulate_value += a_value * weight_value;
                                }
                            }
                        }
                    }
                    (*c_)[g * n_ * output_w * output_h * 4 + i * 4 + j] = accumulate_value;

                }
            }
        }
    }

    float **a_, **weight_, **c_;
    int n_, inputC_, h_, w_, outputC_, p_h_, p_w_, s_h_, s_w_, d_h_, d_w_, k_h_, k_w_;
};

class ConvTest : public Test {
public:
    explicit ConvTest(const std::string &name) : Test(name) {}

    virtual ErrorCode run() const {
        if (ConvTestImpl()) {
            return kNoError;
        }
        return kError;
    }

    virtual ~ConvTest() = default;

private:
    bool ConvTestImpl() const {
        int n = 1, inputC = 160, h = 11, w = 12, outputC = 160;
        int k_h = 3, k_w = 3, d_h = 1, d_w = 1, p_h = 2, p_w = 2, s_h = 1, s_w = 1;
        if (!CheckParamsLegal(h, k_h,d_h, p_h, s_h)) {
            throw std::invalid_argument("无效param.");
        }
        int output_h = CalculateOutputLength(h, k_h, d_h, p_h, s_h),
            output_w = CalculateOutputLength(w, k_w, d_w, p_w, s_w);

        float *a, *weight, *c, *d_a, *d_weight, *d_c;
        float *refer_c;

        {
            // input: NC4HW4, weight: C4, output: NC4HW4.
            // prepare input.
            auto callBackA = InputMallocAndCpy(&a, &d_a, n * h * w * UP_ROUND(inputC, 4));
            auto callBackB = InputMallocAndCpy(&weight, &d_weight, h * w * inputC * UP_ROUND(outputC, 4));
            // prepare output.
            auto callBack = OutputMallocAndDelayCpy(&c, &refer_c, &d_c, n * output_h * output_w * UP_ROUND(outputC, 4));
            TestLaunchKernel(MyConvNaive,
                             Conv(d_a, d_weight, d_c, n, inputC, h, w, outputC, k_w, k_h, d_w, d_h, s_w, s_h, p_w, p_h, kNaive),
                             ConvNC4HW4Refer(&a, &weight, &refer_c, n, inputC, h, w, outputC, p_h, p_w, d_h, d_w, s_h, s_w, k_h, k_w)());
            WarmupKernel(Conv(d_a, d_weight, d_c, n, inputC, h, w, outputC, k_w, k_h, d_w, d_h, s_w, s_h, p_w, p_h, kNaive));
            ProfileKernel(MyConvNaive, Conv(d_a, d_weight, d_c, n, inputC, h, w, outputC, k_w, k_h, d_w, d_h, s_w, s_h, p_w, p_h, kNaive), RepeatTimes);
        }
        return true;
    }
};

REGISTER_TEST(Conv, ConvTest);

#endif //ARM64_TEST_SGEMMTEST_H
