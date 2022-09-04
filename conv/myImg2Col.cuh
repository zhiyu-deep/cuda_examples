//
// Created by 63479 on 2022/9/1.
//

#ifndef ARM64_TEST_MYIMG2COL_CUH
#define ARM64_TEST_MYIMG2COL_CUH

void MyImg2ColConvNaive(const float *a, const float *weight, float *b,
                        int n, int c, int h, int w, int outputc,
                        int kernelX, int kernelY, int dilateX, int dilateY, int strideX, int strideY, int padX, int padY);

#endif //ARM64_TEST_MYIMG2COL_CUH
