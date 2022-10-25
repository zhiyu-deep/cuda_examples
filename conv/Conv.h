//
// Created by 63479 on 2022/9/1.
//

#ifndef ARM64_TEST_CONV_H
#define ARM64_TEST_CONV_H

enum ConvImpl {
    kNaive,
    kOpt1
};

/**
 * @brief a是input矩阵, 按照NC4HW4的方式分布.
 *        weight是权重矩阵, 按照行优先分布, 纵向按照channel优先分布.
 *        b是output矩阵, 按照NC4HW4分布.
 */
void Conv(const float *a, const float *weight, float *b,
          int n, int inputC, int h, int w, int outputC,
          int kernelX, int kernelY, int dilateX, int dilateY, int strideX, int strideY, int padX, int padY,
          ConvImpl implType);

#endif //ARM64_TEST_CONV_H
