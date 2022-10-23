//
// Created by 63479 on 2022/10/22.
//

#include "Conv.h"

void ConvNaive(const float *a, const float *weight, float *b,
               int n, int c, int h, int w, int outputc,
               int kernelX, int kernelY, int dilateX, int dilateY, int strideX, int strideY, int padX, int padY);


void Conv(const float *a, const float *weight, float *b,
          int n, int inputC, int h, int w, int outputC,
          int kernelX, int kernelY, int dilateX, int dilateY, int strideX, int strideY, int padX, int padY,
          ConvImpl implType) {
    switch (implType) {
        case kNaive:
            ConvNaive(a, weight, b, n, inputC, h, w, outputC, kernelX, kernelY, dilateX, dilateY, strideX, strideY,
                      padX, padY);
    }
}