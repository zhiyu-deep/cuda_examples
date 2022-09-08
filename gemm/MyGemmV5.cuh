//
// Created by 63479 on 2022/9/5.
//

#ifndef ARM64_TEST_MYGEMMV5_CUH
#define ARM64_TEST_MYGEMMV5_CUH

void MyGemmGlobalV5Repeats(float *a, float *b, float *c,
                           int M, int N, int K);
void MyGemmGlobalV5NoRepeats(float *a, float *b, float *c,
                             int M, int N, int K);

#endif //ARM64_TEST_MYGEMMV5_CUH
