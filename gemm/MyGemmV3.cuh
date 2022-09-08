//
// Created by 63479 on 2022/9/5.
//

#ifndef ARM64_TEST_MYGEMMV3_CUH
#define ARM64_TEST_MYGEMMV3_CUH

void MyGemmGlobalV3Repeats(float *a, float *b, float *c,
                           int M, int N, int K);
void MyGemmGlobalV3NoRepeats(float *a, float *b, float *c,
                             int M, int N, int K);

#endif //ARM64_TEST_MYGEMMV3_CUH
