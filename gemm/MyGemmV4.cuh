//
// Created by 63479 on 2022/9/5.
//

#ifndef ARM64_TEST_MYGEMMV4_CUH
#define ARM64_TEST_MYGEMMV4_CUH

void MyGemmGlobalV4Repeats(float *a, float *b, float *c,
                           int M, int N, int K);
void MyGemmGlobalV4NoRepeats(float *a, float *b, float *c,
                             int M, int N, int K);

#endif //ARM64_TEST_MYGEMMV4_CUH
