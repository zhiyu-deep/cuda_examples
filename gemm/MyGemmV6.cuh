//
// Created by 63479 on 2022/9/5.
//

#ifndef ARM64_TEST_MYGEMMV6_CUH
#define ARM64_TEST_MYGEMMV6_CUH

void MyGemmGlobalV6Repeats(float *a, float *b, float *c,
                           int M, int N, int K);
void MyGemmGlobalV6NoRepeats(float *a, float *b, float *c,
                             int M, int N, int K);

#endif //ARM64_TEST_MYGEMMV6_CUH
