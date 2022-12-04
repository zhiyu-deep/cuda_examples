//
// Created by 63479 on 2022/11/30.
//

#ifndef ARM64_TEST_PTXS_H
#define ARM64_TEST_PTXS_H

enum FunctionType {
    kAdd,
    kSub,
};

enum LayoutType {
    kRow,
    kCol
};

template<typename T>
void ptx_functions(T *a, T *b, int length, FunctionType type, bool need_sat);

template<typename Mul, typename Add>
void ptx_gemm_cpp(Mul *a, Mul *b, Add *c, Add *d);

#endif //ARM64_TEST_PTXS_H
