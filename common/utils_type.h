//
// Created by 63479 on 2022/9/19.
//

#ifndef ARM64_TEST_UTILS_TYPE_H
#define ARM64_TEST_UTILS_TYPE_H

#include <type_traits>

template <bool Test, class T1, class T2>
using conditional_t = typename std::conditional<Test, T1, T2>::type;

template<int num>
struct NumInfo {
    static constexpr int value = num;
};

template <bool Test, int N1, int N2>
using conditional_num = typename std::conditional<Test, NumInfo<N1>, NumInfo<N2>>::type ;

#endif //ARM64_TEST_UTILS_TYPE_H
