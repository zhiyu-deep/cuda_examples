//
// Created by 63479 on 2022/9/19.
//

#ifndef ARM64_TEST_UTILS_POLICY_H
#define ARM64_TEST_UTILS_POLICY_H

#include "utils_type.h"

// policy:
//  1. 根据静态条件, 确定使用哪个分支.
//  2. 当前的设计下, policy对外使用的时候, 务必先通过valid判断当前policy的有效性, 确定有效后才可使用ActivePolicy.
struct InvalidPolicy {
    constexpr static bool valid = false;
};

// 1. 中间节点.
template<bool test, typename Policy, typename OtherPolicy>
struct ChainPolicy {
    using ActivePolicy = conditional_t<test, Policy, typename OtherPolicy::ActivePolicy>;

//    constexpr static bool valid = ActivePolicy::valid;
};

// 2. 只有一个有效的叶子节点.
template<bool test, typename Policy>
struct ChainPolicy<test, Policy, InvalidPolicy> {
    using ActivePolicy = conditional_t<test, Policy, InvalidPolicy>;

    constexpr static bool valid = ActivePolicy::valid;
};

// 3. 只有一个叶子节点.
template<bool test, typename UltimatePolicy>
struct ChainPolicy<test, UltimatePolicy, UltimatePolicy> {
    using ActivePolicy = UltimatePolicy;

    constexpr static bool valid = true;
};

#endif //ARM64_TEST_UTILS_POLICY_H
