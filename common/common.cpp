//
// Created by 63479 on 2022/8/11.
//

#include "common.cuh"

#include<random>

float GetRand() {
    using namespace std;
    static default_random_engine e;
    static uniform_real_distribution<float> u(-1, 1);
    return u(e);
}

void Default() { return; }