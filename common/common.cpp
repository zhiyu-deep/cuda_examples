//
// Created by 63479 on 2022/8/11.
//

#include "common.cuh"

#include<random>

float GetRand() {
    return rand() / double(RAND_MAX);
}

void Default() { return; }