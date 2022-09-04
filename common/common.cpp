//
// Created by 63479 on 2022/8/11.
//

#include "common.cuh"

int GetRand(int min, int max) {
    return (rand() % (max - min + 1)) + min;
}

void Default() { return; }