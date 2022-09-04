//
// Created by 63479 on 2022/8/11.
//

#ifndef CUDA_TEST_COMMON_CUH
#define CUDA_TEST_COMMON_CUH

#include <cstdlib>
#include <memory>
#include <iostream>

#include "cuda_runtime_api.h"
#include <chrono>

//===========================================================(timer)====================================================
using namespace std::chrono;
class TimerClock
{
public:
    TimerClock()
    {
        update();
    }

    ~TimerClock()
    {
    }

    void update()
    {
        _start = high_resolution_clock::now();
    }
    //获取秒
    double getTimerSecond()
    {
        return getTimerMicroSec() * 0.000001;
    }
    //获取毫秒
    double getTimerMilliSec()
    {
        return getTimerMicroSec()*0.001;
    }
    //获取微妙
    long long getTimerMicroSec()
    {
        //当前时钟减去开始时钟的count
        return duration_cast<microseconds>(high_resolution_clock::now() - _start).count();
    }
private:
    time_point<high_resolution_clock>_start;
};


//========================================================(init)========================================================

int GetRand(int min, int max);

/**
 * @brief 为当前计算任务分配input.
 */
template<typename T>
void InputMallocAndCpy(T **h_ptr, T**d_ptr, int length) {
    *h_ptr = (T*)malloc(sizeof(T) * length);
    cudaMalloc((void**)d_ptr, sizeof(T) * length);
    for (int i = 0; i < length; i++) {
        (*h_ptr)[i] = GetRand(-1, 1);
    }
    cudaMemcpy(*d_ptr, *h_ptr, sizeof(T) * length, cudaMemcpyHostToDevice);
}


template<typename T>
class CallBack {
public:
    CallBack(T *h_ptr, T *refer_ptr, T*d_ptr, int length)
        : h_ptr_(h_ptr),
          refer_ptr_(refer_ptr),
          d_ptr_(d_ptr),
          length_(length) {}

    ~CallBack() {
        MemCpy();
        bool fail = false;
        for (int i = 0; i < length_; i++) {
            auto h_output = h_ptr_[i];
            auto refer_output = refer_ptr_[i];
            if (std::abs(h_output - refer_output) / std::abs(h_output) > 5e-2) {
                fail = true;
            }
        }
        if (!fail) {
            std::cout << "eval success!" << std::endl;
        } else {
            std::cout << "eval fail!" << std::endl;
        }
        free(h_ptr_);
        free(refer_ptr_);
        cudaFree(d_ptr_);
    }

private:
    void MemCpy() {
        cudaMemcpy(h_ptr_, d_ptr_, sizeof(T) * length_, cudaMemcpyDeviceToHost);
    }

    T *h_ptr_;
    T *refer_ptr_;
    T *d_ptr_;
    int length_;
};

void Default();

/**
 * @brief 为当前计算任务分配output指针(cpu output, gpu output, refer output); 务必在InputMallocAndCpy执行后执行.
 * @return 务必用 const& 来引用返回结果, 否则会提前析构.
 */
template<typename T, typename Func = decltype(Default)>
auto OutputMallocAndDelayCpy(T **h_ptr, T **refer_ptr, T **d_ptr, int length,
                             const Func &f = Default) -> CallBack<T> {
    *h_ptr = (T*)malloc(sizeof(T) * length);
    *refer_ptr = (T*) malloc(sizeof(T) * length);
    cudaMalloc((void**)d_ptr,sizeof(T) * length);
    cudaMemset(*d_ptr, 0, sizeof(T) * length);
    cudaDeviceSynchronize();
    // 完成在所有input上的计算, 将结果保存到所有的refer ptr.
    f();
    return CallBack<T>(*h_ptr, *refer_ptr, *d_ptr, length);
}


//=======================================================(launch)=======================================================

#define TestLaunchKernel(funcName, f)                                        \
    cudaGetLastError();                                                      \
    f;                                                                       \
    cudaDeviceSynchronize();                                                 \
    auto status = cudaGetLastError();                                        \
    if (status != cudaSuccess) {                                             \
        printf("%s Error: %s.\n", #funcName, cudaGetErrorString(status));    \
        exit(1);                                                             \
    }

constexpr int RepeatTimes = 100;

#define WarmupKernel(f)                                                     \
    for (int i = 0; i < 10; i++) {                                          \
        f;                                                                  \
    }

#define ProfileKernel(funcName, f, repeat)                                  \
    cudaEvent_t start, end;                                                 \
    cudaEventCreate(&start);                                                \
    cudaEventCreate(&end);                                                  \
    cudaEventRecord(start);                                                 \
    for (int i = 0; i < repeat; i++) {                                      \
        f;                                                                  \
    }                                                                       \
    cudaEventRecord(end);                                                   \
    cudaEventSynchronize(end);                                              \
    float msec;                                                             \
    cudaEventElapsedTime(&msec, start, end);                                \
    msec = msec / repeat;                                                   \
    std::cout << #funcName << " elapse: " << msec << "(ms)" << std::endl;

#endif //CUDA_TEST_COMMON_CUH
