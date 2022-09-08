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

float GetRand();

/**
 * @brief 管理一组input指针.
 */
template<typename T>
class InputCallBack {
public:
    InputCallBack(T **h_ptr, T **d_ptr) : h_ptr_(h_ptr), d_ptr_(d_ptr) {}

    InputCallBack(InputCallBack &&obj) : h_ptr_(obj.h_ptr_), d_ptr_(obj.d_ptr_) {
        obj.h_ptr_ = nullptr;
        obj.d_ptr_ = nullptr;
    }

    ~InputCallBack() {
        if (h_ptr_ && *h_ptr_) {
            free(*h_ptr_);
            *h_ptr_ = nullptr;
        }
        if (d_ptr_ && *d_ptr_) {
            cudaFree(*d_ptr_);
            *d_ptr_ = nullptr;
        }
    }
private:
    T **h_ptr_;
    T **d_ptr_;
};

/**
 * @brief 为当前计算任务分配input; 外部只负责不用, 不负责释放.
 */
template<typename T>
auto InputMallocAndCpy(T **h_ptr, T**d_ptr, int length) -> InputCallBack<T> {
    *h_ptr = (T*)malloc(sizeof(T) * length);
    cudaMalloc((void**)d_ptr, sizeof(T) * length);
    for (int i = 0; i < length; i++) {
        (*h_ptr)[i] = GetRand();
    }
    cudaMemcpy(*d_ptr, *h_ptr, sizeof(T) * length, cudaMemcpyHostToDevice);
    return InputCallBack<T>(h_ptr, d_ptr);
}


/**
 * @brief 管理一组output指针; 外部只负责用, 不负责释放.
 */
template<typename T>
class OutputCallBack {
public:
    OutputCallBack(T **h_ptr, T **refer_ptr, T**d_ptr, int length)
        : h_ptr_(h_ptr),
          refer_ptr_(refer_ptr),
          d_ptr_(d_ptr),
          length_(length) {}

    OutputCallBack(OutputCallBack &&obj) : h_ptr_(obj.h_ptr_), refer_ptr_(obj.refer_ptr_), d_ptr_(obj.d_ptr_), length_(obj.length_) {
        obj.h_ptr_ = nullptr;
        obj.d_ptr_ = nullptr;
        obj.refer_ptr_ = nullptr;
    }

    ~OutputCallBack() {
        MemCpy();
        bool fail = false;
        int index = -1;
        T real, result;
        for (int i = 0; i < length_; i++) {
            auto h_output = (*h_ptr_)[i];
            auto refer_output = (*refer_ptr_)[i];
            if (std::abs(h_output - refer_output) / std::abs(h_output) > 5e-2  && std::abs(h_output - refer_output) >= 0.0005) {
                fail = true;
                real = refer_output;
                result = h_output;
                index = i;
            }
        }
        if (!fail) {
            std::cout << "eval success!" << std::endl;
        } else {
            std::cout << "eval fail! index: " << index << ", real: " << real << ", result: " << result << "." << std::endl;
        }
        if (h_ptr_ && *h_ptr_) {
            free(*h_ptr_);
            h_ptr_ = nullptr;
        }
        if (refer_ptr_ && *refer_ptr_) {
            free(*refer_ptr_);
            refer_ptr_ = nullptr;
        }
        if (d_ptr_ && *d_ptr_) {
            cudaFree(*d_ptr_);
            d_ptr_ = nullptr;
        }
    }

private:
    void MemCpy() {
        cudaMemcpy(*h_ptr_, *d_ptr_, sizeof(T) * length_, cudaMemcpyDeviceToHost);
    }

    T **h_ptr_;
    T **refer_ptr_;
    T **d_ptr_;
    int length_;
};

void Default();

/**
 * @brief 为当前计算任务分配output指针(cpu output, gpu output, refer output); 务必在InputMallocAndCpy执行后执行.
 * @return 务必用 const& 来引用返回结果, 否则会提前析构.
 */
template<typename T>
auto OutputMallocAndDelayCpy(T **h_ptr, T **refer_ptr, T **d_ptr, int length) -> OutputCallBack<T> {
    *h_ptr = (T*)malloc(sizeof(T) * length);
    *refer_ptr = (T*) malloc(sizeof(T) * length);
    cudaMalloc((void**)d_ptr,sizeof(T) * length);
    cudaMemset(*d_ptr, 0, sizeof(T) * length);
    cudaDeviceSynchronize();
    return OutputCallBack<T>(h_ptr, refer_ptr, d_ptr, length);
}


//=======================================================(launch)=======================================================

#define TestLaunchKernel(funcName, f, refer_func)                            \
    refer_func;                                                              \
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
    for (int i = 0; i < (repeat); i++) {                                    \
        f;                                                                  \
    }                                                                       \
    cudaEventRecord(end);                                                   \
    cudaEventSynchronize(end);                                              \
    float msec;                                                             \
    cudaEventElapsedTime(&msec, start, end);                                \
    msec = msec / (repeat);                                                 \
    std::cout << #funcName << " elapse: " << msec << "(ms)" << std::endl;

#endif //CUDA_TEST_COMMON_CUH
