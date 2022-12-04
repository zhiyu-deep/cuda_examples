//
// Created by 63479 on 2022/11/30.
//

#include "stdio.h"
#include <cstdint>
#include "cuda_fp16.h"
#include "ptxs.h"
#include "common.cuh"

__device__ int Add(int a, int b) {
    int c;
    asm volatile (
        "add.s32 %0, %1, %2;\n\t"
        : "=r"(c) : "r"(a), "r"(b));
    return c;
}

// clip to MININT...MAXINT.
__device__ int AddSat(int a, int b) {
    int c;
    asm volatile (
            "add.sat.s32 %0, %1, %2;\n\t"
            : "=r"(c) : "r"(a), "r"(b)
            );
    return c;
}

__device__ int Sub(int a, int b) {
    int c;
    asm  volatile (
            "sub.s32 %0, %1, %2;\n\t"
            : "=r"(c) : "r"(a), "r"(b));
    return c;
}

// clip to MININT...MAXINT.
__device__ int SubSat(int a, int b) {
    int c;
    asm  volatile (
            "sub.sat.s32 %0, %1, %2;\n\t"
            : "=r"(c) : "r"(a), "r"(b));
    return c;
}

template<typename MultiType, typename AddType, LayoutType ALayout = kRow, LayoutType BLayout = kCol>
__device__ void MmaM16N8K8(MultiType *a,
                           MultiType *b,
                           AddType   *c,
                           AddType   *d,
                           int a_stride = 8,
                           int b_stride = 8,
                           int c_stride = 8,
                           int d_stride = 8);

//A(M16N8K8, kRow, fp16):
//|0 | T0:{a0, a1}  | T1:{a0, a1}  | T2:{a0, a1}  | T3:{a0, a1}  |
//|1 | T4:{a0, a1}  | T5:{a0, a1}  | T6:{a0, a1}  | T7:{a0, a1}  |
//|2 | T8:{a0, a1}  | T9:{a0, a1}  | T10:{a0, a1} | T11:{a0, a1} |
//|3 | T12:{a0, a1} | T13:{a0, a1} | T14:{a0, a1} | T15:{a0, a1} |
//|4 | T16:{a0, a1} | T17:{a0, a1} | T18:{a0, a1} | T19:{a0, a1} |
//|5 | T20:{a0, a1} | T21:{a0, a1} | T22:{a0, a1} | T23:{a0, a1} |
//|6 | T24:{a0, a1} | T25:{a0, a1} | T26:{a0, a1} | T27:{a0, a1} |
//|7 | T28:{a0, a1} | T29:{a0, a1} | T30:{a0, a1} | T31:{a0, a1} |
//|8 | T0:{a0, a1}  | T1:{a0, a1}  | T2:{a0, a1}  | T3:{a0, a1}  |
//|9 | T4:{a0, a1}  | T5:{a0, a1}  | T6:{a0, a1}  | T7:{a0, a1}  |
//|10| T8:{a0, a1}  | T9:{a0, a1}  | T10:{a0, a1} | T11:{a0, a1} |
//|11| T12:{a0, a1} | T13:{a0, a1} | T14:{a0, a1} | T15:{a0, a1} |
//|12| T16:{a0, a1} | T17:{a0, a1} | T18:{a0, a1} | T19:{a0, a1} |
//|13| T20:{a0, a1} | T21:{a0, a1} | T22:{a0, a1} | T23:{a0, a1} |
//|14| T24:{a0, a1} | T25:{a0, a1} | T26:{a0, a1} | T27:{a0, a1} |
//|15| T28:{a0, a1} | T29:{a0, a1} | T30:{a0, a1} | T31:{a0, a1} |

//B(M16N8K8, kCol, fp16)
//|0 | T0:b0 | T4:b0 | T8:b0  | T12:b0 | T16:b0 | T20:b0 | T24:b0 | T28:b0 |
//|1 | T0:b1 | T4:b1 | T8:b1  | T12:b1 | T16:b1 | T20:b1 | T24:b0 | T28:b1 |
//|2 | T1:b0 | T5:b0 | T9:b0  | T13:b0 | T17:b0 | T21:b0 | T25:b0 | T29:b0 |
//|3 | T1:b1 | T5:b1 | T9:b1  | T13:b1 | T17:b1 | T21:b1 | T25:b0 | T29:b1 |
//|4 | T2:b0 | T6:b0 | T10:b0 | T14:b0 | T18:b0 | T22:b0 | T26:b0 | T30:b0 |
//|5 | T2:b1 | T6:b1 | T10:b1 | T14:b1 | T18:b1 | T22:b1 | T26:b0 | T30:b1 |
//|6 | T3:b0 | T7:b0 | T11:b0 | T15:b0 | T19:b0 | T23:b0 | T27:b0 | T31:b0 |
//|7 | T3:b1 | T7:b1 | T11:b1 | T15:b1 | T19:b1 | T23:b1 | T27:b0 | T31:b1 |

//C|D(M16N8K8, kRow, fp16)
//|0 | T0:{c0, c1}  | T1:{c0, c1}  | T2:{c0, c1}  | T3:{c0, c1}  |
//|1 | T4:{c0, c1}  | T5:{c0, c1}  | T6:{c0, c1}  | T7:{c0, c1}  |
//|2 | T8:{c0, c1}  | T9:{c0, c1}  | T10:{c0, c1} | T11:{c0, c1} |
//|3 | T12:{c0, c1} | T13:{c0, c1} | T14:{c0, c1} | T15:{c0, c1} |
//|4 | T16:{c0, c1} | T17:{c0, c1} | T18:{c0, c1} | T19:{c0, c1} |
//|5 | T20:{c0, c1} | T21:{c0, c1} | T22:{c0, c1} | T23:{c0, c1} |
//|6 | T24:{c0, c1} | T25:{c0, c1} | T26:{c0, c1} | T27:{c0, c1} |
//|7 | T28:{c0, c1} | T29:{c0, c1} | T30:{c0, c1} | T31:{c0, c1} |
//|8 | T0:{c0, c1}  | T1:{c0, c1}  | T2:{c0, c1}  | T3:{c0, c1}  |
//|9 | T4:{c0, c1}  | T5:{c0, c1}  | T6:{c0, c1}  | T7:{c0, c1}  |
//|10| T8:{c0, c1}  | T9:{c0, c1}  | T10:{c0, c1} | T11:{c0, c1} |
//|11| T12:{c0, c1} | T13:{c0, c1} | T14:{c0, c1} | T15:{c0, c1} |
//|12| T16:{c0, c1} | T17:{c0, c1} | T18:{c0, c1} | T19:{c0, c1} |
//|13| T20:{c0, c1} | T21:{c0, c1} | T22:{c0, c1} | T23:{c0, c1} |
//|14| T24:{c0, c1} | T25:{c0, c1} | T26:{c0, c1} | T27:{c0, c1} |
//|15| T28:{c0, c1} | T29:{c0, c1} | T30:{c0, c1} | T31:{c0, c1} |

/**
 * @brief asm指令中, 数据类型:
 *        1. 操作数都是vector{.float16*2}, 其中每个.float16*2可以用uint32_t传入, vector则是依次传入即可, constraint中务必和cpp数据类型一致, uint32_t->r.
 *        2. 操作数都是vector{.float32}, 其中每个.float32可以用float传入, vector则是依次传入即可, constraint中务必和cpp数据类型一致, float32->f.
 */
template<>
__device__ void MmaM16N8K8<half, half, kRow, kCol>(half* a, half *b, half *c, half*d,
                                                   int a_stride, int b_stride, int c_stride, int d_stride) {
    int warpId = threadIdx.x & 31;

    int a_group1_y = warpId >> 2, a_group1_x = warpId & 3;
    int a_group2_y = a_group1_y + 8, a_group2_x = a_group1_x;
    assert(a_stride % 2 == 0);
    uint32_t a0 = ((uint32_t*)(a))[a_group1_y * a_stride / 2 + a_group1_x],
              a1 = ((uint32_t*)(a))[a_group2_y * a_stride / 2 + a_group2_x];

    int b_y = warpId & 3, b_x = warpId >> 2;
    assert(b_stride % 2 == 0);
    uint32_t b0 = ((uint32_t*)(b))[b_x * b_stride / 2 + b_y];

    assert(c_stride % 2 == 0);
    uint32_t c0 = ((uint32_t *)(c))[a_group1_y * c_stride / 2 + a_group1_x],
             c1 = ((uint32_t*)(c))[a_group2_y * c_stride / 2 + a_group2_x];
    uint32_t d0, d1;

    asm volatile (
            "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%5, %6};\n\t"
            : "=r"(d0), "=r"(d1) : "r"(a0), "r"(a1), "r"(b0), "r"(c0), "r"(c1)
            );

    assert(d_stride % 2 == 0);
    ((uint32_t*)(d))[a_group1_y * d_stride / 2 + a_group1_x] = d0,
    ((uint32_t*)(d))[a_group2_y * d_stride / 2 + a_group2_x] = d1;
}

//A(M16N8K8, kRow, fp16):
//|0 | T0:{a0, a1}  | T1:{a0, a1}  | T2:{a0, a1}  | T3:{a0, a1}  |
//|1 | T4:{a0, a1}  | T5:{a0, a1}  | T6:{a0, a1}  | T7:{a0, a1}  |
//|2 | T8:{a0, a1}  | T9:{a0, a1}  | T10:{a0, a1} | T11:{a0, a1} |
//|3 | T12:{a0, a1} | T13:{a0, a1} | T14:{a0, a1} | T15:{a0, a1} |
//|4 | T16:{a0, a1} | T17:{a0, a1} | T18:{a0, a1} | T19:{a0, a1} |
//|5 | T20:{a0, a1} | T21:{a0, a1} | T22:{a0, a1} | T23:{a0, a1} |
//|6 | T24:{a0, a1} | T25:{a0, a1} | T26:{a0, a1} | T27:{a0, a1} |
//|7 | T28:{a0, a1} | T29:{a0, a1} | T30:{a0, a1} | T31:{a0, a1} |
//|8 | T0:{a0, a1}  | T1:{a0, a1}  | T2:{a0, a1}  | T3:{a0, a1}  |
//|9 | T4:{a0, a1}  | T5:{a0, a1}  | T6:{a0, a1}  | T7:{a0, a1}  |
//|10| T8:{a0, a1}  | T9:{a0, a1}  | T10:{a0, a1} | T11:{a0, a1} |
//|11| T12:{a0, a1} | T13:{a0, a1} | T14:{a0, a1} | T15:{a0, a1} |
//|12| T16:{a0, a1} | T17:{a0, a1} | T18:{a0, a1} | T19:{a0, a1} |
//|13| T20:{a0, a1} | T21:{a0, a1} | T22:{a0, a1} | T23:{a0, a1} |
//|14| T24:{a0, a1} | T25:{a0, a1} | T26:{a0, a1} | T27:{a0, a1} |
//|15| T28:{a0, a1} | T29:{a0, a1} | T30:{a0, a1} | T31:{a0, a1} |

//B(M16N8K8, kCol, fp16)
//|0 | T0:b0 | T4:b0 | T8:b0  | T12:b0 | T16:b0 | T20:b0 | T24:b0 | T28:b0 |
//|1 | T0:b1 | T4:b1 | T8:b1  | T12:b1 | T16:b1 | T20:b1 | T24:b0 | T28:b1 |
//|2 | T1:b0 | T5:b0 | T9:b0  | T13:b0 | T17:b0 | T21:b0 | T25:b0 | T29:b0 |
//|3 | T1:b1 | T5:b1 | T9:b1  | T13:b1 | T17:b1 | T21:b1 | T25:b0 | T29:b1 |
//|4 | T2:b0 | T6:b0 | T10:b0 | T14:b0 | T18:b0 | T22:b0 | T26:b0 | T30:b0 |
//|5 | T2:b1 | T6:b1 | T10:b1 | T14:b1 | T18:b1 | T22:b1 | T26:b0 | T30:b1 |
//|6 | T3:b0 | T7:b0 | T11:b0 | T15:b0 | T19:b0 | T23:b0 | T27:b0 | T31:b0 |
//|7 | T3:b1 | T7:b1 | T11:b1 | T15:b1 | T19:b1 | T23:b1 | T27:b0 | T31:b1 |

//C|D(M16N8K8, kRow, fp32)
//|0 | T0:{c0, c1}  | T1:{c0, c1}  | T2:{c0, c1}  | T3:{c0, c1}  |
//|1 | T4:{c0, c1}  | T5:{c0, c1}  | T6:{c0, c1}  | T7:{c0, c1}  |
//|2 | T8:{c0, c1}  | T9:{c0, c1}  | T10:{c0, c1} | T11:{c0, c1} |
//|3 | T12:{c0, c1} | T13:{c0, c1} | T14:{c0, c1} | T15:{c0, c1} |
//|4 | T16:{c0, c1} | T17:{c0, c1} | T18:{c0, c1} | T19:{c0, c1} |
//|5 | T20:{c0, c1} | T21:{c0, c1} | T22:{c0, c1} | T23:{c0, c1} |
//|6 | T24:{c0, c1} | T25:{c0, c1} | T26:{c0, c1} | T27:{c0, c1} |
//|7 | T28:{c0, c1} | T29:{c0, c1} | T30:{c0, c1} | T31:{c0, c1} |
//|8 | T0:{c0, c1}  | T1:{c0, c1}  | T2:{c0, c1}  | T3:{c0, c1}  |
//|9 | T4:{c0, c1}  | T5:{c0, c1}  | T6:{c0, c1}  | T7:{c0, c1}  |
//|10| T8:{c0, c1}  | T9:{c0, c1}  | T10:{c0, c1} | T11:{c0, c1} |
//|11| T12:{c0, c1} | T13:{c0, c1} | T14:{c0, c1} | T15:{c0, c1} |
//|12| T16:{c0, c1} | T17:{c0, c1} | T18:{c0, c1} | T19:{c0, c1} |
//|13| T20:{c0, c1} | T21:{c0, c1} | T22:{c0, c1} | T23:{c0, c1} |
//|14| T24:{c0, c1} | T25:{c0, c1} | T26:{c0, c1} | T27:{c0, c1} |
//|15| T28:{c0, c1} | T29:{c0, c1} | T30:{c0, c1} | T31:{c0, c1} |
template<>
__device__ void MmaM16N8K8<half, float, kRow, kCol>(half* a, half *b, float *c, float*d,
                                                   int a_stride, int b_stride, int c_stride, int d_stride) {
    int warpId = threadIdx.x & 31;

    int a_group1_y = warpId >> 2, a_group1_x = warpId & 3;
    int a_group2_y = a_group1_y + 8, a_group2_x = a_group1_x;
    assert(a_stride % 2 == 0);
    uint32_t a0 = ((uint32_t*)(a))[a_group1_y * a_stride / 2 + a_group1_x],
             a1 = ((uint32_t*)(a))[a_group2_y * a_stride / 2 + a_group2_x];

    int b_y = warpId & 3, b_x = warpId >> 2;
    assert(b_stride % 2 == 0);
    uint32_t b0 = ((uint32_t*)(b))[b_x * b_stride / 2 + b_y];

    float c0 = ((float *)(c))[a_group1_y * c_stride + a_group1_x * 2],
          c1 = ((float *)(c))[a_group1_y * c_stride + a_group1_x * 2 + 1],
          c2 = ((float *)(c))[a_group2_y * c_stride + a_group2_x * 2],
          c3 = ((float *)(c))[a_group2_y * c_stride + a_group2_x * 2 + 1];
    float d0, d1, d2, d3;

    asm volatile (
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n\t"
            : "=f"(d0), "=f"(d1) , "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(b0), "f"(c0), "f"(c1), "f"(c2), "f"(c3)
            );

    ((float*)(d))[a_group1_y * d_stride + a_group1_x * 2] = d0,
    ((float*)(d))[a_group1_y * d_stride + a_group1_x * 2 + 1] = d1;
    ((float*)(d))[a_group2_y * d_stride + a_group2_x * 2] = d2,
    ((float*)(d))[a_group2_y * d_stride + a_group2_x * 2 + 1] = d3;
}





template<typename T>
__global__ void ptx_functions_impl(T *a, T *b, int length, FunctionType type, bool need_sat) {
    T *a_block = a + gridDim.x * blockIdx.x;
    T a_d = gridDim.x * blockIdx.x + threadIdx.x < length ? a_block[threadIdx.x] : 0;

    T output;
#define dispatch(op)          \
    case k##op:               \
        output = need_sat ? op##Sat(a_d, 2) : op(a_d, 2);  \
        break;

    switch (type) {
        dispatch(Add)
        dispatch(Sub)
        default:
            break;
    }
#undef dispatch

    T *b_block = b + gridDim.x * blockIdx.x;
    if (gridDim.x * blockIdx.x + threadIdx.x < length) {
        b_block[threadIdx.x] = output;
    }
}

template<typename T>
void ptx_functions(T *a, T *b, int length, FunctionType type, bool need_sat) {
    int block_length = 32 * 32;
    dim3 grids = length / block_length, blocks = block_length;
    ptx_functions_impl<<<grids, blocks>>>(a, b, length, type, need_sat);
}

template void ptx_functions(int *a, int *b, int length, FunctionType type, bool need_sat);





template<typename Mul, typename Add>
__global__ void ptx_gemm_cuda(Mul *a, Mul *b, Add *c, Add *d) {
    MmaM16N8K8<Mul, Add, kRow, kCol>(a, b, c, d);
}

template<typename Mul, typename Add>
void ptx_gemm_cpp(Mul *a, Mul *b, Add *c, Add *d) {
    dim3 grid(1), block(32);
    if (std::is_same<Mul, uint16_t>::value) {
        using AType = half;
        using BType = half;
        if (std::is_same<Add, uint16_t>::value) {
            using CType = half;
            using DType = half;
            ptx_gemm_cuda<AType, CType><<<grid, block>>>((AType *)a, (BType *)b, (CType *)c, (DType *)d);
        } else if (std::is_same<Add, float>::value) {
            using CType = float;
            using DType = float;
            ptx_gemm_cuda<AType, CType><<<grid, block>>>((AType *)a, (BType *)b, (CType *)c, (DType *)d);
        }
    }
}

template void ptx_gemm_cpp<uint16_t, uint16_t>(uint16_t *a, uint16_t *b, uint16_t *c, uint16_t *d);
template void ptx_gemm_cpp<uint16_t, float>(uint16_t *a, uint16_t *b, float *c, float *d);
