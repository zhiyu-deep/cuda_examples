//
// Created by 63479 on 2022/11/12.
//

#include "example.cuh"
#include "iostream"
#include <mma.h>
using namespace nvcuda;

__global__ void wmma_ker_cuda(half *a, half *b, float *c) {
    for (int i = threadIdx.x; i < 16 * 16; i += blockDim.x) {
        {
            auto origin = ((float *) a)[i];
            ((half *) a)[i] = (half) origin;
        }
        {
            auto origin = ((float *) b)[i];
            ((half *) b)[i] = (half) origin;
        }
    }

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, reinterpret_cast<const __half *>(a), 16);
    wmma::load_matrix_sync(b_frag, reinterpret_cast<const __half *>(b), 16);

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the output
    wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}

void wmma_ker(float *a, float *b, float *c) {
    wmma_ker_cuda<<<1, 32>>>((half*)a, (half*)b, c);
}
