//
// Created by 63479 on 2022/9/21.
//

#ifndef ARM64_TEST_UTILS_ASSERT_CUH
#define ARM64_TEST_UTILS_ASSERT_CUH

// this requires compute capability 2.x or higher, and is not supported on MacOS. For more details see CUDA C Programming Guide, Section B.16.
#if (__CUDACC_VER_MAJOR__ >= 2 || CUDA_VERSION >= 2000) && !_NVHPC_CUDA && USE_ASSERT
#include <assert.h>
#else
#undef assert
#define assert(condition) \
  if (!(condition)) { return; }
#endif

#endif //ARM64_TEST_UTILS_ASSERT_CUH
