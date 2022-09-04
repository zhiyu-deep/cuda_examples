cmake_minimum_required(VERSION 3.0)
project(arm64_test)

include(cuda-common.cmake)

include_directories(${PROJECT_SOURCE_DIR})

# 1. scan
add_library(scan STATIC
        scan/scan_impl.cu
        )
# 2. gemm
file(GLOB gemm_files gemm/*.cu gemm/*.cpp)
add_library(gemm STATIC "${gemm_files}")
target_include_directories(gemm INTERFACE ${CMAKE_CURRENT_LIST_DIR}/gemm)
# 3. conv
add_library(conv STATIC
        conv/myImg2Col.cu)



# ==============================main=============================
file(GLOB common_files common/*)
include_directories(common)
add_executable(arm64_test main.cpp "${common_files}" register.cpp)

set_target_properties(arm64_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(arm64_test
        PRIVATE cudart
        PRIVATE scan
        PRIVATE gemm
        PRIVATE conv)
