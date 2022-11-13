cmake_minimum_required(VERSION 3.0)
project(arm64_test)

include(cuda-common.cmake)

include_directories(${PROJECT_SOURCE_DIR})



#=======================functions=========================================
# 1. scan
file(GLOB scan_files scan/*.cu scan/*.cpp)
add_library(scan STATIC "${scan_files}")
# 2. gemm
file(GLOB gemm_files gemm/*.cu gemm/*.cpp)
add_library(gemm STATIC "${gemm_files}")
target_include_directories(gemm INTERFACE ${CMAKE_CURRENT_LIST_DIR}/gemm)
# 3. conv
FILE(GLOB conv_files conv/*)
add_library(conv STATIC ${conv_files})
# 4. transpose
file(GLOB transpose_files transpose/*.cu transpose/*.cpp)
add_library(transpose STATIC "${transpose_files}")
include_directories(transpose)
# 5. wmma
file(GLOB wmma_files wmma/*)
add_library(wmma STATIC "${wmma_files}")


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
        PRIVATE conv
        PRIVATE transpose
        PRIVATE scan
        PRIVATE wmma)
