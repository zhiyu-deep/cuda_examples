set(CMAKE_CUDA_STANDARD 11)

# NOTE: cross compiling requires env `CUDA_TOOLKIT_ROOT`
if(CUDA_TOOLKIT_ROOT_DIR AND NOT DEFINED ENV{CUDA_TOOLKIT_ROOT})
    set(ENV{CUDA_TOOLKIT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR})
elseif(DEFINED ENV{CUDA_TOOLKIT_ROOT} AND NOT CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_TOOLKIT_ROOT})
endif()

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
#find_package(CUDA REQUIRED)

# NOTE: `CMAKE_CUDA_COMPILER` and `CMAKE_CUDA_HOST_COMPILER` MUST be placed before enable_language(CUDA) for
# recognizing arch and os
if(MSVC)
    if(NOT CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe)
    endif()
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    enable_language(CUDA)

    if(HAVE_CXX_FLAG_UTF_8)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=/utf-8")
    endif()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=/wd4819,/wd4828")
    set(CUDA_NVCC_FLAGS_DEBUG "-g -O0")
    set(CUDA_NVCC_FLAGS_RELEASE "-O3")
else()
    if(NOT CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
    endif()
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    enable_language(CUDA)

    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=-fPIC,-Wall,-fvisibility=hidden")
    set(CUDA_NVCC_FLAGS_DEBUG "-g")
    set(CUDA_NVCC_FLAGS_RELEASE "-O3")
endif()

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include SYSTEM)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib SYSTEM)

# 默认sm_70.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80 ${CUDA_NVCC_FLAGS}")

macro(hpcc_cuda_use_msvc_static_runtime)
    string(REPLACE -MD -MT CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG}")
    string(REPLACE -MD -MT CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")
endmacro()
