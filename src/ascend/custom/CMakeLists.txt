# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# Copyright (c) 2025 InfiniTensor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Adapted from https://github.com/vllm-project/vllm-ascend/blob/main/csrc/CMakeLists.txt

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(ascend-kernel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
endif()

add_compile_options(-Wunused-value -Wcast-align -Wcast-qual -Wwrite-strings
                    -Wsign-compare -Wextra)

if(${CMAKE_BUILD_TYPE} MATCHES "RELEASE")
    add_compile_options(-O3 -fvisibility=hidden -fvisibility-inlines-hidden
                        -fstack-protector-strong -fPIE -fPIC)
    message(STATUS "build type set to RELEASE")
else()
    add_compile_options(-g -rdynamic)
endif()

set(PROJECT_OP_SRC_BASE ${PROJECT_SOURCE_DIR})
set(PROJECT_BUILD_PATH ${PROJECT_SOURCE_DIR}/build)
set(PROJECT_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/output)

include(cmake/config_envs.cmake)
include(cmake/config_ascend.cmake)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# Shared library output location.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_PATH})

# Host side files.
file(GLOB OP_SRCS
    ${PROJECT_OP_SRC_BASE}/torch_binding.cpp
    ${PROJECT_OP_SRC_BASE}/rms_norm/op_host/rms_norm.cpp
)

# Shared library name — consumed by `kernel_custom.h` variants and by the
# Python side via `import ascend_kernel`.
set(OP_PLUGIN_NAME ascend_kernel)

# Kernel side files (device code compiled by `AscendC` toolchain).
ascendc_library(no_workspace_kernel STATIC
    ${PROJECT_OP_SRC_BASE}/rms_norm/op_kernel/rms_norm.cpp
)

# Create shared library libascend_kernel.so.
add_library(${OP_PLUGIN_NAME} SHARED ${OP_SRCS})

target_link_libraries(${OP_PLUGIN_NAME} PRIVATE
    no_workspace_kernel
    torch_npu
    ascendcl
    tiling_api
    nnopbase
    opapi
    register
    platform
    ascendalog
    dl
)

target_link_directories(${OP_PLUGIN_NAME} PRIVATE
    ${TORCH_DIR}/lib
    ${TORCH_NPU_DIR}/lib
)

target_include_directories(${OP_PLUGIN_NAME} PRIVATE
    ${PROJECT_OP_SRC_BASE}/utils
    ${PROJECT_SOURCE_DIR}/include
    ${TORCH_DIR}/include
    ${TORCH_DIR}/include/torch/csrc/api/include
    ${TORCH_NPU_DIR}/include/third_party/acl/inc
    ${TORCH_NPU_DIR}/include/third_party/hccl/inc
    ${TORCH_NPU_DIR}/include
    ${PYTHON_INCLUDE_DIR}
    ${ASCEND_INCLUDE_DIR}/external
    ${ASCEND_INCLUDE_DIR}/experiment/platform
    ${ASCEND_INCLUDE_DIR}/experiment/runtime
)
