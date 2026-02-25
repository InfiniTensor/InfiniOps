cmake_minimum_required(VERSION 3.18)
project(InfiniOps LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options for backends.
option(WITH_CPU "Enable CPU backend" OFF)
option(WITH_NVIDIA "Enable CUDA backend" OFF)
option(WITH_METAX "Enable MetaX backend" OFF)

option(GENERATE_PYTHON_BINDINGS "Generate Python bindings" OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

if(WITH_NVIDIA)
    add_compile_definitions(WITH_NVIDIA=1)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
elseif(WITH_METAX)
    add_compile_definitions(WITH_METAX=1)

    # Normally can be found at: `/opt/maca/`.
    set(MACA_PATH $ENV{MACA_PATH})
    set(CMAKE_C_COMPILER ${MACA_PATH}/mxgpu_llvm/bin/mxcc)
    set(CMAKE_CXX_COMPILER ${MACA_PATH}/mxgpu_llvm/bin/mxcc)

    include_directories("${MACA_PATH}/include")
    link_directories("${MACA_PATH}/lib")

    # Libraries: mcruntime / mcdnn / mcblas.
    find_library(MACA_RUNTIME_LIB NAMES mcruntime HINTS "${MACA_PATH}/lib" REQUIRED)
    find_library(MACA_DNN_LIB     NAMES mcdnn     HINTS "${MACA_PATH}/lib" REQUIRED)
    find_library(MACA_BLAS_LIB    NAMES mcblas    HINTS "${MACA_PATH}/lib" REQUIRED)
# If all other platforms are not enabled, CPU is enabled by default.
else()
    add_compile_definitions(WITH_CPU=1)
endif()

add_subdirectory(src)

add_subdirectory(examples)
