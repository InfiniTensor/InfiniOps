cmake_minimum_required(VERSION 3.18)
project(InfiniOps LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options for backends
option(USE_CPU "Enable CPU backend" OFF)
option(USE_NVIDIA "Enable CUDA backend" OFF)
option(USE_METAX "Enable MetaX backend" OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

if(USE_NVIDIA)
    add_compile_definitions(USE_NVIDIA=1)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

elseif(USE_METAX)
    add_compile_definitions(USE_METAX=1)

    # Normally can be found at: /opt/maca/
    set(MACA_PATH $ENV{MACA_PATH})
    set(CMAKE_C_COMPILER ${MACA_PATH}/mxgpu_llvm/bin/mxcc)
    set(CMAKE_CXX_COMPILER ${MACA_PATH}/mxgpu_llvm/bin/mxcc)

    include_directories("${MACA_PATH}/include")
    link_directories("${MACA_PATH}/lib")

    # Libraries: mcruntime / mcdnn / mcblas
    find_library(MACA_RUNTIME_LIB NAMES mcruntime HINTS "${MACA_PATH}/lib" REQUIRED)
    find_library(MACA_DNN_LIB     NAMES mcdnn     HINTS "${MACA_PATH}/lib" REQUIRED)
    find_library(MACA_BLAS_LIB    NAMES mcblas    HINTS "${MACA_PATH}/lib" REQUIRED)
   
# if all other platforms are not enabled, default enable CPU
else()
    add_compile_definitions(USE_CPU=1)
    
endif()

add_subdirectory(src)

add_subdirectory(examples)
