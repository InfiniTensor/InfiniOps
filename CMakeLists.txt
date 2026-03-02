cmake_minimum_required(VERSION 3.18)
project(InfiniOps LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Internal variable to control pybind11's automatic optimization flags (like `-flto`).
set(PYBIND11_ENABLE_EXTRAS ON)

# Options for backends.
option(WITH_CPU "Enable CPU backend" OFF)
option(WITH_NVIDIA "Enable CUDA backend" OFF)
option(WITH_ILUVATAR "Enable Iluvatar GPU backend" OFF)
option(WITH_METAX "Enable MetaX backend" OFF)
option(WITH_CAMBRICON "Enable Cambricon backend" OFF)

option(AUTO_DETECT_DEVICES "Automatically detect available devices" OFF)
option(GENERATE_PYTHON_BINDINGS "Generate Python bindings" OFF)

if(AUTO_DETECT_DEVICES)
    message(STATUS "Auto-detecting available devices...")

    set(WITH_CPU ON)

    file(GLOB NVIDIA_DEV_FILES "/dev/nvidia*")

    if(NVIDIA_DEV_FILES)
        set(WITH_NVIDIA ON)
        message(STATUS "Auto-detected NVIDIA environment.")
    endif()

    file(GLOB ILUVATAR_DEV_FILES "/dev/iluvatar*")

    if(ILUVATAR_DEV_FILES)
        set(WITH_ILUVATAR ON)
        message(STATUS "Auto-detected Iluvatar environment.")
    endif()

    if(DEFINED ENV{MACA_PATH})
        set(WITH_METAX ON)
        message(STATUS "Auto-detected MetaX environment from MACA_PATH")
    else()
        execute_process(
            COMMAND sh -c "grep -h 9999 /sys/bus/pci/devices/*/vendor 2>/dev/null"
            OUTPUT_VARIABLE _pci_vendor_output
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        string(FIND "${_pci_vendor_output}" "9999" _found_pos)

        if(_found_pos GREATER -1)
            set(WITH_METAX ON)
            message(STATUS "Detected MetaX GPU from PCI vendor ID 0x9999")
        else()
            set(WITH_METAX OFF)
            message(STATUS "No MetaX GPU detected")
        endif()
    endif()

    if(DEFINED ENV{NEUWARE_HOME})
        set(WITH_CAMBRICON ON)
        message(STATUS "Auto-detected Cambricon environment.")
    endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# NVIDIA and Iluvatar are parallel backends; only one GPU backend at a time.
if(WITH_NVIDIA AND WITH_ILUVATAR)
    message(FATAL_ERROR "`WITH_NVIDIA` and `WITH_ILUVATAR` cannot both be `ON`. Build one GPU backend at a time.")
endif()

if(WITH_NVIDIA)
    add_compile_definitions(WITH_NVIDIA=1)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# Iluvatar: CUDA-compatible device, uses `clang++` with `-x ivcore` (not `nvcc`).
# Reference: `InfiniCore` `xmake/iluvatar.lua`.
if(WITH_ILUVATAR)
    add_compile_definitions(WITH_ILUVATAR=1)
    set(ILUVATAR_ARCH "ivcore20" CACHE STRING "Iluvatar GPU architecture")
    find_program(CLANGXX NAMES clang++)
    if(CLANGXX)
        set(CMAKE_CUDA_COMPILER "${CLANGXX}" CACHE STRING "Iluvatar CUDA compiler (clang++)")
    else()
        set(CMAKE_CUDA_COMPILER "clang++" CACHE STRING "Iluvatar CUDA compiler (clang++)")
    endif()
    set(CMAKE_CUDA_FLAGS "-x ivcore -std=c++17 --cuda-gpu-arch=${ILUVATAR_ARCH} -fPIC -Wno-error=unused-variable -Wno-error=unused-private-field -Wno-unused-variable" CACHE STRING "Iluvatar CUDA flags")
    set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF CACHE BOOL "Disable RDC for Iluvatar")
    message(STATUS "Iluvatar: CUDA compiler ${CMAKE_CUDA_COMPILER}, arch ${ILUVATAR_ARCH}")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(WITH_METAX)
    add_compile_definitions(WITH_METAX=1)

    # Normally can be found at: `/opt/maca/`.
    set(MACA_PATH $ENV{MACA_PATH})
    set(CMAKE_C_COMPILER   ${CMAKE_CURRENT_SOURCE_DIR}/scripts/mxcc_wrapper.sh)
    set(CMAKE_CXX_COMPILER ${CMAKE_CURRENT_SOURCE_DIR}/scripts/mxcc_wrapper.sh)

    include_directories("${MACA_PATH}/include")
    link_directories("${MACA_PATH}/lib")

    # Libraries: mcruntime / mcdnn / mcblas.
    find_library(MACA_RUNTIME_LIB NAMES mcruntime HINTS "${MACA_PATH}/lib" REQUIRED)
    find_library(MACA_DNN_LIB     NAMES mcdnn     HINTS "${MACA_PATH}/lib" REQUIRED)
    find_library(MACA_BLAS_LIB    NAMES mcblas    HINTS "${MACA_PATH}/lib" REQUIRED)
endif()

if(WITH_CAMBRICON)
    add_compile_definitions(WITH_CAMBRICON=1)
    set(NEUWARE_HOME $ENV{NEUWARE_HOME})

    include_directories("${NEUWARE_HOME}/include")
    link_directories("${NEUWARE_HOME}/lib")
    link_directories("${NEUWARE_HOME}/lib64")

    # Libraries: cnrt / cnnl / cnnl_extra / cnpapi.
    find_library(CAMBRICON_RUNTIME_LIB NAMES cnrt HINTS "${NEUWARE_HOME}/lib64" REQUIRED)
    find_library(CAMBRICON_CNNL_LIB     NAMES cnnl HINTS "${NEUWARE_HOME}/lib64" REQUIRED)
    find_library(CAMBRICON_CNNL_EXTRA_LIB   NAMES cnnl_extra HINTS "${NEUWARE_HOME}/lib64" REQUIRED)
    find_library(CAMBRICON_PAPI_LIB    NAMES cnpapi HINTS "${NEUWARE_HOME}/lib64" REQUIRED)
endif()

# If all other platforms are not enabled, CPU is enabled by default.
if(NOT WITH_NVIDIA AND NOT WITH_ILUVATAR AND NOT WITH_METAX)
    add_compile_definitions(WITH_CPU=1)
endif()

if(WITH_METAX)
    set(PYBIND11_ENABLE_EXTRAS OFF)
endif()

add_subdirectory(src)

add_subdirectory(examples)
