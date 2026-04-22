cmake_minimum_required(VERSION 3.18)
project(InfiniOps LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Internal variable to control `pybind11`'s automatic optimization flags (like `-flto`).
set(PYBIND11_ENABLE_EXTRAS ON)

# Options for backends.
option(WITH_CPU "Enable CPU backend" OFF)
option(WITH_NVIDIA "Enable CUDA backend" OFF)
option(WITH_ILUVATAR "Enable Iluvatar GPU backend" OFF)
option(WITH_METAX "Enable MetaX backend" OFF)
option(WITH_CAMBRICON "Enable Cambricon backend" OFF)
option(WITH_MOORE "Enable Moore backend" OFF)
option(WITH_ASCEND "Enable Ascend backend" OFF)

option(WITH_TORCH "Enable PyTorch C++ backend" OFF)

# Custom `AscendC` kernels under `src/ascend/custom/`.  `ON` by default
# so CI and routine dev builds always exercise `implementation_index=1/2`
# for `RmsNorm` / `AddRmsNorm`.  Gated by `WITH_ASCEND` in
# `src/CMakeLists.txt` — non-Ascend builds ignore it.  Pass
# `-DBUILD_ASCEND_CUSTOM=OFF` to skip the `ccec` build on Ascend
# machines where the custom kernels aren't needed.
#
# When `ON`, `src/CMakeLists.txt` drives the standalone
# `src/ascend/custom/build.sh` via `execute_process` at configure time
# (sidesteps a `CANN` `extract_host_stub.py` path bug that breaks
# in-tree `ascendc_library()` under `scikit-build-core` temp-dir builds)
# and links the produced `libno_workspace_kernel.a` into the `ops`
# module with `--whole-archive`.  Requires `torch_npu` and the
# `AscendC` toolchain (`ccec`).
option(BUILD_ASCEND_CUSTOM "Build custom AscendC kernels" ON)

option(AUTO_DETECT_DEVICES "Automatically detect available devices" OFF)
option(AUTO_DETECT_BACKENDS "Automatically detect available backends" OFF)
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

    if(DEFINED ENV{MUSA_ROOT} OR DEFINED ENV{MUSA_HOME} OR DEFINED ENV{MUSA_PATH})
        set(WITH_MOORE ON)
        set(WITH_MOORE ON CACHE BOOL "Enable Moore backend" FORCE)
        message(STATUS "Auto-detected Moore environment.")
    else()
        set(WITH_MOORE OFF)
        set(WITH_MOORE OFF CACHE BOOL "Enable Moore backend" FORCE)
    endif()

    if(DEFINED ENV{ASCEND_HOME_PATH} OR EXISTS "/dev/davinci0")
        set(WITH_ASCEND ON)
        message(STATUS "Auto-detected Ascend environment.")
    endif()
endif()

if(AUTO_DETECT_BACKENDS)
    message(STATUS "Auto-detecting available backends...")

    find_package(Python COMPONENTS Interpreter QUIET)

    if(Python_FOUND)
        execute_process(
            COMMAND ${Python_EXECUTABLE} -c "import torch"
            RESULT_VARIABLE _torch_import_result
            OUTPUT_QUIET
            ERROR_QUIET
        )

        if(_torch_import_result EQUAL 0)
            set(WITH_TORCH ON)
            message(STATUS "Auto-detected PyTorch.")
        endif()
    endif()
endif()

if(WITH_TORCH)
    find_package(Python COMPONENTS Interpreter REQUIRED)

    # Query `torch` paths directly instead of using `find_package(Torch)`,
    # which pulls in Caffe2's CMake config and may fail on platforms with
    # non-standard CUDA toolchains.
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "from torch.utils.cpp_extension import include_paths; print(';'.join(include_paths()))"
        OUTPUT_VARIABLE TORCH_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _torch_result
    )

    if(NOT _torch_result EQUAL 0)
        message(FATAL_ERROR "`WITH_TORCH` is `ON` but `torch` is not installed.")
    endif()

    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "from torch.utils.cpp_extension import library_paths; print(';'.join(library_paths()))"
        OUTPUT_VARIABLE _torch_lib_dirs
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    find_library(TORCH_LIB torch HINTS ${_torch_lib_dirs} REQUIRED)
    find_library(TORCH_CPU_LIB torch_cpu HINTS ${_torch_lib_dirs} REQUIRED)
    find_library(C10_LIB c10 HINTS ${_torch_lib_dirs} REQUIRED)
    set(TORCH_LIBRARIES ${TORCH_LIB} ${TORCH_CPU_LIB} ${C10_LIB})

    # `auditwheel`-repaired `torch` wheels bundle transitive dependencies
    # (e.g. `libgfortran-<hash>.so`, `libopenblasp-<hash>.so`) in a sibling
    # `torch.libs/` directory that `library_paths()` does not return.  When
    # building against such a wheel, the linker needs this path to resolve
    # the bundled `NEEDED` entries (otherwise: `undefined reference to
    # _gfortran_etime@GFORTRAN_8` etc.).
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import os, torch; d = os.path.dirname(torch.__file__); p = os.path.join(os.path.dirname(d), 'torch.libs'); print(p if os.path.isdir(p) else '')"
        OUTPUT_VARIABLE TORCH_BUNDLED_LIBS_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(TORCH_BUNDLED_LIBS_DIR)
        list(APPEND CMAKE_BUILD_RPATH "${TORCH_BUNDLED_LIBS_DIR}")
        list(APPEND CMAKE_INSTALL_RPATH "${TORCH_BUNDLED_LIBS_DIR}")
        # `rpath-link` is linker-only: lets `ld` resolve the bundled
        # transitive `NEEDED` entries at link time without adding them to our
        # own binary's direct `NEEDED` list.
        add_link_options("-Wl,-rpath-link,${TORCH_BUNDLED_LIBS_DIR}")
        message(STATUS "PyTorch bundled libs: ${TORCH_BUNDLED_LIBS_DIR}")
    endif()

    # Query the `CXX11` ABI setting that `torch` was compiled with.
    # A mismatch causes linker errors (e.g. undefined reference to
    # `c10::Device::Device(std::string const&)`).
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch; print(int(torch.compiled_with_cxx11_abi()))"
        OUTPUT_VARIABLE TORCH_CXX11_ABI
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _torch_abi_result
    )

    if(_torch_abi_result EQUAL 0)
        add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
        message(STATUS "PyTorch CXX11 ABI: ${TORCH_CXX11_ABI}")
    endif()

    message(STATUS "Found PyTorch: ${TORCH_INCLUDE_DIRS}")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Only one CUDA-like GPU backend can be enabled at a time.
set(_gpu_backend_count 0)
foreach(_gpu_backend WITH_NVIDIA WITH_ILUVATAR WITH_METAX WITH_MOORE WITH_ASCEND)
    if(${_gpu_backend})
        math(EXPR _gpu_backend_count "${_gpu_backend_count} + 1")
    endif()
endforeach()

if(_gpu_backend_count GREATER 1)
    message(FATAL_ERROR "`WITH_NVIDIA`, `WITH_ILUVATAR`, `WITH_METAX`, `WITH_MOORE`, and `WITH_ASCEND` are mutually exclusive. Build one GPU backend at a time.")
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
    # `-x ivcore` must not be in `CMAKE_CUDA_FLAGS` — CMake passes those flags
    # to both compile and link steps. During linking, `-x ivcore` causes
    # `clang++` to re-parse `.o` files as source code.
    set(CMAKE_CUDA_FLAGS "--cuda-gpu-arch=${ILUVATAR_ARCH} -fPIC -Wno-error=unused-variable -Wno-error=unused-private-field -Wno-unused-variable" CACHE STRING "Iluvatar CUDA flags")
    set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF CACHE BOOL "Disable RDC for Iluvatar")
    set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING "Iluvatar CUDA architectures (passed via CMAKE_CUDA_FLAGS)")
    # Iluvatar does not ship `libcudadevrt`, which CMake's compiler test
    # tries to link. Disable automatic CUDA runtime linking and link
    # manually via `find_package(CUDAToolkit)` instead.
    set(CMAKE_CUDA_RUNTIME_LIBRARY NONE)
    message(STATUS "Iluvatar: CUDA compiler ${CMAKE_CUDA_COMPILER}, arch ${ILUVATAR_ARCH}")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    # Add `-x ivcore` as a compile-only flag so it is not passed during
    # linking, where it would cause `clang++` to re-parse `.o` files as
    # source.
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-x$<SEMICOLON>ivcore>)
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

if(WITH_MOORE)
    add_compile_definitions(WITH_MOORE=1)

    set(MUSA_ROOT "")
    foreach(_musa_env MUSA_ROOT MUSA_HOME MUSA_PATH)
        if(NOT MUSA_ROOT AND DEFINED ENV{${_musa_env}} AND NOT "$ENV{${_musa_env}}" STREQUAL "")
            set(MUSA_ROOT "$ENV{${_musa_env}}")
        endif()
    endforeach()

    if(NOT MUSA_ROOT AND EXISTS "/usr/local/musa")
        set(MUSA_ROOT "/usr/local/musa")
    endif()

    if(NOT MUSA_ROOT)
        message(FATAL_ERROR "`WITH_MOORE` is `ON` but `MUSA_ROOT`/`MUSA_HOME`/`MUSA_PATH` is not set and `/usr/local/musa` was not found.")
    endif()

    if(NOT EXISTS "${MUSA_ROOT}/bin/mcc")
        message(FATAL_ERROR "Could not find `mcc` under `${MUSA_ROOT}/bin`.")
    endif()

    message(STATUS "Using Moore from `${MUSA_ROOT}`.")

    set(CMAKE_C_COMPILER   ${CMAKE_CURRENT_SOURCE_DIR}/scripts/mcc_wrapper.sh)
    set(CMAKE_CXX_COMPILER ${CMAKE_CURRENT_SOURCE_DIR}/scripts/mcc_wrapper.sh)

    include_directories("${MUSA_ROOT}/include")
    link_directories("${MUSA_ROOT}/lib")

    find_library(MUSA_LIB   NAMES musa   HINTS "${MUSA_ROOT}/lib" REQUIRED)
    find_library(MUSART_LIB NAMES musart HINTS "${MUSA_ROOT}/lib" REQUIRED)
    find_library(MUBLAS_LIB NAMES mublas HINTS "${MUSA_ROOT}/lib" REQUIRED)
endif()

if(WITH_CAMBRICON)
    add_compile_definitions(WITH_CAMBRICON=1)
    set(NEUWARE_HOME $ENV{NEUWARE_HOME})

    include_directories("${NEUWARE_HOME}/include")
    link_directories("${NEUWARE_HOME}/lib")
    link_directories("${NEUWARE_HOME}/lib64")

    # Libraries: `cnrt` / `cnnl` / `cnnl_extra` / `cnpapi`.
    find_library(CAMBRICON_RUNTIME_LIB    NAMES cnrt       HINTS "${NEUWARE_HOME}/lib64" REQUIRED)
    find_library(CAMBRICON_CNNL_LIB       NAMES cnnl       HINTS "${NEUWARE_HOME}/lib64" REQUIRED)
    find_library(CAMBRICON_CNNL_EXTRA_LIB NAMES cnnl_extra HINTS "${NEUWARE_HOME}/lib64" REQUIRED)
    find_library(CAMBRICON_PAPI_LIB       NAMES cnpapi     HINTS "${NEUWARE_HOME}/lib64" REQUIRED)
endif()

if(WITH_ASCEND)
    add_compile_definitions(WITH_ASCEND=1)
    if(NOT DEFINED ASCEND_HOME)
        if(DEFINED ENV{ASCEND_HOME_PATH} AND NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
            set(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}" CACHE PATH "Ascend toolkit root")
        else()
            set(ASCEND_HOME "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "Ascend toolkit root")
        endif()
    endif()
    if(NOT EXISTS "${ASCEND_HOME}")
        message(FATAL_ERROR "`WITH_ASCEND` is ON but `${ASCEND_HOME}` was not found. Set ASCEND_HOME_PATH.")
    endif()
    message(STATUS "Using Ascend from `${ASCEND_HOME}`.")
endif()

# If all other platforms are not enabled, CPU is enabled by default.
if(NOT WITH_NVIDIA AND NOT WITH_ILUVATAR AND NOT WITH_METAX AND NOT WITH_MOORE AND NOT WITH_CAMBRICON AND NOT WITH_ASCEND)
    add_compile_definitions(WITH_CPU=1)
endif()

if(WITH_METAX OR WITH_MOORE)
    set(PYBIND11_ENABLE_EXTRAS OFF)
endif()

add_subdirectory(src)

add_subdirectory(examples)
