infini_ops_register_device(
    NAME ascend
    CMAKE_ENTRY plugin.cmake
    DEVICES ascend
    SOURCE_ROOTS src/native/ascend
    OPERATOR_ROOTS src/native/ascend/ops
    DEVICE_HEADERS ascend=native/ascend/device_.h
    TEST_DEVICES ascend=npu)

# ASCEND_HOME is set by the top-level CMakeLists.txt.
file(GLOB_RECURSE ASCEND_SOURCES CONFIGURE_DEPENDS
    "${INFINI_OPS_SRC_DIR}/native/ascend/*.cc"
    "${INFINI_OPS_SRC_DIR}/native/ascend/*.cpp")
# Exclude `kernel_impl.cpp`: `AscendC` device code, not compiled by the host C++ compiler.
list(FILTER ASCEND_SOURCES EXCLUDE REGEX ".*kernel_impl\\.cpp$")
# Exclude custom/: standalone PyTorch extension, built separately.
list(FILTER ASCEND_SOURCES EXCLUDE REGEX ".*/custom/.*")

target_compile_definitions(infiniops PUBLIC WITH_ASCEND=1)
target_sources(infiniops PRIVATE ${ASCEND_SOURCES})

# Resolve the driver lib dir two levels above the toolkit root.
get_filename_component(ASCEND_ROOT "${ASCEND_HOME}/../.." ABSOLUTE)

# Prefer the real driver HAL; fall back to the toolkit stub for build-only
# environments (e.g., Docker CI images without hardware drivers installed).
# CANN <= 8.0: stub at runtime/lib64/stub/; CANN >= 8.5: devlib/<arch>-linux/devlib/.
set(ASCEND_HAL_REAL   "${ASCEND_ROOT}/driver/lib64/driver/libascend_hal.so")
set(ASCEND_HAL_STUB   "${ASCEND_HOME}/runtime/lib64/stub/libascend_hal.so")
set(ASCEND_HAL_DEVLIB "${ASCEND_HOME}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/libascend_hal.so")
if(EXISTS "${ASCEND_HAL_REAL}")
    set(ASCEND_HAL_LIB "${ASCEND_HAL_REAL}")
elseif(EXISTS "${ASCEND_HAL_STUB}")
    set(ASCEND_HAL_LIB "${ASCEND_HAL_STUB}")
    message(STATUS "ascend_hal: driver not found, using stub for linking")
elseif(EXISTS "${ASCEND_HAL_DEVLIB}")
    set(ASCEND_HAL_LIB "${ASCEND_HAL_DEVLIB}")
    message(STATUS "ascend_hal: driver not found, using devlib for linking")
else()
    message(FATAL_ERROR "libascend_hal.so not found (tried ${ASCEND_HAL_REAL}, ${ASCEND_HAL_STUB}, and ${ASCEND_HAL_DEVLIB})")
endif()

target_include_directories(infiniops PUBLIC
    "${ASCEND_HOME}/include"
    "${ASCEND_HOME}/include/aclnn"
    "${ASCEND_HOME}/include/aclnnop")
target_link_libraries(infiniops PUBLIC
    "${ASCEND_HOME}/lib64/libascendcl.so"
    "${ASCEND_HOME}/lib64/libnnopbase.so"
    "${ASCEND_HOME}/lib64/libopapi.so"
    "${ASCEND_HAL_LIB}")

# ATB (Ascend Transformer Boost) provides fused operators like
# `PagedAttention` and `ReshapeAndCache` that are graph-capture safe.
set(ATB_HOME_DIR "$ENV{ATB_HOME_PATH}")
if(NOT ATB_HOME_DIR)
    # Default search path under CANN nnal directory.
    file(GLOB ATB_SEARCH_DIRS "/usr/local/Ascend/nnal/atb/*/atb/cxx_abi_1")
    if(ATB_SEARCH_DIRS)
        list(SORT ATB_SEARCH_DIRS ORDER DESCENDING)
        list(GET ATB_SEARCH_DIRS 0 ATB_HOME_DIR)
    endif()
endif()

if(ATB_HOME_DIR AND EXISTS "${ATB_HOME_DIR}/include/atb/operation.h")
    message(STATUS "ATB found: ${ATB_HOME_DIR}")
    target_compile_definitions(infiniops PUBLIC INFINI_HAS_ATB=1)
    target_include_directories(infiniops PUBLIC "${ATB_HOME_DIR}/include")
    target_link_libraries(infiniops PUBLIC "${ATB_HOME_DIR}/lib/libatb.so")
else()
    message(STATUS "ATB not found - ATB-based operators disabled")
endif()

# Custom `AscendC` kernels (PyTorch extension, requires `torch_npu`).
if(BUILD_CUSTOM_KERNEL)
    add_subdirectory(
        "${INFINI_OPS_SRC_DIR}/native/ascend/custom"
        "${CMAKE_CURRENT_BINARY_DIR}/native/ascend/custom")

    # Link the compiled `AscendC` kernel objects into `infiniops` so that
    # custom kernel implementations (e.g. `RmsNorm` index 1) can call
    # them via the generated launch functions.
    target_compile_definitions(infiniops PUBLIC INFINI_HAS_CUSTOM_KERNELS=1)
endif()
