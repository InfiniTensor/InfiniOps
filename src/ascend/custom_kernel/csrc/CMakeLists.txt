# Set the library output dir to the project output for linking.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_PATH})

# Host side files.
file(GLOB OP_SRCS
    ${PROJECT_OP_SRC_BASE}/register.cpp
    ${PROJECT_OP_SRC_BASE}/ops/rms_norm/op_host/rms_norm.cpp
)

# Set the shared library name.
set(OP_PLUGIN_NAME ascend_kernel)

# Kernel side files (device code compiled by AscendC toolchain).
ascendc_library(no_workspace_kernel STATIC
    ${PROJECT_OP_SRC_BASE}/ops/rms_norm/op_kernel/rms_norm.cpp
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
