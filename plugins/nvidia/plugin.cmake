infini_ops_enable_plugin(cuda-common)
infini_ops_register_device(
    NAME nvidia
    CMAKE_ENTRY plugin.cmake
    DEVICES nvidia
    DEPENDS cuda-common
    SOURCE_ROOTS src/native/cuda/nvidia
    OPERATOR_ROOTS src/native/cuda/nvidia/ops
    DEVICE_HEADERS nvidia=native/cuda/nvidia/device_.h
    TEST_DEVICES nvidia=cuda)

file(GLOB_RECURSE NVIDIA_SOURCES CONFIGURE_DEPENDS
    "${INFINI_OPS_SRC_DIR}/native/cuda/*.cc"
    "${INFINI_OPS_SRC_DIR}/native/cuda/*.cpp"
    "${INFINI_OPS_SRC_DIR}/native/cuda/*.cu"
    "${INFINI_OPS_SRC_DIR}/native/cuda/nvidia/*.cc"
    "${INFINI_OPS_SRC_DIR}/native/cuda/nvidia/*.cpp"
    "${INFINI_OPS_SRC_DIR}/native/cuda/nvidia/*.cu")

enable_language(CUDA)
target_compile_definitions(infiniops PUBLIC WITH_NVIDIA=1)
target_sources(infiniops PRIVATE ${NVIDIA_SOURCES})

find_package(CUDAToolkit REQUIRED)
target_link_libraries(infiniops PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cuda_driver)
set_target_properties(infiniops PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
