infini_ops_enable_plugin(cuda-common)
infini_ops_register_device(
    NAME hygon
    CMAKE_ENTRY plugin.cmake
    DEVICES hygon
    DEPENDS cuda-common
    SOURCE_ROOTS src/native/cuda/hygon
    OPERATOR_ROOTS src/native/cuda/hygon/ops
    DEVICE_HEADERS hygon=native/cuda/hygon/device_.h
    TEST_DEVICES hygon=cuda)

file(GLOB_RECURSE HYGON_SOURCES CONFIGURE_DEPENDS
    "${INFINI_OPS_SRC_DIR}/native/cuda/*.cc"
    "${INFINI_OPS_SRC_DIR}/native/cuda/*.cpp"
    "${INFINI_OPS_SRC_DIR}/native/cuda/*.cu"
    "${INFINI_OPS_SRC_DIR}/native/cuda/hygon/*.cc"
    "${INFINI_OPS_SRC_DIR}/native/cuda/hygon/*.cpp"
    "${INFINI_OPS_SRC_DIR}/native/cuda/hygon/*.cu")

enable_language(CUDA)
target_compile_definitions(infiniops PUBLIC WITH_HYGON=1)
target_sources(infiniops PRIVATE ${HYGON_SOURCES})

find_package(CUDAToolkit REQUIRED)
target_link_libraries(infiniops PUBLIC CUDA::cudart CUDA::cublas)
set_target_properties(infiniops PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
