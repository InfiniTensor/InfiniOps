infini_ops_enable_plugin(cuda-common)
infini_ops_register_device(
    NAME iluvatar
    CMAKE_ENTRY plugin.cmake
    DEVICES iluvatar
    DEPENDS cuda-common
    SOURCE_ROOTS src/native/cuda/iluvatar
    OPERATOR_ROOTS src/native/cuda/iluvatar/ops
    DEVICE_HEADERS iluvatar=native/cuda/iluvatar/device_.h
    TEST_DEVICES iluvatar=cuda)

file(GLOB_RECURSE ILUVATAR_SOURCES CONFIGURE_DEPENDS
    "${INFINI_OPS_SRC_DIR}/native/cuda/*.cc"
    "${INFINI_OPS_SRC_DIR}/native/cuda/*.cpp"
    "${INFINI_OPS_SRC_DIR}/native/cuda/*.cu"
    "${INFINI_OPS_SRC_DIR}/native/cuda/iluvatar/*.cc"
    "${INFINI_OPS_SRC_DIR}/native/cuda/iluvatar/*.cpp"
    "${INFINI_OPS_SRC_DIR}/native/cuda/iluvatar/*.cu")

target_compile_definitions(infiniops PUBLIC WITH_ILUVATAR=1)
target_sources(infiniops PRIVATE ${ILUVATAR_SOURCES})

find_package(CUDAToolkit REQUIRED)
target_link_libraries(infiniops PUBLIC CUDA::cudart CUDA::cublas CUDA::cuda_driver)
