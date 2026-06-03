infini_ops_register_plugin(
    NAME cuda-common
    KIND shared
    CONTRACT_VERSION 1
    CMAKE_ENTRY plugin.cmake
    SOURCE_ROOTS src/native/cuda
    OPERATOR_ROOTS src/native/cuda/ops)
