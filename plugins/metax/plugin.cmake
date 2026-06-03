infini_ops_enable_plugin(cuda-common)
infini_ops_register_device(
    NAME metax
    CMAKE_ENTRY plugin.cmake
    DEVICES metax
    DEPENDS cuda-common
    SOURCE_ROOTS src/native/cuda/metax
    OPERATOR_ROOTS src/native/cuda/metax/ops
    DEVICE_HEADERS metax=native/cuda/metax/device_.h
    TEST_DEVICES metax=cuda)

file(GLOB_RECURSE METAX_SOURCES CONFIGURE_DEPENDS
    "${INFINI_OPS_SRC_DIR}/native/cuda/*.cc"
    "${INFINI_OPS_SRC_DIR}/native/cuda/*.cpp"
    "${INFINI_OPS_SRC_DIR}/native/cuda/metax/*.cc"
    "${INFINI_OPS_SRC_DIR}/native/cuda/metax/*.maca")

set_source_files_properties(${METAX_SOURCES} PROPERTIES LANGUAGE CXX)

target_compile_definitions(infiniops PRIVATE WITH_METAX=1)
target_compile_options(infiniops PRIVATE "-x" "maca")
target_sources(infiniops PRIVATE ${METAX_SOURCES})

target_include_directories(infiniops PUBLIC "${MACA_PATH}/include")
target_link_libraries(infiniops PUBLIC
    ${MACA_RUNTIME_LIB}
    ${MACA_DNN_LIB}
    ${MACA_BLAS_LIB})
