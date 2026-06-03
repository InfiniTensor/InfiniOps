infini_ops_enable_plugin(cuda-common)
infini_ops_register_device(
    NAME moore
    CMAKE_ENTRY plugin.cmake
    DEVICES moore
    DEPENDS cuda-common
    SOURCE_ROOTS src/native/cuda/moore
    OPERATOR_ROOTS src/native/cuda/moore/ops
    DEVICE_HEADERS moore=native/cuda/moore/device_.h
    TEST_DEVICES moore=musa)

file(GLOB_RECURSE MOORE_SOURCES CONFIGURE_DEPENDS
    "${INFINI_OPS_SRC_DIR}/native/cuda/*.cc"
    "${INFINI_OPS_SRC_DIR}/native/cuda/*.cpp"
    "${INFINI_OPS_SRC_DIR}/native/cuda/moore/*.cc"
    "${INFINI_OPS_SRC_DIR}/native/cuda/moore/*.cpp"
    "${INFINI_OPS_SRC_DIR}/native/cuda/moore/*.mu")

set_source_files_properties(${MOORE_SOURCES} PROPERTIES LANGUAGE CXX)

target_compile_definitions(infiniops PRIVATE WITH_MOORE=1)
target_compile_options(infiniops PRIVATE "-x" "musa")
target_sources(infiniops PRIVATE ${MOORE_SOURCES})

target_include_directories(infiniops PUBLIC "${MUSA_ROOT}/include")
target_link_libraries(infiniops PUBLIC ${MUSA_LIB} ${MUSART_LIB} ${MUBLAS_LIB})
