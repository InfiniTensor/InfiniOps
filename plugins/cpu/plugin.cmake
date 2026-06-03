infini_ops_register_device(
    NAME cpu
    CMAKE_ENTRY plugin.cmake
    DEVICES cpu
    SOURCE_ROOTS src/native/cpu
    OPERATOR_ROOTS src/native/cpu/ops
    DEVICE_HEADERS cpu=native/cpu/device_.h
    TEST_DEVICES cpu=cpu)

file(GLOB_RECURSE CPU_SOURCES CONFIGURE_DEPENDS
    "${INFINI_OPS_SRC_DIR}/native/cpu/*.cc"
    "${INFINI_OPS_SRC_DIR}/native/cpu/*.cpp")
if(CPU_SOURCES)
    target_sources(infiniops PRIVATE ${CPU_SOURCES})
endif()

target_compile_definitions(infiniops PUBLIC WITH_CPU=1)

find_package(OpenMP REQUIRED COMPONENTS CXX)
target_link_libraries(infiniops PRIVATE OpenMP::OpenMP_CXX)
