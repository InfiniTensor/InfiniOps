infini_ops_register_device(
    NAME cambricon
    CMAKE_ENTRY plugin.cmake
    DEVICES cambricon
    SOURCE_ROOTS src/native/cambricon
    OPERATOR_ROOTS src/native/cambricon/ops
    DEVICE_HEADERS cambricon=native/cambricon/device_.h
    TEST_DEVICES cambricon=mlu)

file(GLOB_RECURSE CAMBRICON_MLU_SOURCES CONFIGURE_DEPENDS
    "${INFINI_OPS_SRC_DIR}/native/cambricon/ops/*/*.mlu")
find_program(CNCC_COMPILER cncc HINTS "${NEUWARE_HOME}/bin" "$ENV{NEUWARE_HOME}/bin" /usr/local/neuware/bin)
if(CNCC_COMPILER)
    message(STATUS "Found cncc: ${CNCC_COMPILER}")
    set(MLU_COMPILE_OPTS
        -c --bang-mlu-arch=mtp_592 -O3 -fPIC -Wall -Werror -std=c++17 -pthread
        -I${INFINI_OPS_SRC_DIR} -I${NEUWARE_HOME}/include
        -idirafter /usr/local/neuware/lib/clang/11.1.0/include)
    function(compile_mlu_file src_file)
        get_filename_component(name ${src_file} NAME_WE)
        get_filename_component(path ${src_file} DIRECTORY)
        file(RELATIVE_PATH rel_path "${INFINI_OPS_SRC_DIR}" "${path}")
        set(out_file "${CMAKE_CURRENT_BINARY_DIR}/${rel_path}/${name}.o")
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${rel_path}")
        add_custom_command(OUTPUT ${out_file}
            COMMAND ${CNCC_COMPILER} ${MLU_COMPILE_OPTS} -c ${src_file} -o ${out_file}
            DEPENDS ${src_file}
            COMMENT "Building MLU kernel: ${src_file}")
        set_property(DIRECTORY APPEND PROPERTY CAMBRICON_OBJECTS ${out_file})
    endfunction()
    foreach(src ${CAMBRICON_MLU_SOURCES})
        compile_mlu_file(${src})
    endforeach()
    get_directory_property(CAMBRICON_OBJECT_FILES CAMBRICON_OBJECTS)
    if(CAMBRICON_OBJECT_FILES)
        target_sources(infiniops PRIVATE ${CAMBRICON_OBJECT_FILES})
    endif()
else()
    message(WARNING "cncc compiler not found. MLU kernels will not be compiled.")
endif()

target_compile_definitions(infiniops PRIVATE WITH_CAMBRICON=1)
target_include_directories(infiniops PUBLIC "${NEUWARE_HOME}/include")
target_link_libraries(infiniops PUBLIC
    ${CAMBRICON_RUNTIME_LIB}
    ${CAMBRICON_CNNL_LIB}
    ${CAMBRICON_CNNL_EXTRA_LIB}
    ${CAMBRICON_PAPI_LIB})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(infiniops PUBLIC
        "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-idirafter /usr/local/neuware/lib/clang/11.1.0/include>")
endif()
