add_library(infiniops SHARED)

file(GLOB BASE_SRCS CONFIGURE_DEPENDS "*.cc")
target_sources(infiniops PRIVATE ${BASE_SRCS})

set(DEVICE_LIST "")

if(WITH_CPU)
    set(CPU_PATTERNS
        "cpu/*.cc"
        "cpu/*.cpp"
    )

    file(GLOB_RECURSE CPU_SOURCES CONFIGURE_DEPENDS ${CPU_PATTERNS})
    list(APPEND CORE_SOURCES ${CPU_SOURCES})

    target_compile_definitions(infiniops PUBLIC WITH_CPU=1)

    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries(infiniops PRIVATE OpenMP::OpenMP_CXX)

    list(APPEND DEVICE_LIST "cpu")
endif()

if(WITH_NVIDIA)
    set(NVIDIA_PATTERNS
        "cuda/*.cc"
        "cuda/*.cpp"
        "cuda/*.cu"
        "nvidia/*.cc"
        "nvidia/*.cpp"
        "nvidia/*.cu"
    )

    file(GLOB_RECURSE NVIDIA_SOURCES CONFIGURE_DEPENDS ${NVIDIA_PATTERNS})

    enable_language(CUDA)

    target_compile_definitions(infiniops PUBLIC WITH_NVIDIA=1)
    target_sources(infiniops PRIVATE ${NVIDIA_SOURCES})

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(infiniops PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cuda_driver)

    list(APPEND DEVICE_LIST "nvidia")
    set_target_properties(infiniops PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endif()

if(WITH_ILUVATAR)
    set(ILUVATAR_PATTERNS
        "cuda/*.cc"
        "cuda/*.cpp"
        "cuda/*.cu"
        "iluvatar/*.cc"
        "iluvatar/*.cpp"
        "iluvatar/*.cu"
    )

    file(GLOB_RECURSE ILUVATAR_SOURCES CONFIGURE_DEPENDS ${ILUVATAR_PATTERNS})

    enable_language(CUDA)

    target_compile_definitions(infiniops PUBLIC WITH_ILUVATAR=1)
    target_sources(infiniops PRIVATE ${ILUVATAR_SOURCES})

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(infiniops PUBLIC CUDA::cudart CUDA::cublas CUDA::cuda_driver)

    set_target_properties(infiniops PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )

    list(APPEND DEVICE_LIST "iluvatar")
endif()

if(WITH_METAX)
    set(METAX_PATTERNS
        "cuda/*.cc"
        "cuda/*.cpp"
        "metax/*.cc"
        "metax/*.maca"
    )

    file(GLOB_RECURSE METAX_SOURCES CONFIGURE_DEPENDS ${METAX_PATTERNS})

    set_source_files_properties(${METAX_SOURCES} PROPERTIES LANGUAGE CXX)

    target_compile_definitions(infiniops PRIVATE WITH_METAX=1)
    target_compile_options(infiniops PUBLIC "-x" "maca")
    target_sources(infiniops PRIVATE ${METAX_SOURCES})

    target_include_directories(infiniops PUBLIC "${MACA_PATH}/include")
    target_link_libraries(infiniops PUBLIC 
        ${MACA_RUNTIME_LIB} 
        ${MACA_DNN_LIB} 
        ${MACA_BLAS_LIB}
    )

    list(APPEND DEVICE_LIST "metax")
endif()

if(WITH_MOORE)
    set(MOORE_PATTERNS
        "cuda/*.cc"
        "cuda/*.cpp"
        "moore/*.cc"
        "moore/*.cpp"
        "moore/*.mu"
    )

    file(GLOB_RECURSE MOORE_SOURCES CONFIGURE_DEPENDS ${MOORE_PATTERNS})

    set_source_files_properties(${MOORE_SOURCES} PROPERTIES LANGUAGE CXX)

    target_compile_definitions(infiniops PRIVATE WITH_MOORE=1)
    target_compile_options(infiniops PUBLIC "-x" "musa")
    target_sources(infiniops PRIVATE ${MOORE_SOURCES})

    target_include_directories(infiniops PUBLIC "${MUSA_ROOT}/include")
    target_link_libraries(infiniops PUBLIC ${MUSA_LIB} ${MUSART_LIB} ${MUBLAS_LIB})

    list(APPEND DEVICE_LIST "moore")
endif()

if(WITH_CAMBRICON)
    file(GLOB_RECURSE CAMBRICON_MLU_SOURCES CONFIGURE_DEPENDS "cambricon/*/*.mlu")
    find_program(CNCC_COMPILER cncc HINTS "${NEUWARE_HOME}/bin" "$ENV{NEUWARE_HOME}/bin" /usr/local/neuware/bin)
    if(CNCC_COMPILER)
        message(STATUS "Found cncc: ${CNCC_COMPILER}")
        set(MLU_COMPILE_OPTS
            -c --bang-mlu-arch=mtp_592 -O3 -fPIC -Wall -Werror -std=c++17 -pthread
            -I${CMAKE_CURRENT_SOURCE_DIR} -I${NEUWARE_HOME}/include
            -idirafter /usr/local/neuware/lib/clang/11.1.0/include
        )
        function(compile_mlu_file src_file)
            get_filename_component(name ${src_file} NAME_WE)
            get_filename_component(path ${src_file} DIRECTORY)
            set(out_file "${CMAKE_CURRENT_BINARY_DIR}/${path}/${name}.o")
            file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${path}")
            add_custom_command(OUTPUT ${out_file}
                COMMAND ${CNCC_COMPILER} ${MLU_COMPILE_OPTS} -c ${src_file} -o ${out_file}
                DEPENDS ${src_file}
                COMMENT "Building MLU kernel: ${src_file}"
            )
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
    target_link_libraries(infiniops PUBLIC ${CAMBRICON_RUNTIME_LIB} ${CAMBRICON_CNNL_LIB} ${CAMBRICON_CNNL_EXTRA_LIB} ${CAMBRICON_PAPI_LIB})

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(infiniops PUBLIC 
            "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-idirafter /usr/local/neuware/lib/clang/11.1.0/include>"
        )
    endif()

    list(APPEND DEVICE_LIST "cambricon")
endif()

if(WITH_ASCEND)
    # ASCEND_HOME is set by the top-level CMakeLists.txt.
    file(GLOB_RECURSE ASCEND_SOURCES CONFIGURE_DEPENDS
        "ascend/*.cc"
        "ascend/*.cpp"
    )
    # Exclude `kernel_impl.cpp` — AscendC device code, not compiled by the host C++ compiler.
    list(FILTER ASCEND_SOURCES EXCLUDE REGEX ".*kernel_impl\\.cpp$")

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

    list(APPEND DEVICE_LIST "ascend")
endif()

if(WITH_TORCH)
    file(GLOB_RECURSE TORCH_SOURCES CONFIGURE_DEPENDS "torch/*.cc" "torch/*.cpp")

    target_compile_definitions(infiniops PUBLIC WITH_TORCH=1)
    target_link_libraries(infiniops PUBLIC ${TORCH_LIBRARIES})
    target_include_directories(infiniops PUBLIC ${TORCH_INCLUDE_DIRS})

    if(WITH_METAX OR WITH_MOORE)
        # Vendor compilers (mxcc/mcc) cannot compile vendor-forked torch
        # headers. Compile torch sources with the system C++ compiler instead.
        find_program(SYSTEM_CXX NAMES g++ c++)

        if(NOT SYSTEM_CXX)
            message(FATAL_ERROR "Could not find system g++ for torch compilation.")
        endif()

        set(_torch_include_flags "")
        foreach(_dir ${TORCH_INCLUDE_DIRS})
            list(APPEND _torch_include_flags "-isystem" "${_dir}")
        endforeach()

        # Vendor-specific defines required by forked torch headers.
        set(_torch_extra_flags "")
        if(WITH_METAX)
            list(APPEND _torch_extra_flags "-DUSE_MACA=1")
        endif()
        if(DEFINED TORCH_CXX11_ABI)
            list(APPEND _torch_extra_flags "-D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}")
        endif()

        set(TORCH_OBJECT_DIR "${CMAKE_CURRENT_BINARY_DIR}/torch_objs")
        file(MAKE_DIRECTORY "${TORCH_OBJECT_DIR}")

        set(TORCH_OBJECT_FILES)
        foreach(_src ${TORCH_SOURCES})
            file(RELATIVE_PATH _rel ${CMAKE_CURRENT_SOURCE_DIR} ${_src})
            string(REPLACE "/" "_" _obj_name "${_rel}")
            string(REPLACE ".cc" ".o" _obj_name "${_obj_name}")
            string(REPLACE ".cpp" ".o" _obj_name "${_obj_name}")
            set(_obj "${TORCH_OBJECT_DIR}/${_obj_name}")

            add_custom_command(
                OUTPUT "${_obj}"
                COMMAND ${SYSTEM_CXX}
                    -std=c++17 -fPIC -O2
                    "-I${CMAKE_CURRENT_SOURCE_DIR}"
                    ${_torch_include_flags}
                    ${_torch_extra_flags}
                    -c "${_src}" -o "${_obj}"
                DEPENDS "${_src}"
                COMMENT "Compiling ${_rel} with system C++ compiler"
            )
            list(APPEND TORCH_OBJECT_FILES "${_obj}")
        endforeach()

        set_source_files_properties(${TORCH_OBJECT_FILES}
            PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
        target_sources(infiniops PRIVATE ${TORCH_OBJECT_FILES})
    else()
        target_sources(infiniops PRIVATE ${TORCH_SOURCES})
    endif()
endif()

target_include_directories(infiniops PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(GENERATE_PYTHON_BINDINGS)
    find_package(Python COMPONENTS Interpreter REQUIRED)
    # Always regenerate bindings so the included kernel headers match the
    # active device list.  Stale generated files (e.g., committed for one
    # platform) would omit specializations for other enabled backends,
    # causing link-time or runtime failures.

    set(GENERATOR_ARGS --devices ${DEVICE_LIST})
    if(WITH_TORCH)
        list(APPEND GENERATOR_ARGS --with-torch)
    endif()

    execute_process(
        COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/generate_wrappers.py ${GENERATOR_ARGS}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE script_result
    )

    if(NOT script_result EQUAL 0)
        message(FATAL_ERROR "Generating wrappers - failed")
    else()
        message(STATUS "Generating wrappers - done")
    endif()

    set(PYBIND11_SOURCES "${PROJECT_SOURCE_DIR}/generated/bindings/ops.cc")

    # TODO: There might be a better solution.
    if(WITH_NVIDIA OR WITH_ILUVATAR)
        set_source_files_properties(${PYBIND11_SOURCES} PROPERTIES LANGUAGE CUDA)
    endif()

    find_package(Python COMPONENTS Interpreter Development)

    if(NOT pybind11_DIR)
        execute_process(
            COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
            OUTPUT_VARIABLE _pybind11_cmake_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _pybind11_result
        )

        if(_pybind11_result EQUAL 0)
            set(pybind11_DIR "${_pybind11_cmake_dir}" CACHE PATH "pybind11 CMake directory")
        endif()
    endif()

    find_package(pybind11 CONFIG)

    if(PYBIND11_ENABLE_EXTRAS)
        pybind11_add_module(ops ${PYBIND11_SOURCES})
    else()
        pybind11_add_module(ops NO_EXTRAS ${PYBIND11_SOURCES})
    endif()

    target_include_directories(ops PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(ops PRIVATE infiniops)

    set_target_properties(infiniops PROPERTIES INSTALL_RPATH "$ORIGIN")
    set_target_properties(ops PROPERTIES INSTALL_RPATH "$ORIGIN")

    install(TARGETS infiniops ops DESTINATION .)

    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/__init__.py" "")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/__init__.py" DESTINATION .)
endif()
