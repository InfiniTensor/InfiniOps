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

    find_package(OpenMP REQUIRED)
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
    target_link_libraries(infiniops PUBLIC CUDA::cudart CUDA::cublas CUDA::cuda_driver)

    list(APPEND DEVICE_LIST "nvidia")
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

target_include_directories(infiniops PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(GENERATE_PYTHON_BINDINGS)
    execute_process(
        COMMAND python ${PROJECT_SOURCE_DIR}/scripts/generate_wrappers.py --devices ${DEVICE_LIST}
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
    if(WITH_NVIDIA)
        set_source_files_properties(${PYBIND11_SOURCES} PROPERTIES LANGUAGE CUDA)
    endif()

    find_package(Python COMPONENTS Interpreter Development)
    find_package(pybind11 CONFIG)

    pybind11_add_module(ops ${PYBIND11_SOURCES})

    target_include_directories(ops PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(ops PRIVATE infiniops)

    set_target_properties(infiniops PROPERTIES INSTALL_RPATH "$ORIGIN")
    set_target_properties(ops PROPERTIES INSTALL_RPATH "$ORIGIN")

    install(TARGETS infiniops ops DESTINATION .)

    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/__init__.py" "")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/__init__.py" DESTINATION .)
endif()
