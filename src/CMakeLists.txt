add_library(infiniops SHARED)

file(GLOB BASE_SRCS CONFIGURE_DEPENDS "*.cc")
target_sources(infiniops PRIVATE ${BASE_SRCS})

if(USE_CPU)
    set(CPU_PATTERNS
        "cpu/*.cc"
        "cpu/*.cpp"
    )

    file(GLOB_RECURSE CPU_SOURCES CONFIGURE_DEPENDS ${CPU_PATTERNS})
    list(APPEND CORE_SOURCES ${CPU_SOURCES})

    target_compile_definitions(infiniops PRIVATE USE_CPU=1)
    
    # Reserve for OpenMP
    # find_package(OpenMP REQUIRED)
    # target_link_libraries(infiniops PRIVATE OpenMP::OpenMP_CXX)
endif()

if(USE_NVIDIA)
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
    
    target_compile_definitions(infiniops PRIVATE USE_NVIDIA=1)
    target_sources(infiniops PRIVATE ${NVIDIA_SOURCES})
    
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(infiniops PRIVATE CUDA::cublas)
endif()

if(USE_METAX)
    set(METAX_PATTERNS
        "cuda/*.cc"
        "cuda/*.cpp"
        "metax/*.cc"
        "metax/*.maca"
    )

    file(GLOB_RECURSE METAX_SOURCES CONFIGURE_DEPENDS ${METAX_PATTERNS})

    set_source_files_properties(${METAX_SOURCES} PROPERTIES LANGUAGE CXX)

    target_compile_definitions(infiniops PRIVATE USE_METAX=1)
    target_sources(infiniops PRIVATE ${METAX_SOURCES})

    target_include_directories(infiniops PUBLIC "${MACA_PATH}/include")
    target_link_libraries(infiniops PUBLIC 
        ${MACA_RUNTIME_LIB} 
        ${MACA_DNN_LIB} 
        ${MACA_BLAS_LIB}
    )
endif()

target_include_directories(infiniops PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
