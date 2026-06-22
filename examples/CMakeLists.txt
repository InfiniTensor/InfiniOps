option(BUILD_DISTRIBUTED_MATMUL_DEMO
    "Build the InfiniOps + InfiniCCL distributed matmul demo" OFF)

file(GLOB_RECURSE EXAMPLE_SOURCES CONFIGURE_DEPENDS "*.cc")

if(NOT BUILD_DISTRIBUTED_MATMUL_DEMO)
    list(FILTER EXAMPLE_SOURCES EXCLUDE REGEX ".*/distributed_matmul/.*\\.cc$")
endif()

# Iterate through each file and create an executable.
foreach(source_file ${EXAMPLE_SOURCES})
    get_filename_component(example_name ${source_file} NAME_WE)

    add_executable(${example_name} ${source_file})

    target_link_libraries(${example_name} PRIVATE infiniops)

    target_include_directories(${example_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    get_filename_component(example_dir ${source_file} DIRECTORY)

    target_include_directories(${example_name} PRIVATE ${example_dir})

    if(WITH_TORCH)
        foreach(_torch_dir ${TORCH_RUNTIME_DIRS})
            target_link_options(${example_name} PRIVATE "LINKER:-rpath-link,${_torch_dir}")
        endforeach()
    endif()
endforeach()

if(BUILD_DISTRIBUTED_MATMUL_DEMO AND TARGET distributed_matmul)
    set(INFINICCL_ROOT "" CACHE PATH "InfiniCCL source or install root")
    set(INFINICCL_INSTALL "" CACHE PATH "InfiniCCL install prefix")

    set(_infiniccl_prefix_candidates
        "${INFINICCL_INSTALL}"
        "${INFINICCL_ROOT}/install"
        "${INFINICCL_ROOT}"
        "$ENV{INFINICCL_INSTALL}"
        "$ENV{INFINICCL_ROOT}/install"
        "$ENV{INFINICCL_ROOT}"
        "$ENV{HOME}/.infini"
        "/usr/local")

    find_path(INFINICCL_INCLUDE_DIR
        NAMES infiniccl.h
        HINTS ${_infiniccl_prefix_candidates}
        PATH_SUFFIXES include include/infiniccl)
    find_library(INFINICCL_LIBRARY
        NAMES infiniccl
        HINTS ${_infiniccl_prefix_candidates}
        PATH_SUFFIXES lib lib64)

    if(NOT INFINICCL_INCLUDE_DIR OR NOT INFINICCL_LIBRARY)
        message(FATAL_ERROR
            "BUILD_DISTRIBUTED_MATMUL_DEMO=ON requires InfiniCCL. "
            "Set INFINICCL_INSTALL, INFINICCL_ROOT, or install it to ~/.infini.")
    endif()

    target_include_directories(distributed_matmul PRIVATE
        "${INFINICCL_INCLUDE_DIR}")
    target_link_libraries(distributed_matmul PRIVATE
        "${INFINICCL_LIBRARY}")
    target_link_options(distributed_matmul PRIVATE
        "LINKER:-rpath,${CMAKE_CURRENT_BINARY_DIR}/../src")
endif()
