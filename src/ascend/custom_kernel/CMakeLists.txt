cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(ascend-kernel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
endif()

add_compile_options(-Wunused-value -Wcast-align -Wcast-qual -Wwrite-strings
                    -Wsign-compare -Wextra)

if(${CMAKE_BUILD_TYPE} MATCHES "RELEASE")
    add_compile_options(-O3 -fvisibility=hidden -fvisibility-inlines-hidden
                        -fstack-protector-strong -fPIE -fPIC)
    message(STATUS "build type set to RELEASE")
else()
    add_compile_options(-g -rdynamic)
endif()

set(PROJECT_OP_SRC_BASE ${PROJECT_SOURCE_DIR}/csrc)
set(PROJECT_BUILD_PATH ${PROJECT_SOURCE_DIR}/build)
set(PROJECT_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/output)

include(cmake/config_envs.cmake)
include(cmake/config_ascend.cmake)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

add_subdirectory(csrc)
