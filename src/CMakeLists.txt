add_library(infiniops SHARED)

include(GNUInstallDirs)

file(GLOB BASE_SRCS CONFIGURE_DEPENDS "*.cc")
target_sources(infiniops PRIVATE ${BASE_SRCS})

set(DEVICE_LIST "")

if(WITH_CPU)
    set(CPU_PATTERNS
        "native/cpu/*.cc"
        "native/cpu/*.cpp"
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
        "native/cuda/*.cc"
        "native/cuda/*.cpp"
        "native/cuda/*.cu"
        "native/cuda/nvidia/*.cc"
        "native/cuda/nvidia/*.cpp"
        "native/cuda/nvidia/*.cu"
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

if(WITH_NINETOOTHED)
    find_package(Python COMPONENTS Interpreter REQUIRED)

    if(NINETOOTHED_PYTHON_EXECUTABLE)
        set(_ninetoothed_python "${NINETOOTHED_PYTHON_EXECUTABLE}")
    elseif(_TORCH_PYTHON)
        set(_ninetoothed_python "${_TORCH_PYTHON}")
    else()
        set(_ninetoothed_python "${Python_EXECUTABLE}")
    endif()
    message(STATUS "NineToothed codegen Python: ${_ninetoothed_python}")

    set(_ninetoothed_output_dir "${CMAKE_CURRENT_BINARY_DIR}/ninetoothed")
    set(_ninetoothed_generator_args
        "${PROJECT_SOURCE_DIR}/scripts/generate_ninetoothed_ops.py"
        --output-dir "${_ninetoothed_output_dir}")

    execute_process(
        COMMAND "${_ninetoothed_python}" ${_ninetoothed_generator_args}
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        RESULT_VARIABLE _ninetoothed_generation_result
    )

    if(NOT _ninetoothed_generation_result EQUAL 0)
        message(FATAL_ERROR "Generating NineToothed operator sources failed with `${_ninetoothed_python}`. Set `NINETOOTHED_PYTHON_EXECUTABLE` to a Python with `ninetoothed`, `ntops`, `triton`, `sympy`, and CUDA dependencies installed.")
    endif()

    include("${_ninetoothed_output_dir}/manifest.cmake")
    target_include_directories(infiniops PRIVATE
        ${INFINIOPS_NINETOOTHED_INCLUDE_DIRS})
    target_sources(infiniops PRIVATE ${INFINIOPS_NINETOOTHED_SOURCES})
endif()

if(WITH_ILUVATAR)
    set(ILUVATAR_PATTERNS
        "native/cuda/*.cc"
        "native/cuda/*.cpp"
        "native/cuda/*.cu"
        "native/cuda/iluvatar/*.cc"
        "native/cuda/iluvatar/*.cpp"
        "native/cuda/iluvatar/*.cu"
    )

    file(GLOB_RECURSE ILUVATAR_SOURCES CONFIGURE_DEPENDS ${ILUVATAR_PATTERNS})

    target_compile_definitions(infiniops PUBLIC WITH_ILUVATAR=1)
    target_sources(infiniops PRIVATE ${ILUVATAR_SOURCES})

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(infiniops PUBLIC CUDA::cudart CUDA::cublas CUDA::cuda_driver)

    list(APPEND DEVICE_LIST "iluvatar")
endif()

if(WITH_HYGON)
    set(HYGON_PATTERNS
        "native/cuda/*.cc"
        "native/cuda/*.cpp"
        "native/cuda/*.cu"
        "native/cuda/hygon/*.cc"
        "native/cuda/hygon/*.cpp"
        "native/cuda/hygon/*.cu"
    )

    file(GLOB_RECURSE HYGON_SOURCES CONFIGURE_DEPENDS ${HYGON_PATTERNS})

    enable_language(CUDA)

    target_compile_definitions(infiniops PUBLIC WITH_HYGON=1)
    target_sources(infiniops PRIVATE ${HYGON_SOURCES})

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(infiniops PUBLIC CUDA::cudart CUDA::cublas)

    set_target_properties(infiniops PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )

    list(APPEND DEVICE_LIST "hygon")
endif()

if(WITH_METAX)
    set(METAX_PATTERNS
        "native/cuda/*.cc"
        "native/cuda/*.cpp"
        "native/cuda/metax/*.cc"
        "native/cuda/metax/*.maca"
    )

    file(GLOB_RECURSE METAX_SOURCES CONFIGURE_DEPENDS ${METAX_PATTERNS})

    set_source_files_properties(${METAX_SOURCES} PROPERTIES LANGUAGE CXX)

    target_compile_definitions(infiniops PRIVATE WITH_METAX=1)
    target_compile_options(infiniops PRIVATE "-x" "maca")
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
        "native/cuda/*.cc"
        "native/cuda/*.cpp"
        "native/cuda/moore/*.cc"
        "native/cuda/moore/*.cpp"
        "native/cuda/moore/*.mu"
    )

    file(GLOB_RECURSE MOORE_SOURCES CONFIGURE_DEPENDS ${MOORE_PATTERNS})

    set_source_files_properties(${MOORE_SOURCES} PROPERTIES LANGUAGE CXX)

    target_compile_definitions(infiniops PRIVATE WITH_MOORE=1)
    target_compile_options(infiniops PRIVATE "-x" "musa")
    target_sources(infiniops PRIVATE ${MOORE_SOURCES})

    target_include_directories(infiniops PUBLIC "${MUSA_ROOT}/include")
    target_link_libraries(infiniops PUBLIC ${MUSA_LIB} ${MUSART_LIB} ${MUBLAS_LIB})

    list(APPEND DEVICE_LIST "moore")
endif()

if(WITH_CAMBRICON)
    file(GLOB_RECURSE CAMBRICON_MLU_SOURCES CONFIGURE_DEPENDS "native/cambricon/ops/*/*.mlu")
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
        "native/ascend/*.cc"
        "native/ascend/*.cpp"
    )
    # Exclude `kernel_impl.cpp` — `AscendC` device code, not compiled by the host C++ compiler.
    list(FILTER ASCEND_SOURCES EXCLUDE REGEX ".*kernel_impl\\.cpp$")
    # Exclude custom/ — standalone PyTorch extension, built separately.
    list(FILTER ASCEND_SOURCES EXCLUDE REGEX ".*/custom/.*")

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

    # ATB (Ascend Transformer Boost) — provides fused operators like
    # `PagedAttention` and `ReshapeAndCache` that are graph-capture safe.
    set(ATB_HOME_DIR "$ENV{ATB_HOME_PATH}")
    if(NOT ATB_HOME_DIR)
        # Default search path under CANN nnal directory.
        file(GLOB ATB_SEARCH_DIRS "/usr/local/Ascend/nnal/atb/*/atb/cxx_abi_1")
        if(ATB_SEARCH_DIRS)
            list(SORT ATB_SEARCH_DIRS ORDER DESCENDING)
            list(GET ATB_SEARCH_DIRS 0 ATB_HOME_DIR)
        endif()
    endif()

    if(ATB_HOME_DIR AND EXISTS "${ATB_HOME_DIR}/include/atb/operation.h")
        message(STATUS "ATB found: ${ATB_HOME_DIR}")
        target_compile_definitions(infiniops PUBLIC INFINI_HAS_ATB=1)
        target_include_directories(infiniops PUBLIC "${ATB_HOME_DIR}/include")
        target_link_libraries(infiniops PUBLIC "${ATB_HOME_DIR}/lib/libatb.so")
    else()
        message(STATUS "ATB not found — ATB-based operators disabled")
    endif()

    list(APPEND DEVICE_LIST "ascend")

    # Custom `AscendC` kernels (PyTorch extension, requires `torch_npu`).
    if(BUILD_ASCEND_CUSTOM)
        # In-tree `ascendc_library()` trips the `CANN` `extract_host_stub.py`
        # path-handling bug under `scikit-build-core`'s temp-dir builds
        # (`KeyError` on `/./workspace/...` paths in `$<TARGET_OBJECTS>`).
        # Work around it by driving the standalone `src/native/ascend/custom/build.sh`;
        # that script invokes a separate `cmake` with
        # `src/native/ascend/custom/` as its `SOURCE_DIR`, avoiding the buggy
        # path shape.  The produced `.a` is imported and linked into
        # `ops` with `--whole-archive`.
        set(_custom_build_dir "${CMAKE_SOURCE_DIR}/build/build_ascend_custom")
        set(_custom_lib "${_custom_build_dir}/lib/libno_workspace_kernel.a")
        set(_custom_source_dir "${CMAKE_CURRENT_BINARY_DIR}/ascend_custom_source")

        if(NOT DEFINED SOC_VERSION OR "${SOC_VERSION}" STREQUAL "")
            include(${CMAKE_CURRENT_SOURCE_DIR}/native/ascend/custom/cmake/detect_soc.cmake)
            infiniops_detect_soc(SOC_VERSION)
        endif()

        # Drive `build.sh` as a build-phase target with explicit source
        # dependencies so that editing any `op_host/` or `op_kernel/`
        # source re-triggers the build (plain `execute_process` at
        # configure time would only gate on file existence and leave
        # stale `.a` files in place).
        file(GLOB_RECURSE _custom_srcs CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/native/ascend/custom/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/native/ascend/custom/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/native/ascend/custom/build.sh")

        # Scrub env inherited from the outer `scikit-build-core` invocation
        # before handing control to `build.sh`: `CMAKE_GENERATOR` /
        # `CMAKE_EXPORT_COMPILE_COMMANDS` leaking into the inner `cmake`
        # change the path format passed to `ninja`'s `_host_cpp` rule and
        # re-trigger the `CANN` `extract_host_stub.py` `KeyError`
        # (`/./workspace/...`) that standalone `build.sh` avoids.
        #
        # `pip install` MUST be invoked with `--no-build-isolation` on
        # Ascend; otherwise pip's build-isolation overlay shadows system
        # `torch` (via `PYTHONPATH`) and the inner `cmake`'s
        # `import torch` in `config_envs.cmake` fails with
        # `ModuleNotFoundError`.
        add_custom_command(
            OUTPUT ${_custom_lib}
            COMMAND ${CMAKE_COMMAND} -E rm -f "${_custom_source_dir}"
            COMMAND ${CMAKE_COMMAND} -E create_symlink
                    "${CMAKE_CURRENT_SOURCE_DIR}/native/ascend/custom"
                    "${_custom_source_dir}"
            COMMAND ${CMAKE_COMMAND} -E env
                    --unset=CMAKE_GENERATOR
                    --unset=CMAKE_EXPORT_COMPILE_COMMANDS
                    --unset=CMAKE_BUILD_PARALLEL_LEVEL
                    "BUILD_DIR=${_custom_build_dir}"
                    "MAIN_SRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
                    bash ${_custom_source_dir}/build.sh ${SOC_VERSION}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/native/ascend/custom
            DEPENDS ${_custom_srcs}
            COMMENT "Building custom AscendC kernels (SOC_VERSION=${SOC_VERSION})"
            VERBATIM)

        add_custom_target(no_workspace_kernel_build ALL DEPENDS ${_custom_lib})

        add_library(no_workspace_kernel STATIC IMPORTED GLOBAL)
        set_target_properties(no_workspace_kernel PROPERTIES
            IMPORTED_LOCATION "${_custom_lib}")
        add_dependencies(no_workspace_kernel no_workspace_kernel_build)

        # Link the compiled `AscendC` kernel objects into `infiniops` so that
        # custom kernel implementations (e.g. `RmsNorm` index 1) can call
        # them via the generated launch functions.
        target_compile_definitions(infiniops PUBLIC INFINI_HAS_CUSTOM_KERNELS=1)
    endif()
endif()

if(WITH_TORCH)
    # Auto-generate ATen-backed operator wrappers from `scripts/torch_ops.yaml`.
    # The script writes into `${PROJECT_SOURCE_DIR}/generated/` (gitignored),
    # which we then glob below alongside any hand-written torch sources.
    find_package(Python COMPONENTS Interpreter REQUIRED)

    # Pin codegen to the locally installed torch version so vendor
    # forks (Cambricon's `torch_mlu` 2.1.0, etc.) get a schema whose
    # `at::<op>_out` overloads match the headers they ship.  Without
    # this, the codegen targets v2.4.0 and the build fails on older
    # forks with no-known-conversion errors (e.g. `at::all_out`'s
    # `int64_t dim` vs `OptionalIntArrayRef dim`).
    execute_process(
        COMMAND ${_TORCH_PYTHON} -c
            "import torch; print('v' + torch.__version__.split('+')[0])"
        OUTPUT_VARIABLE _torch_version_tag
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _torch_version_result
    )
    if(NOT _torch_version_result EQUAL 0 OR NOT _torch_version_tag)
        set(_torch_version_tag "v2.4.0")
    endif()
    message(STATUS "Codegen schema: PyTorch ${_torch_version_tag}")

    set(INFINIOPS_TORCH_OPS "" CACHE STRING
        "Semicolon-separated PyTorch op allowlist for generated torch wrappers")
    set(_torch_codegen_args
        ${PROJECT_SOURCE_DIR}/scripts/generate_torch_ops.py
        --pytorch-version ${_torch_version_tag})
    if(INFINIOPS_TORCH_OPS)
        string(REPLACE "," ";" _torch_op_allowlist "${INFINIOPS_TORCH_OPS}")
        list(APPEND _torch_codegen_args --ops ${_torch_op_allowlist})
        message(STATUS "Codegen torch op allowlist: ${_torch_op_allowlist}")
    endif()

    execute_process(
        COMMAND ${Python_EXECUTABLE} ${_torch_codegen_args}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE _torch_ops_result
    )
    if(NOT _torch_ops_result EQUAL 0)
        message(FATAL_ERROR "Generating torch op wrappers - failed")
    endif()
    message(STATUS "Generating torch op wrappers - done")

    file(GLOB_RECURSE TORCH_SOURCES CONFIGURE_DEPENDS
        "torch/*.cc" "torch/*.cpp"
        "${PROJECT_SOURCE_DIR}/generated/torch/*.cc"
        "${PROJECT_SOURCE_DIR}/generated/torch/*.cpp"
    )

    set(INFINIOPS_TORCH_UNITY_BATCH_SIZE "8" CACHE STRING
        "Number of torch sources to include in each generated unity translation unit; set to 1 to disable")
    set(TORCH_COMPILE_SOURCES ${TORCH_SOURCES})
    if(INFINIOPS_TORCH_UNITY_BATCH_SIZE GREATER 1)
        set(_torch_unity_dir "${CMAKE_CURRENT_BINARY_DIR}/torch_unity")
        file(REMOVE_RECURSE "${_torch_unity_dir}")
        file(MAKE_DIRECTORY "${_torch_unity_dir}")

        set(TORCH_COMPILE_SOURCES)
        set(_torch_unity_index 0)
        set(_torch_unity_count 0)
        foreach(_src IN LISTS TORCH_SOURCES)
            if(_torch_unity_count EQUAL 0)
                set(_torch_unity_src
                    "${_torch_unity_dir}/torch_unity_${_torch_unity_index}.cc")
                file(WRITE "${_torch_unity_src}"
                    "// Generated by CMake to batch ATen-heavy torch wrappers.\n")
                list(APPEND TORCH_COMPILE_SOURCES "${_torch_unity_src}")
            endif()

            file(APPEND "${_torch_unity_src}" "#include \"${_src}\"\n")

            math(EXPR _torch_unity_count "${_torch_unity_count} + 1")
            if(_torch_unity_count GREATER_EQUAL INFINIOPS_TORCH_UNITY_BATCH_SIZE)
                math(EXPR _torch_unity_index "${_torch_unity_index} + 1")
                set(_torch_unity_count 0)
            endif()
        endforeach()

        list(LENGTH TORCH_SOURCES _torch_source_count)
        list(LENGTH TORCH_COMPILE_SOURCES _torch_unity_source_count)
        message(STATUS
            "Torch unity build: ${_torch_source_count} sources batched into "
            "${_torch_unity_source_count} translation units")
    endif()

    target_compile_definitions(infiniops PUBLIC WITH_TORCH=1)
    target_link_libraries(infiniops PUBLIC ${TORCH_LIBRARIES})
    target_include_directories(infiniops PUBLIC
        ${TORCH_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/generated
    )

    # Each generated `<op>.cc` instantiates `at::<op>_out(...)`, which
    # pulls in roughly 0.5-1 GB of ATen template metaprogramming.  At
    # ninja's default parallelism (one job per CPU), a build with 451
    # ops can blow past 30 GB of RSS and the OOM killer drops
    # `cc1plus`.  Cap the heavyweight torch sources via a Ninja job pool;
    # the rest of the build keeps full parallelism.  CI can raise this when
    # the container has enough memory.
    set(INFINIOPS_TORCH_COMPILE_JOBS "2" CACHE STRING
        "Maximum concurrent generated torch source compilations")
    if(CMAKE_GENERATOR MATCHES "Ninja")
        set_property(GLOBAL APPEND PROPERTY JOB_POOLS
            torch_compile=${INFINIOPS_TORCH_COMPILE_JOBS})
    endif()

    if(WITH_METAX OR WITH_MOORE)
        # Vendor compilers (`mxcc`/`mcc`) cannot compile vendor-forked `torch`
        # headers. Compile `torch` sources with the system C++ compiler instead.
        find_program(SYSTEM_CXX NAMES g++ c++)

        if(NOT SYSTEM_CXX)
            message(FATAL_ERROR "Could not find system `g++` for `torch` compilation.")
        endif()

        set(_torch_include_flags "")
        foreach(_dir ${TORCH_INCLUDE_DIRS})
            list(APPEND _torch_include_flags "-isystem" "${_dir}")
        endforeach()

        # Vendor-specific defines required by forked `torch` headers.
        set(_torch_extra_flags "")
        if(WITH_METAX)
            list(APPEND _torch_extra_flags "-DUSE_MACA=1" "-DWITH_METAX=1")
        endif()
        if(WITH_MOORE)
            list(APPEND _torch_extra_flags "-DWITH_MOORE=1")
        endif()
        if(WITH_CPU)
            list(APPEND _torch_extra_flags "-DWITH_CPU=1")
        endif()
        if(DEFINED TORCH_CXX11_ABI)
            list(APPEND _torch_extra_flags "-D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}")
        endif()

        set(TORCH_OBJECT_DIR "${CMAKE_CURRENT_BINARY_DIR}/torch_objs")
        file(MAKE_DIRECTORY "${TORCH_OBJECT_DIR}")

        set(TORCH_OBJECT_FILES)
        foreach(_src ${TORCH_COMPILE_SOURCES})
            file(RELATIVE_PATH _rel ${CMAKE_CURRENT_SOURCE_DIR} ${_src})
            string(REPLACE "/" "_" _obj_name "${_rel}")
            string(REPLACE ".cc" ".o" _obj_name "${_obj_name}")
            string(REPLACE ".cpp" ".o" _obj_name "${_obj_name}")
            set(_obj "${TORCH_OBJECT_DIR}/${_obj_name}")

            add_custom_command(
                OUTPUT "${_obj}"
                COMMAND ${SYSTEM_CXX}
                    -std=c++17 -fPIC -O0
                    "-I${CMAKE_CURRENT_SOURCE_DIR}"
                    "-I${PROJECT_SOURCE_DIR}/generated"
                    ${_torch_include_flags}
                    ${_torch_extra_flags}
                    -c "${_src}" -o "${_obj}"
                DEPENDS "${_src}"
                COMMENT "Compiling ${_rel} with system C++ compiler"
                JOB_POOL torch_compile
            )
            list(APPEND TORCH_OBJECT_FILES "${_obj}")
        endforeach()

        set_source_files_properties(${TORCH_OBJECT_FILES}
            PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
        target_sources(infiniops PRIVATE ${TORCH_OBJECT_FILES})
    else()
        # Build the heavy torch sources as their own object library so
        # the Ninja `torch_compile` job pool throttles only those
        # compilations and the rest of `infiniops` keeps full
        # parallelism.  Inherit infiniops's compile-time settings via
        # generator expressions (linking would create a cyclic
        # dependency since infiniops then absorbs the object files).
        add_library(infiniops_torch_objs OBJECT ${TORCH_COMPILE_SOURCES})
        set_target_properties(infiniops_torch_objs
            PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_include_directories(infiniops_torch_objs PRIVATE
            $<TARGET_PROPERTY:infiniops,INCLUDE_DIRECTORIES>
            ${TORCH_INCLUDE_DIRS}
            ${PROJECT_SOURCE_DIR}/generated)
        target_compile_definitions(infiniops_torch_objs PRIVATE
            $<TARGET_PROPERTY:infiniops,COMPILE_DEFINITIONS>)
        target_compile_options(infiniops_torch_objs PRIVATE
            $<TARGET_PROPERTY:infiniops,COMPILE_OPTIONS>
            -O0)
        if(CMAKE_GENERATOR MATCHES "Ninja")
            set_target_properties(infiniops_torch_objs
                PROPERTIES JOB_POOL_COMPILE torch_compile)
        endif()
        target_sources(infiniops PRIVATE
            $<TARGET_OBJECTS:infiniops_torch_objs>)
    endif()
endif()

target_include_directories(infiniops
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/generated/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(GENERATE_OPERATOR_CALL_INSTANTIATIONS OR GENERATE_PYTHON_BINDINGS)
    find_package(Python COMPONENTS Interpreter REQUIRED)
    # Always regenerate wrappers so emitted call instantiations and bindings
    # match the active device list.  Stale generated files would omit
    # specializations for enabled backends, causing link-time or runtime
    # failures.

    set(GENERATOR_ARGS --devices ${DEVICE_LIST})
    if(WITH_TORCH)
        list(APPEND GENERATOR_ARGS --with-torch)
    endif()
    if(WITH_NINETOOTHED)
        list(APPEND GENERATOR_ARGS --with-ninetoothed)
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
endif()

if(GENERATE_OPERATOR_CALL_INSTANTIATIONS)
    file(GLOB_RECURSE OPERATOR_CALL_INSTANTIATION_SOURCES CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/generated/src/operator_call_instantiations_*.cc")

    if(WITH_NVIDIA OR WITH_HYGON)
        set_source_files_properties(${OPERATOR_CALL_INSTANTIATION_SOURCES}
            PROPERTIES LANGUAGE CUDA)
        target_sources(infiniops PRIVATE ${OPERATOR_CALL_INSTANTIATION_SOURCES})
    elseif(WITH_ILUVATAR)
        set(_iluvatar_call_instantiation_include_flags
            "-I${CMAKE_CURRENT_SOURCE_DIR}"
            "-I${PROJECT_SOURCE_DIR}"
            "-I${PROJECT_SOURCE_DIR}/generated"
            "-I${PROJECT_SOURCE_DIR}/generated/include")
        foreach(_dir IN LISTS TORCH_INCLUDE_DIRS CUDAToolkit_INCLUDE_DIRS)
            list(APPEND _iluvatar_call_instantiation_include_flags "-I${_dir}")
        endforeach()

        set(_iluvatar_call_instantiation_defs -DWITH_ILUVATAR=1)
        if(WITH_CPU)
            list(APPEND _iluvatar_call_instantiation_defs -DWITH_CPU=1)
        endif()
        if(WITH_TORCH)
            list(APPEND _iluvatar_call_instantiation_defs -DWITH_TORCH=1)
        endif()
        if(DEFINED TORCH_CXX11_ABI)
            list(APPEND _iluvatar_call_instantiation_defs
                "-D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}")
        endif()

        set(ILUVATAR_CALL_INSTANTIATION_OBJECTS)
        set(_iluvatar_call_instantiation_object_dir
            "${CMAKE_CURRENT_BINARY_DIR}/iluvatar_call_instantiation_objs")
        foreach(_src IN LISTS OPERATOR_CALL_INSTANTIATION_SOURCES)
            get_filename_component(_name "${_src}" NAME_WE)
            set(_obj "${_iluvatar_call_instantiation_object_dir}/${_name}.o")
            set(_dep "${_obj}.d")
            set(_depfile_arg)
            if(CMAKE_GENERATOR MATCHES "Ninja")
                set(_depfile_arg DEPFILE "${_dep}")
            endif()
            add_custom_command(
                OUTPUT "${_obj}"
                COMMAND ${CMAKE_COMMAND} -E make_directory
                    "${_iluvatar_call_instantiation_object_dir}"
                COMMAND ${ILUVATAR_CUDA_COMPILER}
                    ${_iluvatar_call_instantiation_defs}
                    ${_iluvatar_call_instantiation_include_flags}
                    ${ILUVATAR_CUDA_FLAGS}
                    -MMD -MF "${_dep}"
                    -c "${_src}" -o "${_obj}"
                DEPENDS "${_src}"
                ${_depfile_arg}
                COMMENT "Compiling ${_name}.cc with CoreX clang++"
                VERBATIM
            )
            list(APPEND ILUVATAR_CALL_INSTANTIATION_OBJECTS "${_obj}")
        endforeach()

        set_source_files_properties(${ILUVATAR_CALL_INSTANTIATION_OBJECTS}
            PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
        target_sources(infiniops PRIVATE ${ILUVATAR_CALL_INSTANTIATION_OBJECTS})
    else()
        target_sources(infiniops PRIVATE ${OPERATOR_CALL_INSTANTIATION_SOURCES})
    endif()
endif()

if(GENERATE_PYTHON_BINDINGS)
    file(GLOB_RECURSE PYBIND11_SOURCES CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/generated/bindings/*.cc")

    set(PYBIND11_DISPATCH_SOURCES)
    set(PYBIND11_UNITY_SOURCES)
    foreach(_src IN LISTS PYBIND11_SOURCES)
        if(_src MATCHES "/generated_dispatch[^/]*\\.cc$")
            list(APPEND PYBIND11_DISPATCH_SOURCES "${_src}")
        else()
            list(APPEND PYBIND11_UNITY_SOURCES "${_src}")
        endif()
    endforeach()

    set(PYBIND11_COMPILE_SOURCES ${PYBIND11_UNITY_SOURCES})
    if(WITH_TORCH)
        set(INFINIOPS_BINDING_UNITY_BATCH_SIZE "8" CACHE STRING
            "Number of generated pybind11 sources to include in each unity translation unit; set to 1 to disable")
        if(INFINIOPS_BINDING_UNITY_BATCH_SIZE GREATER 1)
            set(_binding_unity_dir "${CMAKE_CURRENT_BINARY_DIR}/binding_unity")
            file(REMOVE_RECURSE "${_binding_unity_dir}")
            file(MAKE_DIRECTORY "${_binding_unity_dir}")

            set(PYBIND11_COMPILE_SOURCES)
            set(_binding_unity_index 0)
            set(_binding_unity_count 0)
            foreach(_src IN LISTS PYBIND11_UNITY_SOURCES)
                if(_binding_unity_count EQUAL 0)
                    set(_binding_unity_src
                        "${_binding_unity_dir}/binding_unity_${_binding_unity_index}.cc")
                    file(WRITE "${_binding_unity_src}"
                        "// Generated by CMake to batch pybind11 wrapper sources.\n")
                    list(APPEND PYBIND11_COMPILE_SOURCES "${_binding_unity_src}")
                endif()

                file(APPEND "${_binding_unity_src}" "#include \"${_src}\"\n")

                math(EXPR _binding_unity_count "${_binding_unity_count} + 1")
                if(_binding_unity_count GREATER_EQUAL INFINIOPS_BINDING_UNITY_BATCH_SIZE)
                    math(EXPR _binding_unity_index "${_binding_unity_index} + 1")
                    set(_binding_unity_count 0)
                endif()
            endforeach()

            list(LENGTH PYBIND11_UNITY_SOURCES _binding_source_count)
            list(LENGTH PYBIND11_COMPILE_SOURCES _binding_unity_source_count)
            message(STATUS
                "Binding unity build: ${_binding_source_count} sources batched into "
                "${_binding_unity_source_count} translation units")
        endif()
    endif()
    list(APPEND PYBIND11_COMPILE_SOURCES ${PYBIND11_DISPATCH_SOURCES})

    if(WITH_NVIDIA OR WITH_HYGON)
        set_source_files_properties(${PYBIND11_COMPILE_SOURCES} PROPERTIES LANGUAGE CUDA)
    elseif(WITH_ILUVATAR)
        set(_iluvatar_dispatch_include_flags
            "-I${CMAKE_CURRENT_SOURCE_DIR}"
            "-I${PROJECT_SOURCE_DIR}"
            "-I${PROJECT_SOURCE_DIR}/generated")
        foreach(_dir IN LISTS TORCH_INCLUDE_DIRS CUDAToolkit_INCLUDE_DIRS)
            list(APPEND _iluvatar_dispatch_include_flags "-I${_dir}")
        endforeach()

        set(_iluvatar_dispatch_defs -DWITH_ILUVATAR=1)
        if(WITH_TORCH)
            list(APPEND _iluvatar_dispatch_defs -DWITH_TORCH=1)
        endif()
        if(DEFINED TORCH_CXX11_ABI)
            list(APPEND _iluvatar_dispatch_defs "-D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}")
        endif()

        set(ILUVATAR_DISPATCH_OBJECTS)
        set(_iluvatar_dispatch_object_dir "${CMAKE_CURRENT_BINARY_DIR}/iluvatar_dispatch_objs")
        foreach(_src IN LISTS PYBIND11_DISPATCH_SOURCES)
            get_filename_component(_name "${_src}" NAME_WE)
            set(_obj "${_iluvatar_dispatch_object_dir}/${_name}.o")
            set(_dep "${_obj}.d")
            set(_depfile_arg)
            if(CMAKE_GENERATOR MATCHES "Ninja")
                set(_depfile_arg DEPFILE "${_dep}")
            endif()
            add_custom_command(
                OUTPUT "${_obj}"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${_iluvatar_dispatch_object_dir}"
                COMMAND ${ILUVATAR_CUDA_COMPILER}
                    ${_iluvatar_dispatch_defs}
                    ${_iluvatar_dispatch_include_flags}
                    ${ILUVATAR_CUDA_FLAGS}
                    -MMD -MF "${_dep}"
                    -c "${_src}" -o "${_obj}"
                DEPENDS "${_src}"
                ${_depfile_arg}
                COMMENT "Compiling ${_name}.cc with CoreX clang++"
                VERBATIM
            )
            list(APPEND ILUVATAR_DISPATCH_OBJECTS "${_obj}")
        endforeach()

        list(REMOVE_ITEM PYBIND11_COMPILE_SOURCES ${PYBIND11_DISPATCH_SOURCES})
        set_source_files_properties(${ILUVATAR_DISPATCH_OBJECTS}
            PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
        list(APPEND PYBIND11_COMPILE_SOURCES ${ILUVATAR_DISPATCH_OBJECTS})
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
        pybind11_add_module(ops ${PYBIND11_COMPILE_SOURCES})
    else()
        pybind11_add_module(ops NO_EXTRAS ${PYBIND11_COMPILE_SOURCES})
    endif()

    if(WITH_TORCH AND CMAKE_GENERATOR MATCHES "Ninja")
        set(INFINIOPS_BINDING_COMPILE_JOBS "2" CACHE STRING
            "Maximum concurrent generated pybind11 binding compilations")
        set_property(GLOBAL APPEND PROPERTY JOB_POOLS
            binding_compile=${INFINIOPS_BINDING_COMPILE_JOBS})
        set_property(TARGET ops PROPERTY JOB_POOL_COMPILE binding_compile)
    endif()

    if(WITH_METAX)
        target_compile_options(ops PRIVATE "-x" "maca")
    endif()
    if(WITH_MOORE)
        target_compile_options(ops PRIVATE "-x" "musa")
    endif()

    if(WITH_HYGON)
        set_target_properties(ops PROPERTIES
            RULE_LAUNCH_COMPILE "${_HYGON_RULE_LAUNCH_ENV}"
            RULE_LAUNCH_LINK "${_HYGON_RULE_LAUNCH_ENV}"
        )
    endif()

    target_include_directories(ops PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/generated/include
    )
    if(WITH_NINETOOTHED)
        target_include_directories(ops PRIVATE
            ${INFINIOPS_NINETOOTHED_INCLUDE_DIRS})
    endif()
    target_link_libraries(ops PRIVATE infiniops)

    # Cambricon generated dispatch is compiled into the Python extension and
    # directly references host launch stubs emitted from `.mlu` sources. Link
    # those objects into `ops` too so import does not depend on re-exporting
    # internal C++ symbols from `libinfiniops`.
    if(WITH_CAMBRICON AND CAMBRICON_OBJECT_FILES)
        target_sources(ops PRIVATE ${CAMBRICON_OBJECT_FILES})
    endif()

    # Custom `AscendC` kernel objects must be linked directly into ops
    # because the `AscendC` toolchain compiles host stubs with hidden
    # visibility — `libinfiniops.so` cannot re-export those symbols.
    # The `Operator<..., 1>` template instantiations that call
    # `aclrtlaunch_*` live in `ops.cc`, so link here with
    # `--whole-archive` to ensure all launch functions are available.
    # `$<TARGET_FILE>` works for both real `ascendc_library()` targets and
    # `IMPORTED` targets pointing at a pre-built `.a`.  The
    # `no_workspace_kernel` target is only created inside the
    # `WITH_ASCEND` block above, so this branch must mirror that gate;
    # otherwise non-Ascend builds error out with "No target
    # no_workspace_kernel".
    if(WITH_ASCEND AND BUILD_ASCEND_CUSTOM)
        target_link_libraries(ops PRIVATE
            -Wl,--whole-archive $<TARGET_FILE:no_workspace_kernel> -Wl,--no-whole-archive)
        # `ops` link step must wait for `build.sh` to produce the `.a`.
        add_dependencies(ops no_workspace_kernel_build)
    endif()
    set(_INFINIOPS_INSTALL_RPATH "$ORIGIN")
    if(WITH_TORCH)
        list(APPEND _INFINIOPS_INSTALL_RPATH ${TORCH_RUNTIME_DIRS})
    endif()

    set_target_properties(infiniops PROPERTIES INSTALL_RPATH "${_INFINIOPS_INSTALL_RPATH}")
    set_target_properties(ops PROPERTIES INSTALL_RPATH "${_INFINIOPS_INSTALL_RPATH}")

    install(TARGETS infiniops ops DESTINATION .)

    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/__init__.py" "")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/__init__.py" DESTINATION .)

    if(WITH_TORCH)
        # Ship the per-op metadata alongside the bindings so the unified
        # torch op test can discover what to exercise at runtime.
        install(FILES "${PROJECT_SOURCE_DIR}/generated/torch_ops_metadata.json"
            DESTINATION .)
    endif()
endif()

install(TARGETS infiniops
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(GENERATE_OPERATOR_CALL_INSTANTIATIONS)
    install(FILES
        ${PROJECT_SOURCE_DIR}/generated/include/infini/operator_call_instantiations.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/infini
    )
endif()

file(GLOB INFINIOPS_PUBLIC_CORE_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

install(FILES ${INFINIOPS_PUBLIC_CORE_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/base/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/base
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/common/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/common
    FILES_MATCHING PATTERN "*.h"
)

if(EXISTS ${PROJECT_SOURCE_DIR}/generated/base)
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/generated/base/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/base
        FILES_MATCHING PATTERN "*.h"
    )
endif()
