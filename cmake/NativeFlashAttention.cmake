# Build the pinned inference kernels separately. No Torch include/link flags or
# Python runtime enter this target; Python is used only to prepare pinned source.
function(infiniops_add_native_flash target)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    find_package(CUDAToolkit REQUIRED)
    find_program(_flash_patch NAMES patch REQUIRED)
    set(_provider "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/native/cuda/nvidia/ops/flash_attn_with_kvcache")
    set(_varlen_provider "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/native/cuda/nvidia/ops/flash_attn_varlen_func")
    set(_source "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/..")
    set(_upstream "${_source}/third_party/flash-attention")
    set(_kernel_source "${_upstream}/csrc/flash_attn/src")
    set(_cutlass_include "${_upstream}/csrc/cutlass/include")
    set(_prepared "${CMAKE_CURRENT_BINARY_DIR}/native-flash-inference")
    set(_inputs "${_source}/scripts/prepare_flash_attention_inference.py"
        "${_source}/patches/flash-attention/inference.patch"
        "${_upstream}/csrc/flash_attn/flash_api.cpp"
        "${_upstream}/LICENSE" "${_upstream}/AUTHORS"
        "${_upstream}/csrc/cutlass/LICENSE.txt"
        "${_cutlass_include}/cutlass/cutlass.h")
    # Fixed forward closure: no recursive source scanning or copied CUTLASS tree.
    foreach(_header block_info.h dropout.h flash.h flash_fwd_kernel.h
            flash_fwd_launch_template.h hardware_info.h kernel_traits.h mask.h
            namespace_config.h philox.cuh rotary.h softmax.h static_switch.h utils.h)
        list(APPEND _inputs "${_kernel_source}/${_header}")
    endforeach()
    foreach(_input IN LISTS _inputs)
        if(NOT EXISTS "${_input}")
            message(FATAL_ERROR
                "Native FlashAttention is missing ${_input}. Initialize third_party/flash-attention and its CUTLASS submodule.")
        endif()
    endforeach()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_inputs})
    execute_process(COMMAND "${Python3_EXECUTABLE}"
        "${_source}/scripts/prepare_flash_attention_inference.py"
        --source-dir "${_upstream}" --output "${_prepared}"
        --patch-executable "${_flash_patch}"
        RESULT_VARIABLE _status ERROR_VARIABLE _prepare_error)
    if(NOT _status EQUAL 0)
        message(FATAL_ERROR "Pinned FlashAttention preparation failed: ${_prepare_error}")
    endif()
    set(_launches)
    foreach(_flash_kind fp16 bf16)
        if(_flash_kind STREQUAL fp16)
            set(_flash_type "cutlass::half_t")
        else()
            set(_flash_type "cutlass::bfloat16_t")
        endif()
        foreach(_flash_causal false true)
            foreach(_flash_hdim 32 64 96 128 192 256)
                set(_launch "${CMAKE_CURRENT_BINARY_DIR}/flash_${_flash_hdim}_${_flash_kind}_${_flash_causal}.cu")
                configure_file("${_provider}/launch.cu.in" "${_launch}" @ONLY)
                list(APPEND _launches "${_launch}")
                set(_varlen_launch "${CMAKE_CURRENT_BINARY_DIR}/flash_varlen_${_flash_hdim}_${_flash_kind}_${_flash_causal}.cu")
                configure_file("${_varlen_provider}/launch.cu.in" "${_varlen_launch}" @ONLY)
                list(APPEND _launches "${_varlen_launch}")
            endforeach()
        endforeach()
    endforeach()
    add_library(${target} OBJECT "${_provider}/kernel.cc"
        "${_varlen_provider}/kernel.cc" ${_launches})
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17)
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_definitions(${target} PRIVATE
        FLASH_NAMESPACE=infini_flash FLASHATTENTION_DISABLE_DROPOUT)
    target_include_directories(${target} PRIVATE
        "${_source}/src" "${_source}/include" "${_prepared}"
        "${_cutlass_include}" "${_provider}" "${_varlen_provider}"
        ${INFINI_RT_INCLUDE_DIRS})
    target_compile_options(${target} PRIVATE
        # Match the pinned upstream setup.py numerical compilation mode.
        "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda;--use_fast_math>")
    target_link_libraries(${target} PRIVATE CUDA::cudart)
    install(FILES "${_prepared}/LICENSE"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/InfiniOps"
        RENAME FlashAttention-LICENSE.txt)
    install(FILES "${_prepared}/AUTHORS"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/InfiniOps"
        RENAME FlashAttention-AUTHORS.txt)
    install(FILES "${_prepared}/CUTLASS-LICENSE.txt"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/InfiniOps"
        RENAME FlashAttention-CUTLASS-LICENSE.txt)
endfunction()
