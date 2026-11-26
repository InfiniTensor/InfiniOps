# Build the pinned Hygon FlashMLA kernels with their HIP compiler mode.
function(infiniops_add_hygon_flashmla target)
    set(_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/..")
    set(_source "${_root}/third_party/flashmla_hygon/csrc")
    set(_bridge "${_root}/src/native/cuda/hygon/flash_mla")
    find_program(_flashmla_aicc NAMES aicc HINTS "${DTK_ROOT}/bin" REQUIRED)
    set(_hip_sources
        "${_source}/gfx9/decode/combine/combine.hip"
        "${_source}/gfx9/decode/get_decoding_sched_meta/get_decoding_sched_meta.hip"
        "${_source}/gfx93/decode/sparse_fp8/instantiations/model1_persistent_h64.hip"
        "${_bridge}/bridge.hip")
    file(GLOB_RECURSE _headers CONFIGURE_DEPENDS "${_source}/*.h" "${_source}/*.hpp" "${_source}/*.cuh")
    set(_objects)
    foreach(_src IN LISTS _hip_sources)
        get_filename_component(_name "${_src}" NAME_WE)
        set(_object "${CMAKE_CURRENT_BINARY_DIR}/hygon-flashmla/${_name}.o")
        add_custom_command(OUTPUT "${_object}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/hygon-flashmla"
            COMMAND "${CMAKE_COMMAND}" -E env CPLUS_INCLUDE_PATH= C_INCLUDE_PATH= CPATH=
                "${_flashmla_aicc}" --rocm-path=${DTK_ROOT} --hip-path=${DTK_ROOT}/hip
                -c "${_src}" -o "${_object}"
                -O3 -std=c++20 -fPIC -w -D__HIP_PLATFORM_AMD__ -D__HIP_PLATFORM_HCC__
                -DUSE_ROCM -DFLASH_MLA_BF16_TYPE=1 -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 -DDCU_ASM
                -ffast-math -fno-finite-math-only --offload-arch=${HYGON_ARCH}
                -mcode-object-version=5 -fno-gpu-rdc --gpu-max-threads-per-block=256
                -mllvm=-support-768-vgprs=true -mllvm=-disable-machine-sink -mllvm=-disable-code-sink
                -mllvm=-amdgpu-enable-rewrite-partial-reg-uses=false
                -mllvm=-allow-gvn-convergent-call=true -mllvm=-disallow-uniform-vmed3-combine=true
                -mllvm=-hcu-pre-emit-load-store-opt=false -mllvm=-amdgpu-early-inline-all=true
                -mllvm=-amdgpu-function-calls=false
                "-I${_source}" "-I${_source}/kerutils/include" "-I${_source}/gfx93"
                "-I${_source}/cutlass/cutlass_3.2.1/include" "-I${DTK_ROOT}/include" "-I${_bridge}"
            DEPENDS "${_src}" "${_bridge}/bridge.hpp" ${_headers}
            VERBATIM)
        list(APPEND _objects "${_object}")
    endforeach()
    add_custom_target(${target} DEPENDS ${_objects})
    set_property(TARGET ${target} PROPERTY INFINIOPS_OBJECTS "${_objects}")
    install(FILES "${_root}/third_party/flashmla_hygon/LICENSE"
        DESTINATION share/infiniops/licenses RENAME FlashMLA-Hygon.txt)
    install(FILES "${_root}/third_party/flashmla_hygon/CUTLASS_LICENSE.txt"
        DESTINATION share/infiniops/licenses RENAME FlashMLA-CUTLASS.txt)
endfunction()
