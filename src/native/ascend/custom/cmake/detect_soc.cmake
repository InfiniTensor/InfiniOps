# Auto-detect the Ascend SOC version from torch_npu or `npu-smi info`.
#
# `infiniops_detect_soc(<out_var>)` first asks torch_npu for the exact device
# name, then falls back to a model-qualified `910*` / `310*` token from
# `npu-smi info`.  Falls back to `Ascend910B4` when detection fails (no NPU
# on the host, missing tools, or an output format without a model suffix).
#
# Called from both `src/CMakeLists.txt` (outer `pip install` build, to
# forward `SOC_VERSION` to the standalone `build.sh` invocation) and
# `src/native/ascend/custom/cmake/config_ascend.cmake` (the sub-build driven
# by that `build.sh`).

function(infiniops_detect_soc out_var)
    set(_detected "")
    set(_python "")

    if(DEFINED _TORCH_PYTHON AND NOT "${_TORCH_PYTHON}" STREQUAL "")
        set(_python "${_TORCH_PYTHON}")
    elseif(DEFINED Python_EXECUTABLE AND NOT "${Python_EXECUTABLE}" STREQUAL "")
        set(_python "${Python_EXECUTABLE}")
    else()
        find_program(_python NAMES python3 python)
    endif()

    if(_python)
        execute_process(
            COMMAND "${_python}" -c "import torch, torch_npu; print(torch.npu.get_device_name(0))"
            RESULT_VARIABLE _torch_npu_result
            OUTPUT_VARIABLE _torch_npu_output
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)

        if(_torch_npu_result EQUAL 0 AND
           _torch_npu_output MATCHES "^Ascend(910|310)[A-Za-z0-9_]*$")
            set(_detected "${_torch_npu_output}")
        endif()
    endif()

    if(NOT _detected)
        execute_process(
            COMMAND npu-smi info
            RESULT_VARIABLE _npu_smi_result
            OUTPUT_VARIABLE _npu_smi_output
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)

        if(_npu_smi_result EQUAL 0)
            string(REGEX MATCH
                "(Ascend)?(910|310)[A-Za-z_][A-Za-z0-9_]*"
                _npu_smi_soc
                "${_npu_smi_output}")

            if(_npu_smi_soc)
                if(_npu_smi_soc MATCHES "^Ascend")
                    set(_detected "${_npu_smi_soc}")
                else()
                    set(_detected "Ascend${_npu_smi_soc}")
                endif()
            endif()
        endif()
    endif()

    if(_detected)
        set(${out_var} "${_detected}" PARENT_SCOPE)
    else()
        set(${out_var} "Ascend910B4" PARENT_SCOPE)
    endif()
endfunction()
