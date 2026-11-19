# Query the runtime for the exact SOC name before consulting `npu-smi`.
#
# Called from both `src/CMakeLists.txt` (outer `pip install` build, to
# forward `SOC_VERSION` to the standalone `build.sh` invocation) and
# `src/native/ascend/custom/cmake/config_ascend.cmake` (the sub-build driven
# by that `build.sh`).

function(infiniops_detect_soc out_var)
    set(_soc_pattern "910(B[1-4](-1)?|A|ProA|ProB|PremiumA|_[0-9][0-9A-Za-z]*)|310[BP][0-9]+(vir[0-9]+)?")
    if(_TORCH_PYTHON)
        set(_soc_python "${_TORCH_PYTHON}")
    elseif(Python_EXECUTABLE)
        set(_soc_python "${Python_EXECUTABLE}")
    else()
        find_program(_soc_python NAMES python3 python)
    endif()

    if(_soc_python)
        execute_process(
            COMMAND "${_soc_python}" -c
                "import torch, torch_npu; print(torch.npu.get_device_name(0))"
            OUTPUT_VARIABLE _runtime_soc
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _runtime_result
            ERROR_QUIET
            TIMEOUT 30)
        if(_runtime_result STREQUAL "0")
            string(REGEX MATCH "^(Ascend)?(${_soc_pattern})$"
                _runtime_match "${_runtime_soc}")
            if(_runtime_match)
                set(${out_var} "Ascend${CMAKE_MATCH_2}" PARENT_SCOPE)
                return()
            endif()
        endif()
    endif()

    execute_process(
        COMMAND npu-smi info
        OUTPUT_VARIABLE _smi_output
        RESULT_VARIABLE _smi_result
        ERROR_QUIET
        TIMEOUT 10)
    if(_smi_result STREQUAL "0")
        # Require model-qualified tokens so HBM usage such as `3103` is ignored.
        string(REGEX MATCH "(^|[ \t\r\n|])(Ascend)?(${_soc_pattern})([ \t\r\n|]|$)"
            _smi_match "${_smi_output}")
        if(_smi_match)
            set(${out_var} "Ascend${CMAKE_MATCH_3}" PARENT_SCOPE)
            return()
        endif()
    endif()

    message(FATAL_ERROR
        "Cannot determine an exact Ascend SOC version. "
        "Set SOC_VERSION explicitly (pip: --config-settings=cmake.define.SOC_VERSION=<soc>). "
        "Generic model names and numeric npu-smi fields are not valid SOC versions.")
endfunction()
