
if(DEFINED ASCEND_HOME_PATH)
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "ASCEND CANN package installation directory" FORCE)
endif()

set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})

# Auto-detect `SOC_VERSION` from `npu-smi info` if not set externally.
# Required by `CANN`'s `ascendc.cmake` for `AscendC` kernel compilation.
if(NOT DEFINED SOC_VERSION OR "${SOC_VERSION}" STREQUAL "")
    execute_process(
        COMMAND bash -c "npu-smi info 2>/dev/null | awk '/910B|910A|310/ {for (i=1;i<=NF;i++) if ($i ~ /^(910|310)/) {print \"Ascend\" $i; exit}}'"
        OUTPUT_VARIABLE _DETECTED_SOC
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(_DETECTED_SOC)
        set(SOC_VERSION "${_DETECTED_SOC}" CACHE STRING "Ascend SOC version" FORCE)
    else()
        set(SOC_VERSION "Ascend910B4" CACHE STRING "Ascend SOC version" FORCE)
    endif()

    message(STATUS "SOC_VERSION auto-set to ${SOC_VERSION}")
endif()

if(EXISTS ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)


message(STATUS "ASCEND_CANN_PACKAGE_PATH = ${ASCEND_CANN_PACKAGE_PATH}")
message(STATUS "ASCEND_HOME_PATH = ${ASCEND_HOME_PATH}")
