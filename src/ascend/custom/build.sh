#!/bin/bash
# Build custom AscendC kernels into libascend_kernel.so.
set -e

SOC_VERSION="${1:-Ascend910_9382}"

# Detect CANN toolkit path.
_CANN_TOOLKIT_INSTALL_PATH=$(grep "Toolkit_InstallPath" /etc/Ascend/ascend_cann_install.info | awk -F'=' '{print $2}')
source "${_CANN_TOOLKIT_INSTALL_PATH}/set_env.sh"
echo "CANN: ${ASCEND_TOOLKIT_HOME}"

ASCEND_INCLUDE_DIR=${ASCEND_TOOLKIT_HOME}/$(arch)-linux/include
CURRENT_DIR=$(pwd)
OUTPUT_DIR=${CURRENT_DIR}/output
mkdir -p "${OUTPUT_DIR}"

BUILD_DIR=build
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

cmake \
    -DASCEND_HOME_PATH="${ASCEND_HOME_PATH}" \
    -DASCEND_INCLUDE_DIR="${ASCEND_INCLUDE_DIR}" \
    -DSOC_VERSION="${SOC_VERSION}" \
    -B "${BUILD_DIR}" \
    -S .

cmake --build "${BUILD_DIR}" -j 16

echo "Build complete. Output: ${OUTPUT_DIR}"
