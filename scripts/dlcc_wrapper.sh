#!/bin/bash
# Filter CMake/host flags unsupported or harmful when invoking Denglin `dlcc`
# as CMAKE_CUDA_COMPILER. Emulate enough of `nvcc -v` for CMake 3.22 Clang
# CUDA detection (TOP / NVVMIR_LIBRARY_DIR).
set -euo pipefail

QY_ROOT="${QY_ROOT:-${QY_HOME:-${QY_PATH:-/usr/local/denglin}}}"
DLCC="${QY_ROOT}/sdk/bin/dlcc"
SDK="${QY_ROOT}/sdk"

if [[ ! -x "${DLCC}" ]]; then
  # Container image layout.
  if [[ -x /usr/local/dlgpu/sdk/bin/dlcc ]]; then
    QY_ROOT=/usr/local/dlgpu
    DLCC=/usr/local/dlgpu/sdk/bin/dlcc
    SDK=/usr/local/dlgpu/sdk
  else
    echo "dlcc_wrapper: dlcc not found; set QY_ROOT" >&2
    exit 1
  fi
fi

# CMakeDetermineCUDACompiler runs `${CMAKE_CUDA_COMPILER} -v __cmake_determine_cuda`.
if [[ "${1:-}" == "-v" ]]; then
  echo "clang version 15.0.6"
  echo "Cuda compilation tools, release 11.7, V11.7.0"
  echo "#\$ TOP=${SDK}"
  # Path need only match CMake's `nvvm/libdevice$` regex; Denglin has no nvvm tree.
  echo "#\$ NVVMIR_LIBRARY_DIR=${SDK}/nvvm/libdevice"
  "${DLCC}" --version || true
  exit 0
fi

ARGS=()
skip_next=0
has_cuda_path=0
replaced_cudart=0
for arg in "$@"; do
  if [[ "${skip_next}" -eq 1 ]]; then
    skip_next=0
    continue
  fi
  case "${arg}" in
    -pthread) ;;
    -B) skip_next=1 ;;
    -B*) ;;
    --cuda-path=*)
      has_cuda_path=1
      ARGS+=("${arg}")
      ;;
    # CMake Clang detection probes sm_20/30/52; Denglin only accepts dlgput64.
    --cuda-gpu-arch=sm_*) ;;
    --cuda-gpu-arch=dlgput64)
      ARGS+=("${arg}")
      ;;
    --cuda-gpu-arch=*) ;;
    -Xcompiler=*) ;;
    # Match pre-refactor xmake/qy.lua: drop NVIDIA static CUDA libs.
    -lcudadevrt|-lcudart_static|-lcudart)
      if [[ "${replaced_cudart}" -eq 0 ]]; then
        ARGS+=("-lcurt")
        replaced_cudart=1
      fi
      ;;
    *)
      ARGS+=("${arg}")
      ;;
  esac
done

EXTRA=()
if [[ "${has_cuda_path}" -eq 0 ]]; then
  EXTRA+=("--cuda-path=${SDK}")
fi
# Always compile for Denglin GPU; ignore CMake sm_* probes.
EXTRA+=("--cuda-gpu-arch=dlgput64")

exec "${DLCC}" "${EXTRA[@]}" "${ARGS[@]}"
