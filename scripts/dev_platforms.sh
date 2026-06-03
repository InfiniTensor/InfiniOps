#!/usr/bin/env bash

INFINIOPS_SUPPORTED_PLATFORMS=(
    cpu
    nvidia
    iluvatar
    hygon
    metax
    moore
    cambricon
    ascend
)

infiniops_supported_platforms_usage() {
    local IFS='|'
    printf '%s|auto' "${INFINIOPS_SUPPORTED_PLATFORMS[*]}"
}

infiniops_supported_platforms_csv() {
    local IFS=', '
    printf '%s' "${INFINIOPS_SUPPORTED_PLATFORMS[*]}"
}

infiniops_is_supported_platform() {
    local candidate="${1:-}"
    local platform

    for platform in "${INFINIOPS_SUPPORTED_PLATFORMS[@]}"; do
        if [[ "${platform}" == "${candidate}" ]]; then
            return 0
        fi
    done

    return 1
}

_infiniops_has_glob_match() {
    compgen -G "$1" > /dev/null
}

_infiniops_find_hygon_cuda_root() {
    local dtk_root="$1"
    local candidate
    local versioned_candidates=()

    for candidate in \
        "${dtk_root}/cuda" \
        "${dtk_root}/cuda/cuda"; do
        if [[ -x "${candidate}/bin/nvcc" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    shopt -s nullglob
    versioned_candidates=("${dtk_root}"/cuda/cuda-*)
    shopt -u nullglob

    for candidate in "${versioned_candidates[@]}"; do
        if [[ -x "${candidate}/bin/nvcc" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    return 1
}

_infiniops_detect_nvidia() {
    _infiniops_has_glob_match "${INFINIOPS_NVIDIA_DEVICE_GLOB:-/dev/nvidia*}"
}

_infiniops_detect_iluvatar() {
    _infiniops_has_glob_match "${INFINIOPS_ILUVATAR_DEVICE_GLOB:-/dev/iluvatar*}"
}

_infiniops_detect_hygon() {
    local dtk_root="${DTK_ROOT:-${INFINIOPS_HYGON_DTK_ROOT:-/opt/dtk}}"

    _infiniops_find_hygon_cuda_root "${dtk_root}" > /dev/null
}

_infiniops_detect_metax() {
    if [[ -n "${MACA_PATH:-}" ]]; then
        return 0
    fi

    grep -hqs 9999 ${INFINIOPS_METAX_PCI_VENDOR_GLOB:-/sys/bus/pci/devices/*/vendor} 2> /dev/null
}

_infiniops_detect_moore() {
    [[ -n "${MUSA_ROOT:-}" || -n "${MUSA_HOME:-}" || -n "${MUSA_PATH:-}" ]]
}

_infiniops_detect_cambricon() {
    [[ -n "${NEUWARE_HOME:-}" ]]
}

_infiniops_detect_ascend() {
    [[ -n "${ASCEND_HOME_PATH:-}" ]] || \
        _infiniops_has_glob_match "${INFINIOPS_ASCEND_DEVICE_GLOB:-/dev/davinci0}"
}

infiniops_detect_platforms() {
    local detected=()

    if _infiniops_detect_nvidia; then
        detected+=("nvidia")
    fi

    if _infiniops_detect_iluvatar; then
        detected+=("iluvatar")
    fi

    if _infiniops_detect_hygon; then
        detected+=("hygon")
    fi

    if _infiniops_detect_metax; then
        detected+=("metax")
    fi

    if _infiniops_detect_cambricon; then
        detected+=("cambricon")
    fi

    if _infiniops_detect_moore; then
        detected+=("moore")
    fi

    if _infiniops_detect_ascend; then
        detected+=("ascend")
    fi

    if ((${#detected[@]} > 0)); then
        printf '%s\n' "${detected[@]}"
    fi
}
