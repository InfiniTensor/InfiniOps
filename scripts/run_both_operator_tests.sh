#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OPS_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

OPS_ROOT="${DEFAULT_OPS_ROOT}"
CORE_ROOT="${DEFAULT_OPS_ROOT}/../InfiniCore"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CORE_ROOT_EXPLICIT=0
OPS_ROOT_EXPLICIT=0
PYTHON_LIBDIR=""
VISIBLE_GPUS=""
DEFAULT_SKIP_CORE_OPS_ENABLED=1

RUN_CORE=1
RUN_OPS=1
SKIP_CORE_BUILD=0
SKIP_CORE_CLEAN=0
SKIP_OPS_BUILD=0
CORE_ISOLATE_OPS=0
DRY_RUN=0

PLATFORMS=()
SKIP_CORE_OPS=()
SKIP_CORE_OPS_EXPLICIT=0
CORE_TEST_ARGS=()
OPS_PYTEST_ARGS=()

COMMON_PLATFORMS=(cpu nvidia cambricon ascend iluvatar metax moore hygon)
DEFAULT_SKIP_CORE_OPS=(
    logdet
    index_copy
    asum
    axpy
    blas_amax
    blas_amin
    blas_copy
    blas_dot
    flash_attention
    nrm2
    rot
    rotg
    rotm
    rotmg
    scal
    silu_and_mul
    swap
    fmod
    logical_and
    logical_not
    mha_varlen
    sinh
    upsample_nearest
)

usage() {
    cat <<'EOF'
Usage: scripts/run_both_operator_tests.sh [options]

Build and run operator tests for both InfiniCore and InfiniOps.

Options:
  --platform <name>     Platform shared by both projects. Repeatable.
                        Supported: cpu, nvidia, cambricon, ascend,
                        iluvatar, metax, moore, hygon.
                        Default: nvidia
  --core-root <path>    Path to the InfiniCore repository.
                        Default: ../InfiniCore relative to InfiniOps
  --ops-root <path>     Path to the InfiniOps repository.
                        Default: repo containing this script
  --gpus <list>         Set CUDA_VISIBLE_DEVICES for both test flows.
                        Examples: 0, 4, 4,5
  --skip-core-op <op>   Skip an InfiniCore operator by name. Repeatable.
                        Supports comma-separated values.
                        Example: --skip-core-op logdet,index_copy
  --no-default-skip-core-ops
                        Disable the built-in InfiniCore skip list for known
                        crashing/failing operators.
  --core-isolate-ops    Run InfiniCore one operator per Python process.
                        Useful for pinpointing a segfaulting operator.
  --core-only           Build and test InfiniCore only
  --ops-only            Build and test InfiniOps only
  --skip-core-build     Skip the InfiniCore build/install steps
  --skip-core-clean     Skip `xmake f -c` and `xmake clean --all`
  --skip-ops-build      Skip `pip install .[dev] --no-build-isolation`
  --core-arg <arg>      Extra arg passed to `test/infinicore/run.py`.
                        Repeatable.
  --pytest-arg <arg>    Extra arg passed to `pytest`. Repeatable.
  --dry-run             Print commands without executing them
  -h, --help            Show this help

Examples:
  scripts/run_both_operator_tests.sh
  scripts/run_both_operator_tests.sh --gpus 4
  scripts/run_both_operator_tests.sh --gpus 4 --core-only --skip-core-build
  scripts/run_both_operator_tests.sh --gpus 4 --skip-core-op logdet,index_copy
  scripts/run_both_operator_tests.sh --gpus 4 --no-default-skip-core-ops
  scripts/run_both_operator_tests.sh --gpus 4 --core-only --skip-core-build --core-isolate-ops
  scripts/run_both_operator_tests.sh --platform cpu
  scripts/run_both_operator_tests.sh --skip-core-build --skip-ops-build
  scripts/run_both_operator_tests.sh --core-only --core-arg --verbose
  scripts/run_both_operator_tests.sh --pytest-arg -k --pytest-arg matmul
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

log() {
    echo "==> $*"
}

require_value() {
    local option="$1"

    [[ $# -ge 2 && -n "${2}" ]] || die "Missing value for ${option}"
}

normalize_platform() {
    local value="$1"
    echo "${value,,}"
}

platform_supported() {
    local candidate="$1"
    local platform

    for platform in "${COMMON_PLATFORMS[@]}"; do
        if [[ "${platform}" == "${candidate}" ]]; then
            return 0
        fi
    done

    return 1
}

dedupe_platforms() {
    local unique=()
    local platform
    declare -A seen=()

    for platform in "${PLATFORMS[@]}"; do
        if [[ -n "${seen[${platform}]:-}" ]]; then
            continue
        fi

        seen["${platform}"]=1
        unique+=("${platform}")
    done

    PLATFORMS=("${unique[@]}")
}

resolve_dir() {
    local path="$1"

    [[ -d "${path}" ]] || die "Directory not found: ${path}"
    (
        cd -- "${path}"
        pwd
    )
}

prepend_env_path() {
    local var_name="$1"
    local new_path="$2"
    local current_value="${!var_name:-}"

    [[ -n "${new_path}" ]] || return 0
    [[ -d "${new_path}" ]] || return 0

    case ":${current_value}:" in
        *":${new_path}:"*)
            return 0
            ;;
    esac

    if [[ -n "${current_value}" ]]; then
        export "${var_name}=${new_path}:${current_value}"
    else
        export "${var_name}=${new_path}"
    fi
}

detect_python_libdir() {
    "${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")'
}

trim_whitespace() {
    local value="$1"

    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s\n' "${value}"
}

append_skip_core_ops() {
    local raw_value="$1"
    local part
    local cleaned_part
    local parts=()

    IFS=',' read -r -a parts <<< "${raw_value}"

    for part in "${parts[@]}"; do
        cleaned_part="$(trim_whitespace "${part}")"
        [[ -n "${cleaned_part}" ]] || continue
        SKIP_CORE_OPS+=("${cleaned_part}")
    done
}

dedupe_skip_core_ops() {
    local unique=()
    local op
    declare -A seen=()

    for op in "${SKIP_CORE_OPS[@]}"; do
        if [[ -n "${seen[${op}]:-}" ]]; then
            continue
        fi

        seen["${op}"]=1
        unique+=("${op}")
    done

    SKIP_CORE_OPS=("${unique[@]}")
}

list_core_ops() {
    local ops_dir="$1"
    local file
    local op

    shopt -s nullglob
    for file in "${ops_dir}"/*.py; do
        op="${file##*/}"
        op="${op%.py}"
        [[ "${op}" == "__init__" ]] && continue
        printf '%s\n' "${op}"
    done
    shopt -u nullglob
}

core_args_contain_ops_flag() {
    local arg

    for arg in "${CORE_TEST_ARGS[@]}"; do
        if [[ "${arg}" == "--ops" ]]; then
            return 0
        fi
    done

    return 1
}

print_subshell_cmd() {
    local dir="$1"
    shift

    printf '+ (cd %q &&' "${dir}"
    printf ' %q' "$@"
    printf ')\n'
}

run_in_dir() {
    local dir="$1"
    shift

    if (( DRY_RUN )); then
        print_subshell_cmd "${dir}" "$@"
        return 0
    fi

    print_subshell_cmd "${dir}" "$@"
    (
        cd -- "${dir}"
        "$@"
    )
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)
            require_value "$1" "${2-}"
            PLATFORMS+=("$(normalize_platform "$2")")
            shift 2
            ;;
        --platform=*)
            PLATFORMS+=("$(normalize_platform "${1#*=}")")
            shift
            ;;
        --core-root)
            require_value "$1" "${2-}"
            CORE_ROOT="$2"
            CORE_ROOT_EXPLICIT=1
            shift 2
            ;;
        --core-root=*)
            CORE_ROOT="${1#*=}"
            CORE_ROOT_EXPLICIT=1
            shift
            ;;
        --ops-root)
            require_value "$1" "${2-}"
            OPS_ROOT="$2"
            OPS_ROOT_EXPLICIT=1
            shift 2
            ;;
        --ops-root=*)
            OPS_ROOT="${1#*=}"
            OPS_ROOT_EXPLICIT=1
            shift
            ;;
        --gpus)
            require_value "$1" "${2-}"
            VISIBLE_GPUS="$2"
            shift 2
            ;;
        --gpus=*)
            VISIBLE_GPUS="${1#*=}"
            shift
            ;;
        --skip-core-op)
            require_value "$1" "${2-}"
            SKIP_CORE_OPS_EXPLICIT=1
            append_skip_core_ops "$2"
            shift 2
            ;;
        --skip-core-op=*)
            SKIP_CORE_OPS_EXPLICIT=1
            append_skip_core_ops "${1#*=}"
            shift
            ;;
        --no-default-skip-core-ops)
            DEFAULT_SKIP_CORE_OPS_ENABLED=0
            shift
            ;;
        --core-isolate-ops)
            CORE_ISOLATE_OPS=1
            shift
            ;;
        --core-only)
            RUN_OPS=0
            shift
            ;;
        --ops-only)
            RUN_CORE=0
            shift
            ;;
        --skip-core-build)
            SKIP_CORE_BUILD=1
            shift
            ;;
        --skip-core-clean)
            SKIP_CORE_CLEAN=1
            shift
            ;;
        --skip-ops-build)
            SKIP_OPS_BUILD=1
            shift
            ;;
        --core-arg)
            require_value "$1" "${2-}"
            CORE_TEST_ARGS+=("$2")
            shift 2
            ;;
        --pytest-arg)
            require_value "$1" "${2-}"
            OPS_PYTEST_ARGS+=("$2")
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

if (( RUN_CORE == 0 && RUN_OPS == 0 )); then
    die "Cannot combine --core-only and --ops-only"
fi

if (( DEFAULT_SKIP_CORE_OPS_ENABLED )) && ! core_args_contain_ops_flag; then
    SKIP_CORE_OPS=("${DEFAULT_SKIP_CORE_OPS[@]}" "${SKIP_CORE_OPS[@]}")
fi

dedupe_skip_core_ops

if [[ ${#PLATFORMS[@]} -eq 0 ]]; then
    PLATFORMS=(nvidia)
fi

if (( OPS_ROOT_EXPLICIT == 1 && CORE_ROOT_EXPLICIT == 0 )); then
    CORE_ROOT="${OPS_ROOT}/../InfiniCore"
fi

dedupe_platforms

for platform in "${PLATFORMS[@]}"; do
    if ! platform_supported "${platform}"; then
        die "Unsupported platform '${platform}'. Supported: ${COMMON_PLATFORMS[*]}"
    fi
done

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python executable not found: ${PYTHON_BIN}"
PYTHON_LIBDIR="$(detect_python_libdir)"
prepend_env_path LD_LIBRARY_PATH "${PYTHON_LIBDIR}"
if [[ -n "${VISIBLE_GPUS}" ]]; then
    export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"
fi

if (( RUN_CORE )); then
    if (( SKIP_CORE_BUILD == 0 )); then
        command -v xmake >/dev/null 2>&1 || die "xmake is not available in PATH"
    fi
    CORE_ROOT="$(resolve_dir "${CORE_ROOT}")"
    [[ -f "${CORE_ROOT}/scripts/install.py" ]] || die "Missing ${CORE_ROOT}/scripts/install.py"
    [[ -f "${CORE_ROOT}/test/infinicore/run.py" ]] || die "Missing ${CORE_ROOT}/test/infinicore/run.py"
fi

if (( RUN_OPS )); then
    OPS_ROOT="$(resolve_dir "${OPS_ROOT}")"
    [[ -f "${OPS_ROOT}/pyproject.toml" ]] || die "Missing ${OPS_ROOT}/pyproject.toml"
    [[ -d "${OPS_ROOT}/tests" ]] || die "Missing ${OPS_ROOT}/tests"
fi

log "Python executable: ${PYTHON_BIN}"
if [[ -n "${PYTHON_LIBDIR}" ]]; then
    log "Python runtime libdir: ${PYTHON_LIBDIR}"
fi
if [[ -n "${VISIBLE_GPUS}" ]]; then
    log "CUDA_VISIBLE_DEVICES: ${VISIBLE_GPUS}"
fi
log "Platforms: ${PLATFORMS[*]}"

if (( RUN_CORE )); then
    log "InfiniCore root: ${CORE_ROOT}"
    if (( DEFAULT_SKIP_CORE_OPS_ENABLED )) && ! core_args_contain_ops_flag; then
        log "Applying built-in InfiniCore skip list (${#DEFAULT_SKIP_CORE_OPS[@]} ops)"
    fi

    if (( SKIP_CORE_BUILD == 0 )); then
        log "Building InfiniCore"

        if (( SKIP_CORE_CLEAN == 0 )); then
            run_in_dir "${CORE_ROOT}" xmake f -c
            run_in_dir "${CORE_ROOT}" xmake clean --all
        fi

        run_in_dir "${CORE_ROOT}" "${PYTHON_BIN}" scripts/install.py --nv-gpu=y --ccl=y
        run_in_dir "${CORE_ROOT}" xmake build _infinicore
        run_in_dir "${CORE_ROOT}" xmake install _infinicore
        run_in_dir "${CORE_ROOT}" "${PYTHON_BIN}" -m pip install -e . --no-build-isolation
    else
        log "Skipping InfiniCore build"
    fi

    core_cmd=("${PYTHON_BIN}" test/infinicore/run.py)
    core_selected_ops=()
    for platform in "${PLATFORMS[@]}"; do
        core_cmd+=("--${platform}")
    done

    if (( CORE_ISOLATE_OPS )) || [[ ${#SKIP_CORE_OPS[@]} -gt 0 ]]; then
        local_all_core_ops=()
        local_unknown_skip_ops=()
        declare -A skip_core_op_set=()
        declare -A available_core_op_set=()
        op=""

        if core_args_contain_ops_flag && (( CORE_ISOLATE_OPS || SKIP_CORE_OPS_EXPLICIT )); then
            die "Cannot combine --core-isolate-ops or explicit --skip-core-op with --core-arg --ops. Use one filtering mode."
        fi

        while IFS= read -r op; do
            [[ -n "${op}" ]] || continue
            local_all_core_ops+=("${op}")
            available_core_op_set["${op}"]=1
        done < <(list_core_ops "${CORE_ROOT}/test/infinicore/ops")

        [[ ${#local_all_core_ops[@]} -gt 0 ]] || die "No InfiniCore ops found under ${CORE_ROOT}/test/infinicore/ops"

        for op in "${SKIP_CORE_OPS[@]}"; do
            skip_core_op_set["${op}"]=1
            if [[ -z "${available_core_op_set[${op}]:-}" ]]; then
                local_unknown_skip_ops+=("${op}")
            fi
        done

        if [[ ${#local_unknown_skip_ops[@]} -gt 0 ]]; then
            log "Warning: unknown InfiniCore ops to skip: ${local_unknown_skip_ops[*]}"
        fi

        for op in "${local_all_core_ops[@]}"; do
            if [[ -n "${skip_core_op_set[${op}]:-}" ]]; then
                continue
            fi

            core_selected_ops+=("${op}")
        done

        [[ ${#core_selected_ops[@]} -gt 0 ]] || die "No InfiniCore operators left after applying --skip-core-op"

        if [[ ${#SKIP_CORE_OPS[@]} -gt 0 ]]; then
            log "Skipping InfiniCore operators: ${SKIP_CORE_OPS[*]}"
        fi

        if (( CORE_ISOLATE_OPS == 0 )); then
            core_cmd+=("--ops")
            core_cmd+=("${core_selected_ops[@]}")
        fi
    fi

    if (( CORE_ISOLATE_OPS )); then
        if [[ ${#core_selected_ops[@]} -eq 0 ]]; then
            while IFS= read -r op; do
                [[ -n "${op}" ]] || continue
                core_selected_ops+=("${op}")
            done < <(list_core_ops "${CORE_ROOT}/test/infinicore/ops")
        fi

        log "Running InfiniCore operator tests in isolated mode"
        isolated_failed_core_ops=()

        for op in "${core_selected_ops[@]}"; do
            op_cmd=("${PYTHON_BIN}" test/infinicore/run.py)
            for platform in "${PLATFORMS[@]}"; do
                op_cmd+=("--${platform}")
            done
            op_cmd+=("--ops" "${op}")
            op_cmd+=("${CORE_TEST_ARGS[@]}")

            log "Running InfiniCore operator: ${op}"
            set +e
            run_in_dir "${CORE_ROOT}" "${op_cmd[@]}"
            rc=$?
            set -e

            if (( rc == 0 )); then
                continue
            fi

            if (( rc >= 128 )); then
                log "InfiniCore operator crashed: ${op} (exit ${rc}, signal $((rc - 128)))"
                exit "${rc}"
            fi

            log "InfiniCore operator failed: ${op} (exit ${rc})"
            isolated_failed_core_ops+=("${op}:${rc}")
        done

        if [[ ${#isolated_failed_core_ops[@]} -gt 0 ]]; then
            log "InfiniCore isolated mode completed with failing operators: ${isolated_failed_core_ops[*]}"
            exit 1
        fi
    else
        core_cmd+=("${CORE_TEST_ARGS[@]}")

        log "Running InfiniCore operator tests"
        run_in_dir "${CORE_ROOT}" "${core_cmd[@]}"
    fi
fi

if (( RUN_OPS )); then
    log "InfiniOps root: ${OPS_ROOT}"

    if (( SKIP_OPS_BUILD == 0 )); then
        log "Building InfiniOps"
        run_in_dir "${OPS_ROOT}" "${PYTHON_BIN}" -m pip install ".[dev]" --no-build-isolation
    else
        log "Skipping InfiniOps build"
    fi

    ops_cmd=("${PYTHON_BIN}" -m pytest --devices)
    ops_cmd+=("${PLATFORMS[@]}")
    ops_cmd+=("${OPS_PYTEST_ARGS[@]}")

    log "Running InfiniOps operator tests"
    run_in_dir "${OPS_ROOT}" "${ops_cmd[@]}"
fi

log "Done"
