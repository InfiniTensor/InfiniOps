#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/dev_test.sh [cambricon|metax|cpu|auto] [--report PATH] [--no-build] [-- pytest-args...]

Examples:
  scripts/dev_test.sh cambricon
  scripts/dev_test.sh metax
  scripts/dev_test.sh cambricon -- tests/test_torch_ops.py -k index

Default behavior:
  1. Run `scripts/dev_build.sh <platform>`
  2. Run pytest with `--devices <platform> --op-report reports/<platform>.json`
  3. Print case-level and operator-level skip stats

Notes:
  - `--no-build` skips the build step and reuses the last local install.
  - Extra args after `--` are passed straight to pytest.
EOF
}

detect_platform() {
    if [[ -n "${NEUWARE_HOME:-}" ]]; then
        echo "cambricon"
        return 0
    fi

    if [[ -n "${MACA_PATH:-}" ]]; then
        echo "metax"
        return 0
    fi

    return 1
}

platform="auto"
report_path=""
do_build=1
pytest_args=()
pytest_targets=()

while (($#)); do
    case "$1" in
        cambricon|metax|cpu|auto)
            platform="$1"
            shift
            ;;
        --report)
            report_path="$2"
            shift 2
            ;;
        --no-build)
            do_build=0
            shift
            ;;
        --)
            shift
            pytest_args=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$platform" == "auto" ]]; then
    if ! platform="$(detect_platform)"; then
        echo "Could not auto-detect platform. Pass one of: cambricon, metax, cpu." >&2
        exit 1
    fi
fi

if [[ -z "$report_path" ]]; then
    report_path="${repo_root}/reports/${platform}.json"
fi

install_dir="${repo_root}/build-${platform}/install"
summary_script="${repo_root}/scripts/summarize_op_report.py"
detail_path="${report_path%.json}.details.jsonl"
text_path="${report_path%.json}.summary.txt"

if [[ "$do_build" -eq 1 ]]; then
    "${repo_root}/scripts/dev_build.sh" "$platform"
fi

if [[ ! -d "$install_dir" ]]; then
    echo "Install dir not found: $install_dir" >&2
    echo "Run scripts/dev_build.sh $platform first." >&2
    exit 1
fi

mkdir -p "$(dirname "$report_path")"
rm -f "$report_path" "$detail_path" "$text_path"

for arg in "${pytest_args[@]}"; do
    candidate="${arg%%::*}"

    if [[ -e "$candidate" || -e "${repo_root}/$candidate" ]]; then
        pytest_targets+=("$arg")
    fi
done

if [[ ${#pytest_targets[@]} -eq 0 ]]; then
    pytest_targets=("tests")
fi

echo "[dev_test] platform : ${platform}"
echo "[dev_test] report   : ${report_path}"
echo "[dev_test] install  : ${install_dir}"
echo "[dev_test] targets  : ${pytest_targets[*]}"

set +e
PYTHONPATH="${install_dir}${PYTHONPATH:+:${PYTHONPATH}}" \
    python3 -m pytest "${pytest_targets[@]}" --devices "${platform}" --op-report "${report_path}" "${pytest_args[@]}"
pytest_status=$?
set -e

if [[ -f "$report_path" ]]; then
    echo ""
    python3 "$summary_script" "$report_path"
else
    echo "[dev_test] report not found: ${report_path}" >&2
fi

exit "$pytest_status"
