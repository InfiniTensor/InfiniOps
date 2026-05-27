#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/dev_build.sh [cambricon|metax|cpu|auto] [--jobs N]

Examples:
  scripts/dev_build.sh
  scripts/dev_build.sh cambricon
  scripts/dev_build.sh metax --jobs 8

What it does:
  1. Re-runs CMake configure in a persistent build directory.
  2. Incrementally builds the Python extension target (`ops`).
  3. Installs the result into `build-<platform>/install/infini/`.

Why this is faster than `pip install .[dev] --no-build-isolation`:
  - It reuses the same build directory, so unchanged objects are not rebuilt.
  - It avoids wheel build/install work in a temp directory on every run.
  - `[dev]` dependencies only need to be installed once.

Important:
  - This repo's wrapper/codegen runs during CMake configure, not only build.
    So this script always runs `cmake -S -B ...` first, ensuring generator
    changes (`generate_torch_ops.py`, `generate_wrappers.py`, YAML edits) are
    picked up before the incremental build.
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
jobs="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
torch_jobs="${INFINIOPS_TORCH_COMPILE_JOBS:-2}"
binding_jobs="${INFINIOPS_BINDING_COMPILE_JOBS:-2}"

while (($#)); do
    case "$1" in
        cambricon|metax|cpu|auto)
            platform="$1"
            shift
            ;;
        -j|--jobs)
            jobs="$2"
            shift 2
            ;;
        --torch-jobs)
            torch_jobs="$2"
            shift 2
            ;;
        --binding-jobs)
            binding_jobs="$2"
            shift 2
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
python_bin="${PYTHON_BIN:-$(command -v python3)}"

if [[ "$platform" == "auto" ]]; then
    if ! platform="$(detect_platform)"; then
        echo "Could not auto-detect platform. Pass one of: cambricon, metax, cpu." >&2
        exit 1
    fi
fi

with_cpu="ON"
with_torch="ON"
with_cambricon="OFF"
with_metax="OFF"

case "$platform" in
    cambricon)
        with_cambricon="ON"
        ;;
    metax)
        with_metax="ON"
        ;;
    cpu)
        ;;
    *)
        echo "Unsupported platform: $platform" >&2
        exit 1
        ;;
esac

build_dir="${repo_root}/build-${platform}"
install_root="${build_dir}/install"
install_dir="${install_root}/infini"
generator="${CMAKE_GENERATOR:-}"

if [[ -z "${generator}" && -f "${build_dir}/CMakeCache.txt" ]]; then
    generator="$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "${build_dir}/CMakeCache.txt" | head -n 1)"
fi

if [[ -z "${generator}" ]]; then
    generator="Ninja"
fi

echo "[dev_build] repo      : ${repo_root}"
echo "[dev_build] platform  : ${platform}"
echo "[dev_build] python    : ${python_bin}"
echo "[dev_build] build dir : ${build_dir}"
echo "[dev_build] install   : ${install_root}"
echo "[dev_build] package   : ${install_dir}"
echo "[dev_build] generator : ${generator}"
echo "[dev_build] jobs      : build=${jobs} torch=${torch_jobs} binding=${binding_jobs}"

cmake -S "${repo_root}" -B "${build_dir}" -G "${generator}" \
    -DPython_EXECUTABLE="${python_bin}" \
    -DWITH_CPU="${with_cpu}" \
    -DWITH_TORCH="${with_torch}" \
    -DWITH_CAMBRICON="${with_cambricon}" \
    -DWITH_METAX="${with_metax}" \
    -DAUTO_DETECT_DEVICES=OFF \
    -DAUTO_DETECT_BACKENDS=OFF \
    -DGENERATE_PYTHON_BINDINGS=ON \
    -DINFINIOPS_TORCH_COMPILE_JOBS="${torch_jobs}" \
    -DINFINIOPS_BINDING_COMPILE_JOBS="${binding_jobs}"

cmake --build "${build_dir}" --target ops -j "${jobs}"
mkdir -p "${install_root}" "${install_dir}"
rm -f \
    "${install_root}/__init__.py" \
    "${install_root}/libinfiniops.so" \
    "${install_root}/torch_ops_metadata.json" \
    "${install_root}"/ops*.so
mkdir -p "${install_dir}"
cmake --install "${build_dir}" --prefix "${install_dir}"

cat <<EOF

[dev_build] done

Use this build in pytest with:
  PYTHONPATH="${install_root}:\$PYTHONPATH" pytest ...

For example:
  PYTHONPATH="${install_root}:\$PYTHONPATH" pytest --devices ${platform}

Installed files:
  ${install_dir}/ops*.so
  ${install_dir}/torch_ops_metadata.json
EOF
