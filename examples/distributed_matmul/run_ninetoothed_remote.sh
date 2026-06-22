#!/usr/bin/env bash
set -euo pipefail

platform="${1:?usage: $0 <nvidia|metax|iluvatar|moore|cambricon|ascend>}"
case "${platform}" in
  nvidia) host="${HOST:-nvidia}"; image="${IMAGE:-accelerator-dev/nvidia:latest}"; docker_args="--gpus=all" ;;
  metax) host="${HOST:-metax}"; image="${IMAGE:-accelerator-dev/metax:latest}"; docker_args="--privileged" ;;
  iluvatar) host="${HOST:-iluvatar}"; image="${IMAGE:-accelerator-dev/iluvatar:latest}"; docker_args="--privileged" ;;
  moore) host="${HOST:-moore}"; image="${IMAGE:-accelerator-dev/moore:latest}"; docker_args="--privileged" ;;
  cambricon) host="${HOST:-cambricon}"; image="${IMAGE:-accelerator-dev/cambricon:latest}"; docker_args="--privileged" ;;
  ascend) host="${HOST:-ascend}"; image="${IMAGE:-accelerator-dev/ascend:latest}"; docker_args="--privileged" ;;
  *) echo "unknown platform: ${platform}" >&2; exit 2 ;;
esac

source_root="${NINETOOTHED_ROOT:-/home/voltjia/ninetoothed}"
remote_root="${REMOTE_ROOT:-/tmp/ninetoothed-matmul-${USER}}"
m_size="${M:-256}"
n_size="${N:-256}"
k_size="${K:-256}"
warmup="${WARMUP:-1}"
iters="${ITERS:-2}"
atol="${ATOL:-2e-2}"
rtol="${RTOL:-2e-2}"
docker_timeout="${NINETOOTHED_DOCKER_TIMEOUT:-900}"
ssh_opts=(-o LogLevel=ERROR)


if [ ! -d "${source_root}" ]; then
  echo "ninetoothed source not found: ${source_root}" >&2
  exit 2
fi

ssh "${ssh_opts[@]}" "${host}" "mkdir -p ${remote_root}/ninetoothed"
rsync -az --delete --exclude .git --exclude __pycache__ --exclude .pytest_cache -e "ssh -o LogLevel=ERROR" "${source_root}/" "${host}:${remote_root}/ninetoothed/"
rsync -az -e "ssh -o LogLevel=ERROR" "$(dirname "${BASH_SOURCE[0]}")/ninetoothed_matmul_demo.py" "${host}:${remote_root}/ninetoothed_matmul_demo.py"

ssh "${ssh_opts[@]}" "${host}" bash -s -- "${remote_root}" <<'REMOTE_SETUP'
set -euo pipefail
remote_root="$1"
cat >"${remote_root}/run_ninetoothed_inside.sh" <<'INNER'
#!/usr/bin/env bash
set -euo pipefail
export PATH="/opt/conda/bin:/usr/local/bin:${PATH}"
cd /workspace/ninetoothed
export PYTHONPATH="/workspace/ninetoothed/src:/workspace/ninetoothed:${PYTHONPATH:-}"
python3 -m pip install -q -i "${PYPI_INDEX_URL:-https://pypi.org/simple}" pytest sympy numpy || true
python3 - <<'PY'
import sys
import torch
for module in ("torch_mlu", "torch_musa", "torch_npu"):
    try:
        __import__(module)
    except Exception as exc:
        print(f"{module}_import", repr(exc))
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count() if torch.cuda.is_available() else 0)
print("mlu_available", bool(hasattr(torch, "mlu") and torch.mlu.is_available()))
print("musa_available", bool(hasattr(torch, "musa") and torch.musa.is_available()))
print("npu_available", bool(hasattr(torch, "npu") and torch.npu.is_available()))
try:
    import triton
    print("triton", triton.__version__)
except Exception as exc:
    print("triton_error", repr(exc))
PY
python3 /workspace/ninetoothed_matmul_demo.py --m "${NINETOOTHED_M}" --n "${NINETOOTHED_N}" --k "${NINETOOTHED_K}" --warmup "${NINETOOTHED_WARMUP}" --iters "${NINETOOTHED_ITERS}" --atol "${NINETOOTHED_ATOL}" --rtol "${NINETOOTHED_RTOL}"
INNER
chmod +x "${remote_root}/run_ninetoothed_inside.sh"
REMOTE_SETUP

ssh "${ssh_opts[@]}" "${host}" bash -s -- "${remote_root}" "${image}" "${platform}" "${m_size}" "${n_size}" "${k_size}" "${warmup}" "${iters}" "${atol}" "${rtol}" "${docker_timeout}" "${docker_args}" <<'REMOTE_RUN'
set -euo pipefail
remote_root="$1"
image="$2"
platform="$3"
m_size="$4"
n_size="$5"
k_size="$6"
warmup="$7"
iters="$8"
atol="$9"
rtol="${10}"
docker_timeout="${11}"
docker_args="${12}"
container="ninetoothed-matmul-${platform}-$(id -u)"
log_file="${remote_root}/ninetoothed_${platform}.out"

docker rm -f "${container}" >/dev/null 2>&1 || true
set +e
timeout "${docker_timeout}" docker run --rm --name "${container}" --network host --ipc host --ulimit memlock=-1 --ulimit stack=67108864 ${docker_args} -e NINETOOTHED_M="${m_size}" -e NINETOOTHED_N="${n_size}" -e NINETOOTHED_K="${k_size}" -e NINETOOTHED_WARMUP="${warmup}" -e NINETOOTHED_ITERS="${iters}" -e NINETOOTHED_ATOL="${atol}" -e NINETOOTHED_RTOL="${rtol}" -v "${remote_root}:/workspace" "${image}" /workspace/run_ninetoothed_inside.sh 2>&1 | tee "${log_file}"
status="${PIPESTATUS[0]}"
set -e
if [ "${status}" -eq 124 ]; then
  echo "ninetoothed pytest timed out after ${docker_timeout}s"
fi
exit "${status}"
REMOTE_RUN
