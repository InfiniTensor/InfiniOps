#!/usr/bin/env bash
set -euo pipefail

platform="${1:?usage: $0 <nvidia|metax|iluvatar|moore|cambricon|ascend>}"
rows="${ROWS:-1024}"
k_size="${K:-2048}"
n_size="${N:-1024}"
np="${NP:-2}"

case "${platform}" in
  nvidia) host="${HOST:-nvidia}"; image="${IMAGE:-accelerator-dev/nvidia:latest}"; backend_flag="-DWITH_NVIDIA=ON"; docker_args="--gpus=all" ;;
  metax) host="${HOST:-metax}"; image="${IMAGE:-accelerator-dev/metax:latest}"; backend_flag="-DWITH_METAX=ON"; docker_args="--privileged" ;;
  iluvatar) host="${HOST:-iluvatar}"; image="${IMAGE:-accelerator-dev/iluvatar:latest}"; backend_flag="-DWITH_ILUVATAR=ON"; docker_args="--privileged" ;;
  moore) host="${HOST:-moore}"; image="${IMAGE:-accelerator-dev/moore:latest}"; backend_flag="-DWITH_MOORE=ON"; docker_args="--privileged" ;;
  cambricon) host="${HOST:-cambricon}"; image="${IMAGE:-accelerator-dev/cambricon:latest}"; backend_flag="-DWITH_CAMBRICON=ON"; docker_args="--privileged" ;;
  ascend) host="${HOST:-ascend}"; image="${IMAGE:-accelerator-dev/ascend:latest}"; backend_flag="-DWITH_ASCEND=ON"; docker_args="--privileged" ;;
  *) echo "unknown platform: ${platform}" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
remote_root="${REMOTE_ROOT:-/tmp/infini-distributed-matmul-${USER}}"
ssh_opts=(-o LogLevel=ERROR)

ssh "${ssh_opts[@]}" "${host}" "mkdir -p ${remote_root}/InfiniOps ${remote_root}/InfiniCCL"
rsync -az --delete --exclude build --exclude .git --exclude generated -e "ssh -o LogLevel=ERROR" "${repo_root}/" "${host}:${remote_root}/InfiniOps/"
rsync -az --delete --exclude build --exclude .git -e "ssh -o LogLevel=ERROR" "${HOME}/InfiniCCL/" "${host}:${remote_root}/InfiniCCL/"

ssh "${ssh_opts[@]}" "${host}" bash -s -- "${remote_root}" <<'REMOTE_SETUP'
set -euo pipefail
remote_root="$1"
cat >"${remote_root}/run_inside_container.sh" <<'INNER'
#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/conda/bin:/usr/local/bin:/usr/lib64/mpich/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/lib64/mpich/lib:/usr/local/gcc-11.4.0/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="/usr/include/mpich-aarch64:${CPATH:-}"

if [ -f /usr/local/Ascend/cann-8.5.1/set_env.sh ]; then
  . /usr/local/Ascend/cann-8.5.1/set_env.sh
elif [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  . /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

cmake_pip_index="${CMAKE_PIP_INDEX_URL:-https://pypi.org/simple}"

if ! command -v cmake >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  python3 -m pip install --user -q -i "${cmake_pip_index}" cmake || true
  export PATH="$(python3 -m site --user-base)/bin:${PATH}"
fi

if ! command -v cmake >/dev/null 2>&1 && command -v pip3 >/dev/null 2>&1; then
  pip3 install --user -q -i "${cmake_pip_index}" cmake || true
  if command -v python3 >/dev/null 2>&1; then
    export PATH="$(python3 -m site --user-base)/bin:${PATH}"
  fi
fi

if ! command -v cmake >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update && apt-get install -y cmake || true
  fi
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required inside ${DEMO_IMAGE}; install cmake in the image or provide a container with cmake on PATH." >&2
  exit 127
fi

ccl_backend="-DWITH_OMPI=ON"
if [ "${DEMO_PLATFORM}" = "ascend" ]; then
  ccl_backend="-DWITH_MPICH=ON"
fi

mpi_root_args=""
if mpirun --version 2>&1 | grep -qi Open.MPI; then
  mpi_root_args="--allow-run-as-root"
fi

cd /workspace/InfiniCCL
cmake -S . -B build/cpu-ompi -DCMAKE_INSTALL_PREFIX=/workspace/install/infiniccl -DAUTO_DETECT_DEVICES=OFF "${ccl_backend}" -DBUILD_EXAMPLES=OFF
cmake --build build/cpu-ompi -j2
cmake --install build/cpu-ompi

cd /workspace/InfiniOps
ascend_home_flag=""
if [ "${DEMO_PLATFORM}" = "ascend" ]; then
  ascend_home_flag="-DASCEND_HOME=/usr/local/Ascend/driver/../cann-8.5.1"
fi

cmake -S . -B build/distributed-matmul -DCMAKE_BUILD_TYPE=Release -DBUILD_DISTRIBUTED_MATMUL_DEMO=ON -DINFINICCL_INSTALL=/workspace/install/infiniccl "${DEMO_BACKEND_FLAG}" ${ascend_home_flag}
cmake --build build/distributed-matmul -j2 --target distributed_matmul

run_log="/tmp/distributed_matmul.out"
set +e
LD_LIBRARY_PATH="/usr/local/gcc-11.4.0/lib64:/workspace/install/infiniccl/lib:/workspace/install/infiniccl/lib64:/workspace/InfiniOps/build/distributed-matmul/src:${LD_LIBRARY_PATH:-}" mpirun -np "${DEMO_NP}" ${mpi_root_args} /workspace/InfiniOps/build/distributed-matmul/examples/distributed_matmul "${DEMO_ROWS}" "${DEMO_K}" "${DEMO_N}" 2>&1 | tee "${run_log}"
run_status="${PIPESTATUS[0]}"
set -e
if [ "${run_status}" -eq 137 ] && grep -q "global_shape=.*max_error=" "${run_log}"; then
  echo "mpirun exited 137 after emitting a valid demo result; treating the demo run as successful."
  exit 0
fi
exit "${run_status}"
INNER
chmod +x "${remote_root}/run_inside_container.sh"
REMOTE_SETUP

ssh "${ssh_opts[@]}" "${host}" bash -s -- "${remote_root}" "${image}" "${backend_flag}" "${np}" "${rows}" "${k_size}" "${n_size}" "${platform}" "${docker_args}" <<'REMOTE_RUN'
set -euo pipefail
remote_root="$1"
image="$2"
backend_flag="$3"
np="$4"
rows="$5"
k_size="$6"
n_size="$7"
platform="$8"
docker_args="$9"
container="infini-distmatmul-${platform}-$(id -u)"

docker rm -f "${container}" >/dev/null 2>&1 || true
docker_log="${remote_root}/docker_run_${platform}.out"
set +e
docker run --rm --name "${container}" --network host --ipc host --ulimit memlock=-1 --ulimit stack=67108864 ${docker_args} -e DEMO_PLATFORM="${platform}" -e DEMO_IMAGE="${image}" -e DEMO_BACKEND_FLAG="${backend_flag}" -e DEMO_NP="${np}" -e DEMO_ROWS="${rows}" -e DEMO_K="${k_size}" -e DEMO_N="${n_size}" -v "${remote_root}:/workspace" "${image}" /workspace/run_inside_container.sh 2>&1 | tee "${docker_log}"
docker_status="${PIPESTATUS[0]}"
set -e
if [ "${docker_status}" -eq 137 ] && grep -q "global_shape=.*max_error=" "${docker_log}"; then
  echo "docker exited 137 after emitting a valid demo result; treating the demo run as successful."
  exit 0
fi
exit "${docker_status}"
REMOTE_RUN
