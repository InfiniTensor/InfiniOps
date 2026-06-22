#!/usr/bin/env bash
set -euo pipefail
platform="${1:?usage: $0 <nvidia|metax|iluvatar|moore|cambricon|ascend>}"
rows="${ROWS:-64}"; k_size="${K:-128}"; n_size="${N:-96}"; np="${NP:-2}"
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
ssh "${host}" "mkdir -p ${remote_root}/InfiniOps ${remote_root}/InfiniCCL"
rsync -az --delete --exclude build --exclude .git --exclude generated "${repo_root}/" "${host}:${remote_root}/InfiniOps/"
rsync -az --delete --exclude build --exclude .git "${HOME}/InfiniCCL/" "${host}:${remote_root}/InfiniCCL/"
ssh "${host}" bash -s -- "${remote_root}" "${image}" "${backend_flag}" "${np}" "${rows}" "${k_size}" "${n_size}" "${platform}" "${docker_args}" <<'REMOTE'
set -euo pipefail
remote_root="$1"; image="$2"; backend_flag="$3"; np="$4"; rows="$5"; k_size="$6"; n_size="$7"; platform="$8"; docker_args="${9}"
container="infini-distmatmul-${platform}-$(id -u)"
docker rm -f "${container}" >/dev/null 2>&1 || true
docker run --rm --name "${container}" --network host --ipc host --ulimit memlock=-1 --ulimit stack=67108864 ${docker_args} -v "${remote_root}:/workspace" "${image}" bash -lc "
set -euo pipefail
export PATH="/usr/lib64/mpich/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/lib64/mpich/lib:/usr/local/gcc-11.4.0/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="/usr/include/mpich-aarch64:${CPATH:-}"
if [ -f /usr/local/Ascend/cann-8.5.1/set_env.sh ]; then
  . /usr/local/Ascend/cann-8.5.1/set_env.sh
elif [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  . /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if ! command -v cmake >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update && apt-get install -y cmake
  else
    python3 -m pip install --user -q cmake
    export PATH="$(python3 -m site --user-base)/bin:${PATH}"
  fi
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required inside ${image}; install cmake in the image or provide a container with cmake on PATH." >&2
  exit 127
fi
ccl_backend="-DWITH_OMPI=ON"
if [ "${platform}" = "ascend" ]; then
  ccl_backend="-DWITH_MPICH=ON"
fi
mpi_root_args=""
if mpirun --version 2>&1 | grep -qi Open.MPI; then
  mpi_root_args="--allow-run-as-root"
fi
cd /workspace/InfiniCCL
cmake -S . -B build/cpu-ompi -DCMAKE_INSTALL_PREFIX=/workspace/install/infiniccl -DAUTO_DETECT_DEVICES=OFF \${ccl_backend} -DBUILD_EXAMPLES=OFF
cmake --build build/cpu-ompi -j2
cmake --install build/cpu-ompi
cd /workspace/InfiniOps
ascend_home_flag=
if [ ${platform} = ascend ]; then
  ascend_home_flag=-DASCEND_HOME=/usr/local/Ascend/driver/../cann-8.5.1
fi
cmake -S . -B build/distributed-matmul -DCMAKE_BUILD_TYPE=Release -DBUILD_DISTRIBUTED_MATMUL_DEMO=ON -DINFINICCL_INSTALL=/workspace/install/infiniccl ${backend_flag} \${ascend_home_flag}
cmake --build build/distributed-matmul -j2 --target distributed_matmul
LD_LIBRARY_PATH=/usr/local/gcc-11.4.0/lib64:/workspace/install/infiniccl/lib:/workspace/install/infiniccl/lib64:/workspace/InfiniOps/build/distributed-matmul/src mpirun -np ${np} \${mpi_root_args} /workspace/InfiniOps/build/distributed-matmul/examples/distributed_matmul ${rows} ${k_size} ${n_size}
"
REMOTE
