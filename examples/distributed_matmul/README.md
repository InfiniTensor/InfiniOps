# Distributed Matmul Demo

This demo uses InfiniOps for per-rank GEMM on the local accelerator and InfiniCCL/OpenMPI for process bootstrap plus AllGather of host result tiles. InfiniCCL is built as CPU/OMPI on purpose so the same demo also works on Ascend, where this InfiniCCL checkout does not provide an Ascend device backend.

Build InfiniCCL inside the accelerator container. If the image does not ship cmake, install it in the temporary container first, for example apt-get update && apt-get install -y cmake, or python3 -m pip install --user cmake and add $(python3 -m site --user-base)/bin to PATH.

Build InfiniCCL inside the accelerator container:

    cd /workspace/InfiniCCL
    cmake -S . -B build/cpu-ompi \
      -DCMAKE_INSTALL_PREFIX=/workspace/install/infiniccl \
      -DAUTO_DETECT_DEVICES=OFF \
      -DWITH_OMPI=ON \
      -DBUILD_EXAMPLES=OFF
    cmake --build build/cpu-ompi -j$(nproc)
    cmake --install build/cpu-ompi

Build InfiniOps and this demo for the current accelerator:

    cd /workspace/InfiniOps
    cmake -S . -B build/distributed-matmul \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_DISTRIBUTED_MATMUL_DEMO=ON \
      -DINFINICCL_INSTALL=/workspace/install/infiniccl \
      -DWITH_NVIDIA=ON
    cmake --build build/distributed-matmul -j$(nproc) --target distributed_matmul

Replace WITH_NVIDIA with WITH_METAX, WITH_ILUVATAR, WITH_MOORE, WITH_CAMBRICON, or WITH_ASCEND. On the current Ascend image, pass -DASCEND_HOME=/usr/local/Ascend/driver/../cann-8.5.1 so InfiniOps links the real driver instead of the CANN devlib stub.

Run two local ranks:

    LD_LIBRARY_PATH=/usr/local/gcc-11.4.0/lib64:/workspace/install/infiniccl/lib:/workspace/install/infiniccl/lib64:/workspace/InfiniOps/build/distributed-matmul/src \
    mpirun -np 2 --allow-run-as-root \
      /workspace/InfiniOps/build/distributed-matmul/examples/distributed_matmul 64 128 96

Arguments are: rows_per_rank k n. The global m dimension is rows_per_rank * world_size.

Remote helper examples:

    ./examples/distributed_matmul/run_remote.sh nvidia
    ./examples/distributed_matmul/run_remote.sh metax
    ./examples/distributed_matmul/run_remote.sh iluvatar
    ./examples/distributed_matmul/run_remote.sh moore
    ./examples/distributed_matmul/run_remote.sh cambricon
    ./examples/distributed_matmul/run_remote.sh ascend

Override size or rank count:

    NP=4 ROWS=128 K=256 N=192 ./examples/distributed_matmul/run_remote.sh nvidia
