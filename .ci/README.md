# .ci — CI 镜像与流水线

本目录管理 CI 所用的 Docker 镜像构建与测试流水线执行。

## 目录结构

```
.ci/
├── config.yaml              # 统一配置（registry、镜像、job 定义）
├── build.py                 # 镜像构建脚本
├── run.py                   # CI 流水线执行脚本
├── README.md
└── images/
    ├── nvidia/Dockerfile    # NVIDIA 平台镜像
    └── ascend/Dockerfile    # 昇腾平台镜像
```

## 前置依赖

- Docker
- Python 3.10+
- pyyaml (`pip install pyyaml`)

## 配置文件 `config.yaml`

```yaml
repo:
  url: https://github.com/InfiniTensor/InfiniOps.git
  branch: master

registry:
  url: ""                    # Harbor 地址，本地开发时留空
  project: infiniops
  credentials_env: REGISTRY_TOKEN

images:
  nvidia:
    dockerfile: .ci/images/nvidia/
    build_args:
      BASE_IMAGE: nvcr.io/nvidia/pytorch:24.10-py3
  ascend:
    dockerfile: .ci/images/ascend/
    build_args:
      BASE_IMAGE: ascendhub.huawei.com/public-ascendhub/ascend-pytorch:24.0.0
    private_sdk:
      source: "${PRIVATE_SDK_URL}"

jobs:
  nvidia_gpu:
    image: stable            # stable | latest | 具体 commit hash
    platform: nvidia
    resources:
      gpu_ids: "0"           # GPU 设备 ID，如 "0" "0,2" "all"
      gpu_type: A100
      memory: 32GB
      timeout: 3600
    setup: pip install .[dev]
    stages:
      - name: test
        run: pytest tests/ -v --tb=short --junitxml=/workspace/test-results.xml
```

- **`registry.url`** 为空时镜像仅保存在本地，tag 格式为 `<project>-ci/<platform>:<tag>`。
- **`images.<platform>.build_args`** 会作为 `--build-arg` 传入 `docker build`。
- **`jobs.<name>.image`** 支持 `stable`、`latest` 或具体 commit hash。
- **`resources.gpu_ids`** 指定 GPU 设备 ID，支持 `"0"`、`"0,2"`、`"all"` 等格式，映射为 `docker run --gpus "device=..."`。也可保留 `gpu_count` 按数量分配。

## 镜像构建 `build.py`

```bash
python .ci/build.py [options]
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--platform` | `all` | 构建平台：`nvidia`、`ascend` 或 `all` |
| `--commit` | `HEAD` | 用于镜像 tag 的 git ref |
| `--push` | — | 构建后推送到 registry |
| `--force` | — | 跳过变更检测，强制构建 |
| `--dry-run` | — | 仅打印命令，不执行 |
| `--config` | `.ci/config.yaml` | 配置文件路径 |

### 示例

```bash
# 构建 nvidia 镜像（自动检测 Dockerfile 变更，无变更则跳过）
python .ci/build.py --platform nvidia

# 强制构建
python .ci/build.py --platform nvidia --force

# 构建全部平台并推送到 registry
python .ci/build.py --push --force

# 预览实际执行的 docker 命令
python .ci/build.py --platform nvidia --force --dry-run
```

### 构建流程

1. 通过 `git diff HEAD~1` 检测 Dockerfile 目录是否有变更（`--force` 跳过此步）
2. `docker build` 构建镜像，同时打 `<commit-hash>` 和 `latest` 两个 tag
3. 自动透传宿主机的 `http_proxy`/`https_proxy`/`no_proxy` 到构建容器
4. 若指定 `--push`，将两个 tag 推送到 registry

### 产物

| Tag | 说明 |
|---|---|
| `infiniops-ci/<platform>:<commit-hash>` | 精确追溯到某次构建 |
| `infiniops-ci/<platform>:latest` | 最近一次构建 |

## 流水线执行 `run.py`

```bash
python .ci/run.py [options]
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--job` | 配置中第一个 job | 要执行的 job 名称 |
| `--branch` | `config.yaml` 中的 `repo.branch` | 覆盖克隆分支 |
| `--stage` | 全部 | 仅运行指定 stage |
| `--image-tag` | job 中的 `image` 字段 | 覆盖镜像版本 |
| `--gpu-id` | config 中的 `gpu_ids` | GPU 设备 ID，如 `0`、`0,2`、`all` |
| `--dry-run` | — | 仅打印 docker 命令，不执行 |
| `--config` | `.ci/config.yaml` | 配置文件路径 |

### 示例

```bash
# 运行默认 job
python .ci/run.py

# 指定分支和镜像版本
python .ci/run.py --branch feature-xxx --image-tag latest

# 只用 GPU 0 运行
python .ci/run.py --gpu-id 0

# 用 GPU 0 和 2 运行
python .ci/run.py --gpu-id 0,2

# 使用全部 GPU
python .ci/run.py --gpu-id all

# 只跑 test stage
python .ci/run.py --stage test

# 预览 docker 命令
python .ci/run.py --dry-run
```

### 执行流程

1. 解析 job 配置，拉取对应镜像
2. `docker run` 启动容器（自动挂载 GPU、限制内存）
3. 容器内 `git clone` → `checkout` → 执行 `setup` 命令
4. 依次执行各 stage，汇总结果

## 代理配置

如果网络环境需要代理，在宿主机设置环境变量后即可：

```bash
export http_proxy=http://localhost:9991
export https_proxy=http://localhost:9991
```

- **`build.py`** 会自动透传代理到 `docker build`（通过 `--build-arg` + `--network host`）。
- **`run.py`** 使用 `--network host`，容器内可直接访问宿主机代理。
