# .ci — CI 镜像与流水线

```
.ci/
├── config.yaml              # 统一配置（镜像、job、Agent 定义）
├── utils.py                 # 共享工具（load_config、normalize_config、get_git_commit）
├── agent.py                 # Runner Agent（调度、Webhook、远程触发）
├── build.py                 # 镜像构建
├── run.py                   # CI 流水线执行（Docker 层）
├── ci_resource.py           # GPU/内存资源检测与分配
├── github_status.py         # GitHub Commit Status 上报
├── images/
│   ├── nvidia/Dockerfile
│   ├── iluvatar/Dockerfile
│   └── ascend/Dockerfile
└── tests/                   # 单元测试
    ├── conftest.py
    ├── test_agent.py
    ├── test_build.py
    ├── test_run.py
    ├── test_resource.py
    ├── test_github_status.py
    └── test_utils.py
```

**前置依赖**：Docker、Python 3.10+、`pip install pyyaml`

---

## 配置文件 `config.yaml`

配置以 **platform** 为顶级结构，每个平台包含镜像定义、平台级默认值和 job 列表。
加载时自动展平为 `{platform}_{job}` 格式（如 `nvidia_gpu`）。

```yaml
repo:
  url: https://github.com/InfiniTensor/InfiniOps.git
  branch: master

github:
  status_context_prefix: "ci/infiniops"

agents:                                  # 远程 Agent 地址（CLI 跨机器触发用）
  nvidia:
    url: http://nvidia-host:8080
  iluvatar:
    url: http://iluvatar-host:8080

platforms:
  nvidia:
    image:                              # 镜像定义
      dockerfile: .ci/images/nvidia/
      build_args:
        BASE_IMAGE: nvcr.io/nvidia/pytorch:24.10-py3
    setup: pip install .[dev] --no-build-isolation
    jobs:
      gpu:                              # 展平后为 nvidia_gpu
        resources:
          gpu_ids: "0"                  # "0" | "0,2" | "all"
          memory: 32GB
          shm_size: 16g
          timeout: 3600
        stages:
          - name: test
            run: pytest tests/ -n 8 -v --tb=short --junitxml=/workspace/results/test-results.xml

  iluvatar:
    image:
      dockerfile: .ci/images/iluvatar/
      build_args:
        BASE_IMAGE: corex:qs_pj20250825
        APT_MIRROR: http://archive.ubuntu.com/ubuntu
        PIP_INDEX_URL: https://pypi.org/simple
    docker_args:                        # 平台级 docker 参数，所有 job 继承
      - "--privileged"
      - "--cap-add=ALL"
      - "--pid=host"
      - "--ipc=host"
    volumes:
      - /dev:/dev
      - /lib/firmware:/lib/firmware
      - /usr/src:/usr/src
      - /lib/modules:/lib/modules
    setup: pip install .[dev] --no-build-isolation
    jobs:
      gpu:                              # 展平后为 iluvatar_gpu
        resources:
          gpu_ids: "0"
          gpu_style: none               # CoreX 设备通过 --privileged + /dev 挂载
          memory: 32GB
          shm_size: 16g
          timeout: 3600
        stages:
          - name: test
            run: pytest tests/ -n 8 -v --tb=short --junitxml=/workspace/results/test-results.xml
```

### 配置层级说明

| 层级 | 字段 | 说明 |
|---|---|---|
| **平台级** | `image` | 镜像定义（dockerfile、build_args） |
| | `image_tag` | 默认镜像 tag（默认 `latest`） |
| | `docker_args` | 额外 docker run 参数（如 `--privileged`） |
| | `volumes` | 额外挂载卷 |
| | `setup` | 容器内 setup 命令 |
| | `env` | 注入容器环境变量 |
| **Job 级** | `resources.gpu_ids` | GPU 设备 ID |
| | `resources.gpu_style` | GPU 透传方式：`nvidia`（默认）或 `none` |
| | `resources.memory` | 容器内存限制 |
| | `resources.shm_size` | 共享内存大小 |
| | `resources.timeout` | 容器内脚本最大运行秒数 |
| | `stages` | 执行阶段列表 |
| | 以上平台级字段 | Job 可覆盖任意平台级默认值 |

---

## 镜像构建 `build.py`

| 参数 | 说明 |
|---|---|
| `--platform nvidia\|iluvatar\|ascend\|all` | 构建平台，默认 `all` |
| `--commit` | 指定 commit ref 作为镜像 tag（默认 HEAD） |
| `--force` | 跳过 Dockerfile 变更检测 |
| `--dry-run` | 打印命令不执行 |

```bash
# 检测变更后构建（无变更自动跳过）
python .ci/build.py --platform nvidia

# 构建 Iluvatar 镜像
python .ci/build.py --platform iluvatar --force

# 强制构建全部
python .ci/build.py --force
```

构建产物以宿主机本地镜像 tag 存储：`infiniops-ci/<platform>:<commit-hash>` 和 `:latest`。
代理、`no_proxy` 自动从宿主机环境变量透传到 `docker build`。

> `--push` 为预留功能，需在 `config.yaml` 中配置 `registry` 段后方可使用。

---

## 流水线执行 `run.py`

平台自动发现（通过检测 `nvidia-smi`/`ixsmi`），无需手动指定。

| 参数 | 说明 |
|---|---|
| `--config` | 配置文件路径（默认 `.ci/config.yaml`） |
| `--job` | job 名称：短名（`gpu`）或完整名（`nvidia_gpu`）。缺省运行当前平台所有 job |
| `--branch` | 覆盖克隆分支（默认读 config `repo.branch`） |
| `--stage` | 只运行指定 stage |
| `--image-tag` | 覆盖镜像 tag |
| `--gpu-id` | 覆盖 GPU 设备 ID（nvidia 通过 `--gpus`，其他平台通过 `CUDA_VISIBLE_DEVICES`） |
| `--results-dir` | 宿主机目录，挂载到容器 `/workspace/results` |
| `--dry-run` | 打印 docker 命令不执行 |

```bash
# 最简用法：自动检测平台，运行所有 job，使用 config 默认分支
python .ci/run.py

# 指定 job 短名
python .ci/run.py --job gpu

# 完整 job 名（向后兼容）
python .ci/run.py --job nvidia_gpu

# 只跑 test stage，预览命令
python .ci/run.py --job gpu --stage test --dry-run
```

容器内执行流程：`git clone` → `checkout` → `setup` → stages。
代理从宿主机透传，测试结果写入 `--results-dir`。每次运行均为干净环境（不挂载宿主机 pip 缓存）。

---

## 平台差异

| 平台 | GPU 透传方式 | 基础镜像 | 备注 |
|---|---|---|---|
| NVIDIA | `--gpus` (NVIDIA Container Toolkit) | `nvcr.io/nvidia/pytorch:24.10-py3` | 标准 CUDA |
| Iluvatar | `--privileged` + `/dev` 挂载 | `corex:qs_pj20250825` | CoreX 运行时，CUDA 兼容 |
| Ascend | TODO | `ascend-pytorch:24.0.0` | 待完善，镜像和 job 尚未就绪 |

---

## Runner Agent `agent.py`

Runner Agent 支持 CLI 手动触发、GitHub Webhook 自动触发、资源感知的动态调度，以及跨机器远程触发。

### CLI 手动执行

```bash
# 运行所有 job（分发到远程 Agent，使用 config 默认分支）
python .ci/agent.py run

# 指定分支
python .ci/agent.py run --branch feat/xxx

# 运行指定 job
python .ci/agent.py run --job nvidia_gpu

# 按平台运行
python .ci/agent.py run --platform nvidia

# 预览命令
python .ci/agent.py run --dry-run
```

| 参数 | 说明 |
|---|---|
| `--branch` | 测试分支（默认读 config `repo.branch`） |
| `--job` | 指定 job 名称 |
| `--platform` | 按平台过滤 job |
| `--commit` | 覆盖 commit SHA |
| `--image-tag` | 覆盖镜像 tag |
| `--dry-run` | 预览模式 |

### Webhook 服务

每台平台机器部署一个 Agent 实例（平台自动发现）：

```bash
# NVIDIA 机器
python .ci/agent.py serve --port 8080

# Iluvatar 机器
python .ci/agent.py serve --port 8080
```

`serve` 子命令额外参数：

| 参数 | 说明 |
|---|---|
| `--port` | 监听端口（默认 8080） |
| `--host` | 监听地址（默认 `0.0.0.0`） |
| `--webhook-secret` | GitHub Webhook 签名密钥（或 `WEBHOOK_SECRET` 环境变量） |
| `--api-token` | `/api/run` Bearer 认证令牌（或 `AGENT_API_TOKEN` 环境变量） |
| `--results-dir` | 结果目录（默认 `ci-results`） |
| `--utilization-threshold` | GPU 空闲阈值百分比（默认 10） |

| 端点 | 方法 | 说明 |
|---|---|---|
| `/webhook` | POST | GitHub Webhook（push/pull_request） |
| `/api/run` | POST | 远程触发 job |
| `/api/job/{id}` | GET | 查询 job 状态 |
| `/health` | GET | 健康检查 |
| `/status` | GET | 队列 + 资源状态 |

Webhook 支持 `X-Hub-Signature-256` 签名验证，通过 `--webhook-secret` 或 `WEBHOOK_SECRET` 环境变量配置。

### 远程 Agent 配置

在 `config.yaml` 中配置各平台 Agent 地址，CLI 执行时自动将远程 job 分发到对应 Agent：

```yaml
agents:
  nvidia:
    url: http://nvidia-host:8080
  iluvatar:
    url: http://iluvatar-host:8080
```

### 资源调度

Agent 自动检测 GPU 利用率和系统内存，动态决定并行度：
- GPU 利用率 < 阈值（默认 10%）且未被 Agent 分配 → 可用
- 资源不足时 job 自动排队，已完成 job 释放资源后自动调度排队任务

### GitHub Status

设置 `GITHUB_TOKEN` 环境变量后，Agent 会自动上报 commit status：
- `pending` — job 开始执行
- `success` / `failure` — job 执行完成

Status context 格式：`ci/infiniops/{job_name}`

---

## 多机部署指南

以 NVIDIA + Iluvatar 双平台为例，说明如何在两台机器上部署 Agent 并实现跨平台并行测试。

### 前置条件（两台机器共同）

```bash
# 1. Python 3.10+ 和依赖
pip install pyyaml

# 2. Docker 已安装
docker --version

# 3. 克隆仓库
git clone https://github.com/InfiniTensor/InfiniOps.git
cd InfiniOps
```

### NVIDIA 机器配置

```bash
# 1. 安装 NVIDIA Container Toolkit
#    参考: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# 2. 验证 GPU 可见
nvidia-smi

# 3. 构建 CI 镜像
python .ci/build.py --platform nvidia
```

### Iluvatar 机器配置

```bash
# 1. 确认 CoreX 运行时已安装
ixsmi

# 2. 确认基础镜像已导入（非公开镜像，需提前准备）
docker images | grep corex    # 应有 corex:qs_pj20250825

# 3. 构建 CI 镜像
python .ci/build.py --platform iluvatar
```

### 启动 Agent 服务

在各自机器上启动 Agent：

```bash
# NVIDIA 机器（平台自动发现）
python .ci/agent.py serve --port 8080

# Iluvatar 机器（平台自动发现）
python .ci/agent.py serve --port 8080
```

验证连通性：

```bash
curl http://<nvidia-ip>:8080/health
curl http://<iluvatar-ip>:8080/health
```

### 配置远程 Agent 地址

在触发端的 `config.yaml` 中添加 `agents` 段：

```yaml
agents:
  nvidia:
    url: http://<nvidia-ip>:8080
  iluvatar:
    url: http://<iluvatar-ip>:8080
```

### 触发跨平台测试

```bash
# 一键运行所有平台的 job（使用 config 默认分支）
python .ci/agent.py run

# 预览模式（不实际执行）
python .ci/agent.py run --dry-run

# 只运行指定平台
python .ci/agent.py run --platform nvidia
```

### 可选配置

#### GitHub Status 上报

两台机器均设置环境变量，各自上报所属平台的测试状态：

```bash
export GITHUB_TOKEN=ghp_xxxxxxxxxxxx
```

#### API Token 认证

Agent 暴露在非可信网络时，建议启用 Token 认证：

```bash
# 启动 Agent 时指定 token
python .ci/agent.py serve --port 8080 --api-token <secret>

# 或通过环境变量
export AGENT_API_TOKEN=<secret>
```

#### GitHub Webhook 自动触发

在 GitHub repo → Settings → Webhooks 中为每台机器添加 Webhook：

| 字段 | 值 |
|---|---|
| Payload URL | `http://<机器IP>:8080/webhook` |
| Content type | `application/json` |
| Secret | 与 `--webhook-secret` 一致 |
| Events | `push` 和 `pull_request` |

启动时配置 secret：

```bash
python .ci/agent.py serve --port 8080 --webhook-secret <github-secret>

# 或通过环境变量
export WEBHOOK_SECRET=<github-secret>
```

### 验证清单

```bash
# 1. 各机器单独 dry-run
python .ci/agent.py run --platform nvidia --dry-run
python .ci/agent.py run --platform iluvatar --dry-run

# 2. 健康检查
curl http://<nvidia-ip>:8080/health
curl http://<iluvatar-ip>:8080/health

# 3. 查看资源状态
curl http://<nvidia-ip>:8080/status
curl http://<iluvatar-ip>:8080/status

# 4. 跨平台一键测试
python .ci/agent.py run --branch master
```
