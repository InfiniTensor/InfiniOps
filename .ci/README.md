# .ci — CI 镜像与流水线

```
.ci/
├── config.yaml              # 统一配置（镜像、job、Agent 定义）
├── utils.py                 # 共享工具（load_config、get_git_commit）
├── agent.py                 # Runner Agent（调度、Webhook、远程触发）
├── build.py                 # 镜像构建
├── run.py                   # CI 流水线执行（Docker 层）
├── ci_resource.py           # GPU/内存资源检测与分配
├── github_status.py         # GitHub Commit Status 上报
└── images/
    ├── nvidia/Dockerfile
    ├── iluvatar/Dockerfile
    └── ascend/Dockerfile
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

platforms:
  nvidia:
    image:                              # 镜像定义
      dockerfile: .ci/images/nvidia/
      build_args:
        BASE_IMAGE: nvcr.io/nvidia/pytorch:24.10-py3
    setup: pip install .[dev]           # 平台级默认值，job 可覆盖
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
    setup: pip install .[dev]
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

| 参数 | 说明 |
|---|---|
| `--job` | 指定 job 名称（默认第一个） |
| `--branch` | 覆盖克隆分支 |
| `--stage` | 只运行指定 stage |
| `--image-tag` | 覆盖镜像 tag |
| `--gpu-id` | 覆盖 GPU 设备 ID（仅 nvidia gpu_style） |
| `--results-dir` | 宿主机目录，挂载到容器 `/workspace/results` |
| `--dry-run` | 打印 docker 命令不执行 |

```bash
# 运行 NVIDIA job
python .ci/run.py --job nvidia_gpu --branch master

# 运行 Iluvatar job
python .ci/run.py --job iluvatar_gpu --branch feat/ci-nvidia

# 只跑 test stage，预览命令
python .ci/run.py --job iluvatar_gpu --stage test --dry-run
```

容器内执行流程：`git clone` → `checkout` → `setup` → stages。
代理从宿主机透传，测试结果写入 `--results-dir`。每次运行均为干净环境（不挂载宿主机 pip 缓存）。

---

## 平台差异

| 平台 | GPU 透传方式 | 基础镜像 | 备注 |
|---|---|---|---|
| NVIDIA | `--gpus` (NVIDIA Container Toolkit) | `nvcr.io/nvidia/pytorch:24.10-py3` | 标准 CUDA |
| Iluvatar | `--privileged` + `/dev` 挂载 | `corex:qs_pj20250825` | CoreX 运行时，CUDA 兼容 |
| Ascend | TODO | `ascend-pytorch:24.0.0` | 待完善 |

---

## Runner Agent `agent.py`

Runner Agent 支持 CLI 手动触发、GitHub Webhook 自动触发、资源感知的动态调度，以及跨机器远程触发。

### CLI 手动执行

```bash
# 运行所有 job（本地 + 远程 Agent）
python .ci/agent.py run --branch master

# 运行指定 job
python .ci/agent.py run --branch master --job nvidia_gpu

# 按平台运行
python .ci/agent.py run --branch master --platform nvidia

# 预览命令
python .ci/agent.py run --branch master --dry-run --no-status
```

| 参数 | 说明 |
|---|---|
| `--branch` | 测试分支（必填） |
| `--job` | 指定 job 名称 |
| `--platform` | 按平台过滤 job |
| `--commit` | 覆盖 commit SHA |
| `--image-tag` | 覆盖镜像 tag |
| `--results-dir` | 结果目录（默认 `ci-results`） |
| `--utilization-threshold` | GPU 空闲阈值百分比（默认 10） |
| `--no-status` | 跳过 GitHub Status 上报 |
| `--dry-run` | 预览模式 |

### Webhook 服务

每台平台机器部署一个 Agent 实例：

```bash
# NVIDIA 机器
python .ci/agent.py serve --platform nvidia --port 8080

# Iluvatar 机器
python .ci/agent.py serve --platform iluvatar --port 8080
```

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
