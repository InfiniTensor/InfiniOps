# .ci — CI 镜像与流水线

```
.ci/
├── config.yaml              # 统一配置（镜像、job 定义）
├── build.py                 # 镜像构建
├── run.py                   # CI 流水线执行
└── images/
    ├── nvidia/Dockerfile
    └── ascend/Dockerfile
```

**前置依赖**：Docker、Python 3.10+、`pip install pyyaml`

---

## 配置文件 `config.yaml`

```yaml
repo:
  url: https://github.com/InfiniTensor/InfiniOps.git
  branch: master

images:
  nvidia:
    dockerfile: .ci/images/nvidia/
    build_args:
      BASE_IMAGE: nvcr.io/nvidia/pytorch:24.10-py3

jobs:
  nvidia_gpu:
    image: latest            # latest | <commit-hash>
    platform: nvidia
    resources:
      gpu_ids: "0"           # "0" | "0,2" | "all"
      memory: 32GB
      shm_size: 16g          # 避免 PyTorch SHMEM 不足
      timeout: 3600          # 容器内脚本最大运行秒数
    setup: pip install .[dev]
    env:                     # 可选，注入容器环境变量
      MY_VAR: value
    stages:
      - name: test
        run: pytest tests/ -n auto -v --tb=short --junitxml=/workspace/results/test-results.xml
```

---

## 镜像构建 `build.py`

| 参数 | 说明 |
|---|---|
| `--platform nvidia\|ascend\|all` | 构建平台，默认 `all` |
| `--force` | 跳过 Dockerfile 变更检测 |
| `--dry-run` | 打印命令不执行 |

```bash
# 检测变更后构建（无变更自动跳过）
python .ci/build.py --platform nvidia

# 强制构建
python .ci/build.py --platform nvidia --force
```

构建产物以宿主机本地镜像 tag 存储：`infiniops-ci/<platform>:<commit-hash>` 和 `:latest`。
代理、`no_proxy` 自动从宿主机环境变量透传到 `docker build`。

> `--push` 为预留功能，需在 `config.yaml` 中配置 `registry` 段后方可使用。

---

## 流水线执行 `run.py`

| 参数 | 说明 |
|---|---|
| `--branch` | 覆盖克隆分支 |
| `--stage` | 只运行指定 stage |
| `--image-tag` | 覆盖镜像 tag |
| `--gpu-id` | 覆盖 GPU 设备 ID |
| `--results-dir` | 宿主机目录，挂载到容器 `/workspace/results` |
| `--dry-run` | 打印 docker 命令不执行 |

```bash
# 运行默认 job
python .ci/run.py --branch feat/my-feature --results-dir ./ci-results

# 只跑 test stage，预览命令
python .ci/run.py --stage test --dry-run
```

容器内执行流程：`git clone` → `checkout` → `setup` → stages。
代理从宿主机透传，测试结果写入 `--results-dir`。每次运行均为干净环境（不挂载宿主机 pip 缓存）。
