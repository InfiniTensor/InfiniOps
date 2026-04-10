# .ci — CI Images and Pipeline

```
.ci/
├── config.yaml              # Unified config (images, jobs, agent definitions)
├── utils.py                 # Shared utilities (load_config, normalize_config, get_git_commit)
├── agent.py                 # Runner Agent (scheduler, webhooks, remote dispatch)
├── build.py                 # Image builder
├── run.py                   # CI pipeline runner (Docker layer)
├── ci_resource.py           # GPU/memory detection and allocation
├── github_status.py         # GitHub Commit Status reporting
├── images/
│   ├── nvidia/Dockerfile
│   ├── iluvatar/Dockerfile
│   ├── metax/Dockerfile
│   ├── moore/Dockerfile
│   ├── cambricon/Dockerfile
│   └── ascend/Dockerfile
└── tests/                   # Unit tests
    ├── conftest.py
    ├── test_agent.py
    ├── test_build.py
    ├── test_run.py
    ├── test_resource.py
    ├── test_github_status.py
    └── test_utils.py
```

**Prerequisites**: Docker, Python 3.10+, `pip install pyyaml`

---

## Configuration `config.yaml`

Config uses a **platform-centric** top-level structure. Each platform defines its image, platform-level defaults, and job list.
At load time, jobs are flattened to `{platform}_{job}` format (e.g., `nvidia_gpu`).

```yaml
repo:
  url: https://github.com/InfiniTensor/InfiniOps.git
  branch: master

github:
  status_context_prefix: "ci/infiniops"

agents:                                  # Remote agent URLs (used by CLI for cross-machine dispatch)
  nvidia:
    url: http://nvidia-host:8080
  iluvatar:
    url: http://iluvatar-host:8080

platforms:
  nvidia:
    image:                              # Image definition
      dockerfile: .ci/images/nvidia/
      build_args:
        BASE_IMAGE: nvcr.io/nvidia/pytorch:25.12-py3
    setup: pip install .[dev] --no-build-isolation
    jobs:
      gpu:                              # Flattened as `nvidia_gpu`.
        resources:
          ngpus: 1                      # Scheduler auto-picks this many free GPUs.
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
    docker_args:                        # Platform-level docker args, inherited by all jobs
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
      gpu:                              # Flattened as `iluvatar_gpu`.
        resources:
          ngpus: 1
          gpu_ids: auto
          memory: 32GB
          shm_size: 16g
          timeout: 3600
        stages:
          - name: test
            run: pytest tests/ -n 8 -v --tb=short --junitxml=/workspace/results/test-results.xml
```

### Config hierarchy

| Level | Field | Description |
|---|---|---|
| **Platform** | `image` | Image definition (dockerfile, build_args) |
| | `image_tag` | Default image tag (defaults to `latest`) |
| | `docker_args` | Extra `docker run` args (e.g., `--privileged`) |
| | `volumes` | Extra volume mounts |
| | `setup` | In-container setup command |
| | `env` | Injected container env vars |
| **Job** | `resources.ngpus` | Number of GPUs to allocate (default: 1). Used with `gpu_ids: auto` for dynamic allocation |
| | `resources.gpu_ids` | `auto`: scheduler picks `ngpus` least-loaded GPUs. Static: pin to specific IDs (e.g., `"0"`, `"0,2"`). `all`: use all GPUs |
| | `resources.memory` | Container memory limit |
| | `resources.shm_size` | Shared memory size |
| | `resources.timeout` | Max run time in seconds |
| | `stages` | Execution stage list |
| | Any platform field | Jobs can override any platform-level default |

---

## Image builder `build.py`

| Flag | Description |
|---|---|
| `--platform nvidia\|iluvatar\|metax\|moore\|ascend\|all` | Target platform (default: `all`) |
| `--commit` | Use specific commit ref as image tag (default: HEAD) |
| `--force` | Skip Dockerfile change detection |
| `--dry-run` | Print commands without executing |

```bash
# Build with change detection (skips if no Dockerfile changes)
python .ci/build.py --platform nvidia

# Build Iluvatar image
python .ci/build.py --platform iluvatar --force

# Force build all platforms
python .ci/build.py --force
```

Build artifacts are stored as local Docker image tags: `infiniops-ci/<platform>:<commit-hash>` and `:latest`.
Proxy and `no_proxy` env vars are forwarded from the host to `docker build` automatically.

> `--push` is reserved for future use; requires a `registry` section in `config.yaml`.

---

## Pipeline runner `run.py`

Platform is auto-detected (via `nvidia-smi`/`ixsmi`/`mx-smi`/`mthreads-gmi`/`cnmon`/`npu-smi` on PATH), no manual specification needed.

| Flag | Description |
|---|---|
| `--config` | Config file path (default: `.ci/config.yaml`) |
| `--job` | Job name (e.g., `nvidia_gpu`, `ascend_npu`). Defaults to all jobs for the current platform |
| `--branch` | Override clone branch (default: config `repo.branch`) |
| `--stage` | Run only the specified stage |
| `--image-tag` | Override image tag |
| `--gpu-id` | Override GPU device IDs (nvidia via `--gpus`, others via platform-specific env var) |
| `--test` | Override pytest test path (e.g., `tests/test_gemm.py::test_gemm`) |
| `--results-dir` | Host directory mounted to `/workspace/results` inside the container |
| `--local` | Mount current directory (read-only) instead of cloning from git |
| `--dry-run` | Print docker command without executing |

```bash
# Simplest usage: auto-detect platform, run all jobs, use config default branch
python .ci/run.py

# Run a specific job
python .ci/run.py --job nvidia_gpu

# Run only the test stage, preview mode
python .ci/run.py --job nvidia_gpu --stage test --dry-run

# Test local uncommitted changes without pushing
python .ci/run.py --local
```

Container execution flow: `git clone` → `checkout` → `setup` → stages (fail-fast: first failure breaks the loop and preserves the real exit code).
With `--local`, the current directory is mounted read-only at `/workspace/repo` and copied to a writable temp directory inside the container before setup runs — host files are never modified.
Proxy vars are forwarded from the host. Test results are written to `--results-dir` (each run gets a unique directory with timestamp + UUID suffix). Each run uses a clean environment (no host pip cache mounted).

---

## Platform differences

| Platform | GPU passthrough | Device env var | Base image | Detection tool |
|---|---|---|---|---|
| NVIDIA | `--gpus` (NVIDIA Container Toolkit) | — (uses Docker flag) | `nvcr.io/nvidia/pytorch:25.12-py3` | `nvidia-smi` |
| Iluvatar | `--privileged` + `/dev` mount | `CUDA_VISIBLE_DEVICES` | `corex:qs_pj20250825` | `ixsmi` |
| MetaX | `--privileged` | `CUDA_VISIBLE_DEVICES` | `maca-pytorch:3.2.1.4-...` | `mx-smi` |
| Moore | `--privileged` | `MTHREADS_VISIBLE_DEVICES` | `vllm_musa:20251112_hygon` | `mthreads-gmi` |
| Cambricon | `--privileged` | `MLU_VISIBLE_DEVICES` | `cambricon/pytorch:v1.25.3` | `cnmon` |
| Ascend | `--privileged` + device mounts | `ASCEND_VISIBLE_DEVICES` | `vllm-ascend:v0.18.0rc1-openeuler` | `npu-smi` |

Device visibility is derived from the platform name (see `PLATFORM_DEVICE_ENV` in `ci_resource.py`). NVIDIA uses Docker's `--gpus` flag; all other platforms use `--privileged` and control visibility via a platform-specific environment variable.

---

## Runner Agent `agent.py`

The Runner Agent supports CLI manual dispatch, GitHub webhook triggers, resource-aware dynamic scheduling, and cross-machine remote dispatch.

### CLI manual execution

```bash
# Run all jobs (dispatched to remote agents, using config default branch)
python .ci/agent.py run

# Specify branch
python .ci/agent.py run --branch feat/xxx

# Run a specific job
python .ci/agent.py run --job nvidia_gpu

# Filter by platform
python .ci/agent.py run --platform nvidia

# Preview mode
python .ci/agent.py run --dry-run
```

| Flag | Description |
|---|---|
| `--branch` | Test branch (default: config `repo.branch`) |
| `--job` | Specific job name |
| `--platform` | Filter jobs by platform |
| `--commit` | Override commit SHA used for GitHub status reporting |
| `--image-tag` | Override image tag |
| `--dry-run` | Preview mode |

### Webhook server

Deploy one Agent instance per platform machine (platform is auto-detected). On each machine:

```bash
python .ci/agent.py serve --port 8080
```

Additional `serve` flags:

| Flag | Description |
|---|---|
| `--port` | Listen port (default: 8080) |
| `--host` | Listen address (default: `0.0.0.0`) |
| `--webhook-secret` | GitHub webhook signing secret (or `WEBHOOK_SECRET` env var) |
| `--api-token` | `/api/run` Bearer auth token (or `AGENT_API_TOKEN` env var) |
| `--results-dir` | Results directory (default: `ci-results`) |
| `--utilization-threshold` | GPU idle threshold percentage (default: 10) |

| Endpoint | Method | Description |
|---|---|---|
| `/webhook` | POST | GitHub webhook (push/pull_request) |
| `/api/run` | POST | Remote job trigger |
| `/api/job/{id}` | GET | Query job status |
| `/api/job/{id}/log` | GET | Full job log (text/plain) |
| `/health` | GET | Health check |
| `/status` | GET | Queue + resource status |

Webhook supports `X-Hub-Signature-256` signature verification via `--webhook-secret` or `WEBHOOK_SECRET` env var.

### Remote agent configuration

Configure agent URLs in `config.yaml`; the CLI automatically dispatches remote jobs to the corresponding agents:

```yaml
agents:
  nvidia:
    url: http://<nvidia-ip>:8080
  iluvatar:
    url: http://<iluvatar-ip>:8080
  metax:
    url: http://<metax-ip>:8080
  moore:
    url: http://<moore-ip>:8080
```

### Resource scheduling

The Agent auto-detects GPU utilization and system memory to dynamically determine parallelism:
- GPUs with utilization < threshold (default 10%) and not already allocated → available
- Allocation picks the **least-loaded** GPUs first (sorted by utilization ascending)
- When `gpu_ids: auto` (default), the scheduler allocates `ngpus` GPUs per job
- When resources are insufficient, jobs are queued automatically (max 100 pending); completed jobs release resources and trigger scheduling of queued tasks
- Docker execution has a Python-level timeout fallback (job timeout + 120s) to prevent stuck containers

### GitHub Status

Set the `GITHUB_TOKEN` env var and the Agent will automatically report commit status:
- `pending` — job started
- `success` / `failure` — job completed

Status context format: `ci/infiniops/{job_name}`

---

## Multi-machine deployment guide

### Per-platform setup

Each machine needs Docker installed, the platform runtime, and the base CI image built.

| Platform | Runtime check | Base image | Build command |
|---|---|---|---|
| NVIDIA | `nvidia-smi` (+ [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)) | `nvcr.io/nvidia/pytorch:25.12-py3` (public) | `python .ci/build.py --platform nvidia` |
| Iluvatar | `ixsmi` | `corex:qs_pj20250825` (import in advance) | `python .ci/build.py --platform iluvatar` |
| MetaX | `mx-smi` | `maca-pytorch:3.2.1.4-...` (import in advance) | `python .ci/build.py --platform metax` |
| Moore | `mthreads-gmi` | `vllm_musa:20251112_hygon` (import in advance) | `python .ci/build.py --platform moore` |
| Cambricon | `cnmon` | `cambricon/pytorch:v1.25.3` (import in advance) | `python .ci/build.py --platform cambricon` |
| Ascend | `npu-smi` (+ Ascend driver + CANN toolkit) | `vllm-ascend:v0.18.0rc1-openeuler` (import in advance) | `python .ci/build.py --platform ascend` |

### Start Agent services

On each machine (platform is auto-detected):

```bash
python .ci/agent.py serve --port 8080
```

### Configure remote agent URLs

On the trigger machine, add the `agents` section to `config.yaml` (see [Remote agent configuration](#remote-agent-configuration) above for the format).

### Trigger cross-platform tests

```bash
# Run all platform jobs at once (using config default branch)
python .ci/agent.py run

# Preview mode (no actual execution)
python .ci/agent.py run --dry-run

# Run only a specific platform
python .ci/agent.py run --platform nvidia
```

### Optional configuration

#### GitHub Status reporting

Set the env var on all machines so each reports its own platform's test status:

```bash
export GITHUB_TOKEN=ghp_xxxxxxxxxxxx
```

#### API Token authentication

When agents are exposed on untrusted networks, enable token auth:

```bash
python .ci/agent.py serve --port 8080 --api-token <secret>
# Or: export AGENT_API_TOKEN=<secret>
```

#### GitHub Webhook auto-trigger

In GitHub repo → Settings → Webhooks, add a webhook for each machine:

| Field | Value |
|---|---|
| Payload URL | `http://<machine-ip>:8080/webhook` |
| Content type | `application/json` |
| Secret | Must match `--webhook-secret` |
| Events | `push` and `pull_request` |

```bash
python .ci/agent.py serve --port 8080 --webhook-secret <github-secret>
# Or: export WEBHOOK_SECRET=<github-secret>
```

### Verification checklist

```bash
# 1. Dry-run each machine individually
for platform in nvidia iluvatar metax moore cambricon ascend; do
  python .ci/agent.py run --platform $platform --dry-run
done

# 2. Health and resource checks
for ip in <nvidia-ip> <iluvatar-ip> <metax-ip> <moore-ip> <cambricon-ip> <ascend-ip>; do
  curl http://$ip:8080/health
  curl http://$ip:8080/status
done

# 3. Cross-platform test
python .ci/agent.py run --branch master
```
