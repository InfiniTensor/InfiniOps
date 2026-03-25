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
        BASE_IMAGE: nvcr.io/nvidia/pytorch:24.10-py3
    setup: pip install .[dev] --no-build-isolation
    jobs:
      gpu:                              # Flattened as nvidia_gpu
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
      gpu:                              # Flattened as iluvatar_gpu
        resources:
          gpu_ids: "0"
          gpu_style: none               # CoreX: passthrough via --privileged + /dev mount
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
| **Job** | `resources.gpu_ids` | GPU device IDs |
| | `resources.gpu_style` | GPU passthrough: `nvidia` (default) or `none` |
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

Platform is auto-detected (via `nvidia-smi`/`ixsmi`/`mx-smi`/`mthreads-gmi` on PATH), no manual specification needed.

| Flag | Description |
|---|---|
| `--config` | Config file path (default: `.ci/config.yaml`) |
| `--job` | Job name: short (`gpu`) or full (`nvidia_gpu`). Defaults to all jobs for the current platform |
| `--branch` | Override clone branch (default: config `repo.branch`) |
| `--stage` | Run only the specified stage |
| `--image-tag` | Override image tag |
| `--gpu-id` | Override GPU device IDs (nvidia via `--gpus`, others via `CUDA_VISIBLE_DEVICES`) |
| `--results-dir` | Host directory mounted to `/workspace/results` inside the container |
| `--dry-run` | Print docker command without executing |

```bash
# Simplest usage: auto-detect platform, run all jobs, use config default branch
python .ci/run.py

# Specify short job name
python .ci/run.py --job gpu

# Full job name (backward compatible)
python .ci/run.py --job nvidia_gpu

# Run only the test stage, preview mode
python .ci/run.py --job gpu --stage test --dry-run
```

Container execution flow: `git clone` → `checkout` → `setup` → stages.
Proxy vars are forwarded from the host. Test results are written to `--results-dir`. Each run uses a clean environment (no host pip cache mounted).

---

## Platform differences

| Platform | GPU passthrough | Base image | Notes |
|---|---|---|---|
| NVIDIA | `--gpus` (NVIDIA Container Toolkit) | `nvcr.io/nvidia/pytorch:24.10-py3` | Standard CUDA |
| Iluvatar | `--privileged` + `/dev` mount | `corex:qs_pj20250825` | CoreX runtime, CUDA compatible |
| MetaX | `--privileged` | `maca-pytorch:3.2.1.4` | MACA runtime, detected via `mx-smi` |
| Moore | `--privileged` | `vllm_musa:20251112_hygon` | MUSA runtime, detected via `mthreads-gmi` |
| Cambricon | `--privileged` | `cambricon/pytorch:v1.25.3` | Neuware runtime, detected via `cnmon` |
| Ascend | TODO | `ascend-pytorch:24.0.0` | Not ready, image and jobs pending |

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
| `--commit` | Override commit SHA |
| `--image-tag` | Override image tag |
| `--dry-run` | Preview mode |

### Webhook server

Deploy one Agent instance per platform machine (platform is auto-detected):

```bash
# NVIDIA machine
python .ci/agent.py serve --port 8080

# Iluvatar machine
python .ci/agent.py serve --port 8080

# MetaX machine
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
| `/health` | GET | Health check |
| `/status` | GET | Queue + resource status |

Webhook supports `X-Hub-Signature-256` signature verification via `--webhook-secret` or `WEBHOOK_SECRET` env var.

### Remote agent configuration

Configure agent URLs in `config.yaml`; the CLI automatically dispatches remote jobs to the corresponding agents:

```yaml
agents:
  nvidia:
    url: http://nvidia-host:8080
  iluvatar:
    url: http://iluvatar-host:8080
  metax:
    url: http://metax-host:8080
  moore:
    url: http://moore-host:8080
```

### Resource scheduling

The Agent auto-detects GPU utilization and system memory to dynamically determine parallelism:
- GPU utilization < threshold (default 10%) and not allocated by Agent → available
- When resources are insufficient, jobs are queued automatically; completed jobs release resources and trigger scheduling of queued tasks

### GitHub Status

Set the `GITHUB_TOKEN` env var and the Agent will automatically report commit status:
- `pending` — job started
- `success` / `failure` — job completed

Status context format: `ci/infiniops/{job_name}`

---

## Multi-machine deployment guide

Example with NVIDIA + Iluvatar + MetaX + Moore multi-platform setup, showing how to deploy agents across machines for cross-platform parallel testing.

### Prerequisites (all machines)

```bash
# 1. Python 3.10+ and dependencies
pip install pyyaml

# 2. Docker installed
docker --version

# 3. Clone the repository
git clone https://github.com/InfiniTensor/InfiniOps.git
cd InfiniOps
```

### NVIDIA machine setup

```bash
# 1. Install NVIDIA Container Toolkit
#    See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# 2. Verify GPU visibility
nvidia-smi

# 3. Build CI image
python .ci/build.py --platform nvidia
```

### Iluvatar machine setup

```bash
# 1. Verify CoreX runtime is installed
ixsmi

# 2. Verify base image is imported (non-public, must be prepared in advance)
docker images | grep corex    # Should show corex:qs_pj20250825

# 3. Build CI image
python .ci/build.py --platform iluvatar
```

### MetaX machine setup

```bash
# 1. Verify MACA runtime is installed
mx-smi

# 2. Verify base image is imported (non-public, must be prepared in advance)
docker images | grep maca-pytorch    # Should show maca-pytorch:3.2.1.4-torch2.4-py310-ubuntu22.04-amd64

# 3. Build CI image
python .ci/build.py --platform metax
```

### Moore machine setup

```bash
# 1. Verify MUSA runtime is installed
mthreads-gmi

# 2. Verify base image is imported (non-public, must be prepared in advance)
docker images | grep vllm_musa    # Should show vllm_musa:20251112_hygon

# 3. Build CI image
python .ci/build.py --platform moore
```

### Start Agent services

Start the Agent on each machine:

```bash
# NVIDIA machine (platform auto-detected)
python .ci/agent.py serve --port 8080

# Iluvatar machine (platform auto-detected)
python .ci/agent.py serve --port 8080

# MetaX machine (platform auto-detected)
python .ci/agent.py serve --port 8080

# Moore machine (platform auto-detected)
python .ci/agent.py serve --port 8080
```

Verify connectivity:

```bash
curl http://<nvidia-ip>:8080/health
curl http://<iluvatar-ip>:8080/health
curl http://<metax-ip>:8080/health
curl http://<moore-ip>:8080/health
```

### Configure remote agent URLs

Add the `agents` section to `config.yaml` on the trigger machine:

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
# Specify token at startup
python .ci/agent.py serve --port 8080 --api-token <secret>

# Or via env var
export AGENT_API_TOKEN=<secret>
```

#### GitHub Webhook auto-trigger

In GitHub repo → Settings → Webhooks, add a webhook for each machine:

| Field | Value |
|---|---|
| Payload URL | `http://<machine-ip>:8080/webhook` |
| Content type | `application/json` |
| Secret | Must match `--webhook-secret` |
| Events | `push` and `pull_request` |

Configure the secret at startup:

```bash
python .ci/agent.py serve --port 8080 --webhook-secret <github-secret>

# Or via env var
export WEBHOOK_SECRET=<github-secret>
```

### Verification checklist

```bash
# 1. Dry-run each machine individually
python .ci/agent.py run --platform nvidia --dry-run
python .ci/agent.py run --platform iluvatar --dry-run
python .ci/agent.py run --platform metax --dry-run
python .ci/agent.py run --platform moore --dry-run

# 2. Health checks
curl http://<nvidia-ip>:8080/health
curl http://<iluvatar-ip>:8080/health
curl http://<metax-ip>:8080/health
curl http://<moore-ip>:8080/health

# 3. Resource status
curl http://<nvidia-ip>:8080/status
curl http://<iluvatar-ip>:8080/status
curl http://<metax-ip>:8080/status
curl http://<moore-ip>:8080/status

# 4. Cross-platform test
python .ci/agent.py run --branch master
```
