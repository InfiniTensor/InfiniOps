#!/usr/bin/env python3
"""Standalone Docker CI runner: clone repo, setup, run stages. Output to stdout."""

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from ci_resource import (
    GPU_STYLE_NVIDIA,
    GPU_STYLE_NONE,
    GPU_STYLE_MLU,
    GPU_STYLE_NPU,
    ResourcePool,
    detect_platform,
)
from utils import get_git_commit, load_config

# Flags that consume the next token as their value (e.g. -n 4, -k expr).
_PYTEST_VALUE_FLAGS = {"-n", "-k", "-m", "-p", "--tb", "--junitxml", "--rootdir"}


def apply_test_override(run_cmd, test_path):
    """Replace positional test path(s) in a pytest stage command.

    For example: ``pytest tests/ -n 4 ...`` becomes
    ``pytest tests/test_gemm.py -n 4 ...`` when ``test_path`` is
    ``tests/test_gemm.py``.
    """
    parts = shlex.split(run_cmd)

    if not parts or parts[0] != "pytest":
        return run_cmd

    result = ["pytest", test_path]
    skip_next = False

    for p in parts[1:]:
        if skip_next:
            result.append(p)
            skip_next = False
            continue

        if p.startswith("-"):
            result.append(p)
            if p in _PYTEST_VALUE_FLAGS:
                skip_next = True
            continue

        # Skip existing test paths; the override is already in result[1].
        if not ("/" in p or p.endswith(".py") or "::" in p):
            result.append(p)

    return shlex.join(result)


def build_results_dir(base, platform, stages, commit):
    """Build a results directory path: `{base}/{platform}_{stages}_{commit}_{timestamp}`."""
    stage_names = "+".join(s["name"] for s in stages)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dirname = f"{platform}_{stage_names}_{commit}_{timestamp}"

    return Path(base) / dirname


def resolve_image(config, platform, image_tag):
    """Resolve an image reference to a full image name.

    Accepts `stable`, `latest`, or a commit hash as `image_tag`. When config
    contains a registry section, returns a registry-prefixed URL. Otherwise
    returns a local tag (current default).
    """
    registry = config.get("registry", {})
    registry_url = registry.get("url", "")
    project = registry.get("project", "infiniops")

    if not registry_url:
        return f"{project}-ci/{platform}:{image_tag}"

    return f"{registry_url}/{project}/{platform}:{image_tag}"


def build_runner_script():
    return r"""
set -e
cd /workspace
mkdir -p /workspace/results
if [ -n "$LOCAL_SRC" ]; then
  cp -r "$LOCAL_SRC" /tmp/src
  cd /tmp/src
else
  git clone "$REPO_URL" repo
  cd repo
  git checkout "$BRANCH"
fi
echo "========== Setup =========="
eval "$SETUP_CMD"
set +e
failed=0
for i in $(seq 1 "$NUM_STAGES"); do
  name_var="STAGE_${i}_NAME"
  cmd_var="STAGE_${i}_CMD"
  name="${!name_var}"
  cmd="${!cmd_var}"
  echo "========== Stage: $name =========="
  [ -n "$cmd" ] && { eval "$cmd" || failed=1; }
done
echo "========== Summary =========="
if [ -n "$HOST_UID" ] && [ -n "$HOST_GID" ]; then
  chown -R "$HOST_UID:$HOST_GID" /workspace/results 2>/dev/null || true
fi
exit $failed
"""


def build_docker_args(
    config,
    job_name,
    repo_url,
    branch,
    stages,
    workdir,
    image_tag_override,
    gpu_id_override=None,
    results_dir=None,
    local_path=None,
):
    job = config["jobs"][job_name]
    platform = job.get("platform", "nvidia")
    image_tag = image_tag_override or job.get("image", "latest")
    image = resolve_image(config, platform, image_tag)
    resources = job.get("resources", {})
    setup_raw = job.get("setup", "pip install .[dev]")

    if isinstance(setup_raw, list):
        setup_cmd = "\n".join(setup_raw)
    else:
        setup_cmd = setup_raw

    args = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "-i",
        "-w",
        workdir,
        "-e",
        f"REPO_URL={repo_url}",
        "-e",
        f"BRANCH={branch}",
        "-e",
        f"SETUP_CMD={setup_cmd}",
        "-e",
        f"NUM_STAGES={len(stages)}",
        "-e",
        f"HOST_UID={os.getuid()}",
        "-e",
        f"HOST_GID={os.getgid()}",
    ]

    for proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
        proxy_val = os.environ.get(proxy_var) or os.environ.get(proxy_var.lower())

        if proxy_val:
            args.extend(["-e", f"{proxy_var}={proxy_val}"])
            args.extend(["-e", f"{proxy_var.lower()}={proxy_val}"])

    for key, value in job.get("env", {}).items():
        args.extend(["-e", f"{key}={value}"])

    if results_dir:
        args.extend(["-v", f"{results_dir.resolve()}:/workspace/results"])

    if local_path:
        args.extend(["-v", f"{local_path}:/workspace/repo:ro"])
        args.extend(["-e", "LOCAL_SRC=/workspace/repo"])

    for i, s in enumerate(stages):
        args.append("-e")
        args.append(f"STAGE_{i + 1}_NAME={s['name']}")
        args.append("-e")
        args.append(f"STAGE_{i + 1}_CMD={s.get('run', '')}")

    # Platform-specific device access
    for flag in job.get("docker_args", []):
        args.append(flag)

    for vol in job.get("volumes", []):
        args.extend(["-v", vol])

    gpu_id = gpu_id_override or str(resources.get("gpu_ids", ""))
    ngpus = resources.get("ngpus")
    gpu_style = resources.get("gpu_style", GPU_STYLE_NVIDIA)

    if gpu_style == GPU_STYLE_NVIDIA:
        if gpu_id:
            if gpu_id == "all":
                args.extend(["--gpus", "all"])
            else:
                args.extend(["--gpus", f'"device={gpu_id}"'])
        elif ngpus:
            args.extend(["--gpus", f"count={ngpus}"])
    elif gpu_style == GPU_STYLE_NONE and gpu_id and gpu_id != "all":
        # For platforms like Iluvatar/CoreX that use --privileged + /dev mount,
        # control visible GPUs via CUDA_VISIBLE_DEVICES.
        args.extend(["-e", f"CUDA_VISIBLE_DEVICES={gpu_id}"])
    elif gpu_style == GPU_STYLE_MLU and gpu_id and gpu_id != "all":
        # For Cambricon MLU platforms that use --privileged,
        # control visible devices via MLU_VISIBLE_DEVICES.
        args.extend(["-e", f"MLU_VISIBLE_DEVICES={gpu_id}"])
    elif gpu_style == GPU_STYLE_NPU and gpu_id and gpu_id != "all":
        # Ascend: control visible NPU via ASCEND_VISIBLE_DEVICES.
        args.extend(["-e", f"ASCEND_VISIBLE_DEVICES={gpu_id}"])

    memory = resources.get("memory")

    if memory:
        mem = str(memory).lower().replace("gb", "g").replace("mb", "m")

        if not mem.endswith("g") and not mem.endswith("m"):
            mem = f"{mem}g"

        args.extend(["--memory", mem])

    shm_size = resources.get("shm_size")

    if shm_size:
        args.extend(["--shm-size", str(shm_size)])

    timeout_sec = resources.get("timeout")
    args.append(image)

    if timeout_sec:
        # Requires coreutils `timeout` inside the container image.
        args.extend(["timeout", str(timeout_sec)])

    args.extend(["bash", "-c", build_runner_script().strip()])

    return args


def resolve_job_names(jobs, platform, job=None):
    """Resolve job names for a platform.

    - ``job=None`` — all jobs for the platform.
    - ``job="gpu"`` (short name) — matched via ``short_name`` field.
    - ``job="nvidia_gpu"`` (full name) — direct lookup.
    """
    if job and job in jobs:
        return [job]

    if job:
        matches = [
            name
            for name, cfg in jobs.items()
            if cfg.get("platform") == platform and cfg.get("short_name") == job
        ]

        if not matches:
            print(
                f"error: job {job!r} not found for platform {platform!r}",
                file=sys.stderr,
            )
            sys.exit(1)

        return matches

    matches = [name for name, cfg in jobs.items() if cfg.get("platform") == platform]

    if not matches:
        print(f"error: no jobs for platform {platform!r}", file=sys.stderr)
        sys.exit(1)

    return matches


def main():
    parser = argparse.ArgumentParser(description="Run Docker CI pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--branch", type=str, help="Override repo branch (default: config repo.branch)"
    )
    parser.add_argument(
        "--job",
        type=str,
        help="Job name: short name (gpu) or full name (nvidia_gpu). Default: all jobs",
    )
    parser.add_argument(
        "--stage",
        type=str,
        help="Run only this stage name (still runs setup first)",
    )
    parser.add_argument(
        "--image-tag",
        type=str,
        help="Override image tag (stable, latest, or commit hash)",
    )
    parser.add_argument(
        "--gpu-id",
        type=str,
        help='GPU device IDs to use, e.g. "0", "0,2", "all"',
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("ci-results"),
        help="Base directory for test results (default: ./ci-results)",
    )
    parser.add_argument(
        "--test",
        type=str,
        help='Override pytest test path, e.g. "tests/test_gemm.py" or "tests/test_gemm.py::test_gemm"',
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Mount current directory (read-only) into the container instead of cloning from git",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print docker command and exit",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    repo = config.get("repo", {})
    repo_url = repo.get("url", "https://github.com/InfiniTensor/InfiniOps.git")
    branch = args.branch or repo.get("branch", "master")

    platform = detect_platform()

    if not platform:
        tools = ", ".join(ResourcePool.GPU_QUERY_TOOLS.values())
        print(f"error: could not detect platform (no {tools} found)", file=sys.stderr)
        sys.exit(1)

    print(f"platform: {platform}", file=sys.stderr)

    jobs = config.get("jobs", {})

    if not jobs:
        print("error: no jobs in config", file=sys.stderr)
        sys.exit(1)

    job_names = resolve_job_names(jobs, platform, job=args.job)
    failed = 0

    for job_name in job_names:
        job = jobs[job_name]
        all_stages = job.get("stages", [])

        if args.stage:
            stages = [s for s in all_stages if s["name"] == args.stage]

            if not stages:
                print(
                    f"error: stage {args.stage!r} not found in {job_name}",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            stages = all_stages

        if args.test:
            stages = [
                {**s, "run": apply_test_override(s.get("run", ""), args.test)}
                for s in stages
            ]

        job_platform = job.get("platform", platform)
        commit = get_git_commit()
        results_dir = build_results_dir(args.results_dir, job_platform, stages, commit)

        local_path = Path.cwd().resolve() if args.local else None
        docker_args = build_docker_args(
            config,
            job_name,
            repo_url,
            branch,
            stages,
            "/workspace",
            args.image_tag,
            gpu_id_override=args.gpu_id,
            results_dir=results_dir,
            local_path=local_path,
        )

        if args.dry_run:
            print(shlex.join(docker_args))
            continue

        print(f"==> running job: {job_name}", file=sys.stderr)
        results_dir.mkdir(parents=True, exist_ok=True)
        returncode = subprocess.run(docker_args).returncode

        if returncode != 0:
            print(f"job {job_name} failed (exit code {returncode})", file=sys.stderr)
            failed += 1

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
