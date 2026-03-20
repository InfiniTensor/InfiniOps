#!/usr/bin/env python3
"""Standalone Docker CI runner: clone repo, setup, run stages. Output to stdout."""

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    print(
        "error: pyyaml is required. Install with: pip install pyyaml", file=sys.stderr
    )
    sys.exit(1)


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_git_commit(ref="HEAD"):
    result = subprocess.run(
        ["git", "rev-parse", "--short", ref],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return "unknown"

    return result.stdout.strip()


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
git clone "$REPO_URL" repo
cd repo
git checkout "$BRANCH"
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
  eval "$cmd" || failed=1
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

    for i, s in enumerate(stages):
        args.append("-e")
        args.append(f"STAGE_{i + 1}_NAME={s['name']}")
        args.append("-e")
        args.append(f"STAGE_{i + 1}_CMD={s['run']}")

    gpu_id = gpu_id_override or str(resources.get("gpu_ids", ""))
    gpu_count = resources.get("gpu_count", 0)

    if gpu_id:
        if gpu_id == "all":
            args.extend(["--gpus", "all"])
        else:
            args.extend(["--gpus", f'"device={gpu_id}"'])
    elif gpu_count and gpu_count > 0:
        args.extend(["--gpus", f"count={gpu_count}"])

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


def main():
    parser = argparse.ArgumentParser(description="Run Docker CI pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument("--branch", type=str, help="Override repo branch")
    parser.add_argument("--job", type=str, help="Job name to run (default: first job)")
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
        "--dry-run",
        action="store_true",
        help="Print docker command and exit",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    repo = config.get("repo", {})
    repo_url = repo.get("url", "https://github.com/InfiniTensor/InfiniOps.git")
    branch = args.branch or repo.get("branch", "master")

    jobs = config.get("jobs", {})

    if not jobs:
        print("error: no jobs in config", file=sys.stderr)
        sys.exit(1)

    job_name = args.job or next(iter(jobs))

    if job_name not in jobs:
        print(f"error: job {job_name!r} not in config", file=sys.stderr)
        sys.exit(1)

    job = jobs[job_name]
    all_stages = job.get("stages", [])

    if args.stage:
        stages = [s for s in all_stages if s["name"] == args.stage]

        if not stages:
            print(f"error: stage {args.stage!r} not found", file=sys.stderr)
            sys.exit(1)
    else:
        stages = all_stages

    platform = job.get("platform", "nvidia")
    commit = get_git_commit()
    results_dir = build_results_dir(args.results_dir, platform, stages, commit)

    workdir = "/workspace"
    docker_args = build_docker_args(
        config,
        job_name,
        repo_url,
        branch,
        stages,
        workdir,
        args.image_tag,
        gpu_id_override=args.gpu_id,
        results_dir=results_dir,
    )

    if args.dry_run:
        print(shlex.join(docker_args))
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    sys.exit(subprocess.run(docker_args).returncode)


if __name__ == "__main__":
    main()
