#!/usr/bin/env python3
"""Standalone Docker CI runner: clone repo, setup, run stages. Output to stdout."""

import argparse
import subprocess
import sys
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


def resolve_image(config, platform, image_tag):
    """Resolve an image reference ('stable', 'latest', or commit hash) to a full URL."""
    registry = config.get("registry", {})
    registry_url = registry.get("url", "")
    project = registry.get("project", "infiniops")

    if not registry_url:
        return f"{project}-ci/{platform}:{image_tag}"

    return f"{registry_url}/{project}/{platform}:{image_tag}"


def build_runner_script():
    return r"""
export https_proxy=http://localhost:9991
set -e
cd /workspace
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
exit $failed
"""


def build_docker_args(
    config, job_name, repo_url, branch, stages, workdir, image_tag_override,
    gpu_id_override=None,
):
    job = config["jobs"][job_name]
    platform = job.get("platform", "nvidia")
    image_tag = image_tag_override or job.get("image", "stable")
    image = resolve_image(config, platform, image_tag)
    resources = job.get("resources", {})
    setup_cmd = job.get("setup", "pip install .[dev]")

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
    ]
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
        mem = str(memory).upper().replace("GB", "g").replace("MB", "m")
        if not mem.endswith("g") and not mem.endswith("m"):
            mem = f"{mem}g"
        args.extend(["--memory", mem])

    timeout_sec = resources.get("timeout")
    if timeout_sec:
        args.extend(["--stop-timeout", str(timeout_sec)])

    args.append(image)
    args.append("bash")
    args.append("-c")
    args.append(build_runner_script().strip())

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
        "--dry-run",
        action="store_true",
        help="Print docker command and exit",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    repo = config.get("repo", {})
    repo_url = repo.get("url", "https://github.com/InfiniTensor/InfiniOps.git")
    branch = args.branch or repo.get("branch", "dev-infra")

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

    workdir = "/workspace"
    docker_args = build_docker_args(
        config, job_name, repo_url, branch, stages, workdir, args.image_tag,
        gpu_id_override=args.gpu_id,
    )

    if args.dry_run:
        print(" ".join(docker_args))

        return

    sys.exit(subprocess.run(docker_args).returncode)


if __name__ == "__main__":
    main()
