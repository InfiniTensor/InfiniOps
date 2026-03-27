#!/usr/bin/env python3
"""Convert .ci/config.yaml into a GitHub Actions matrix JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


THIS_FILE = Path(__file__).resolve()
CI_DIR = THIS_FILE.parents[1]
if str(CI_DIR) not in sys.path:
    sys.path.insert(0, str(CI_DIR))

from ci_resource import GPU_STYLE_MLU, GPU_STYLE_NVIDIA  # noqa: E402
from utils import load_config  # noqa: E402


def _normalize_build_args(build_args: dict[str, Any] | None) -> list[str]:
    return [f"{key}={value}" for key, value in (build_args or {}).items()]


def _collect_test_command(job_cfg: dict[str, Any]) -> str:
    stages = job_cfg.get("stages", [])
    runs = [stage.get("run", "").strip() for stage in stages if stage.get("run")]
    return " && ".join(cmd for cmd in runs if cmd)


def _normalize_gpu_style(value: Any) -> str:
    gpu_style = str(value or GPU_STYLE_NVIDIA).strip().lower()
    if gpu_style in {GPU_STYLE_NVIDIA, "none", GPU_STYLE_MLU}:
        return gpu_style
    return GPU_STYLE_NVIDIA


def _entry_from_flat_job(
    job_id: str, job_cfg: dict[str, Any], image_cfg: dict[str, Any]
) -> dict[str, Any]:
    platform = str(job_cfg.get("platform", "")).strip()
    short_name = str(job_cfg.get("short_name", "")).strip() or job_id
    resources = job_cfg.get("resources", {})

    dockerfile_dir = str(image_cfg.get("dockerfile", "")).rstrip("/")
    dockerfile = f"{dockerfile_dir}/Dockerfile" if dockerfile_dir else ""

    timeout_seconds = int(resources.get("timeout", 3600))
    timeout_minutes = max(1, timeout_seconds // 60)

    return {
        "id": job_id,
        "platform": platform,
        "job_name": short_name,
        "runner_label": platform,
        "dockerfile": dockerfile,
        "build_args": _normalize_build_args(image_cfg.get("build_args")),
        "docker_args": job_cfg.get("docker_args", []),
        "volumes": job_cfg.get("volumes", []),
        "setup": str(job_cfg.get("setup", "")).strip(),
        "test_cmd": _collect_test_command(job_cfg),
        "timeout_minutes": timeout_minutes,
        "memory": resources.get("memory", ""),
        "shm_size": resources.get("shm_size", ""),
        "gpu_ids": str(resources.get("gpu_ids", "")).strip(),
        "ngpus": resources.get("ngpus", ""),
        "gpu_style": _normalize_gpu_style(resources.get("gpu_style")),
        "job_env": job_cfg.get("env", {}),
    }


def convert(config: dict[str, Any]) -> dict[str, Any]:
    include = []
    jobs = config.get("jobs", {})
    images = config.get("images", {})

    for job_id, job_cfg in jobs.items():
        platform = str(job_cfg.get("platform", "")).strip()
        image_cfg = images.get(platform, {})
        include.append(_entry_from_flat_job(job_id, job_cfg, image_cfg))

    if not include:
        raise ValueError("No jobs found in normalized config['jobs']")
    return {"include": include}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert custom CI config to GitHub matrix JSON"
    )
    parser.add_argument(
        "--config", type=Path, default=Path(".ci/config.yaml"), help="Path to config file"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    matrix = convert(config)
    print(json.dumps(matrix, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
