#!/usr/bin/env python3
"""Shared utilities for the CI toolchain."""

import subprocess
import sys

try:
    import yaml
except ImportError:
    print(
        "error: pyyaml is required. Install with: pip install pyyaml", file=sys.stderr
    )
    sys.exit(1)


def normalize_config(raw):
    """Convert platform-centric config to flat images/jobs format.

    Input (new format):
        platforms:
          nvidia:
            image: {dockerfile: ..., build_args: ...}
            setup: pip install .[dev]
            jobs:
              gpu: {resources: ..., stages: ...}

    Output (flat format consumed by run.py / build.py / agent.py):
        images:
          nvidia: {dockerfile: ..., build_args: ...}
        jobs:
          nvidia_gpu: {platform: nvidia, setup: ..., resources: ..., stages: ...}

    If the config already uses the flat format (no 'platforms' key), returns as-is.
    """
    if "platforms" not in raw:
        return raw

    config = {}

    for key in ("repo", "github", "agents"):
        if key in raw:
            config[key] = raw[key]

    config["images"] = {}
    config["jobs"] = {}

    for platform, pcfg in raw.get("platforms", {}).items():
        # Image config
        if "image" in pcfg:
            config["images"][platform] = pcfg["image"]

        # Platform-level defaults inherited by jobs
        defaults = {}

        for key in ("image_tag", "docker_args", "volumes", "setup", "env"):
            if key in pcfg:
                defaults[key] = pcfg[key]

        # Flatten jobs: {platform}_{job_name}
        for job_name, job_cfg in pcfg.get("jobs", {}).items():
            full_name = f"{platform}_{job_name}"
            flat = {
                "platform": platform,
                "image": defaults.get("image_tag", "latest"),
            }

            # Apply platform defaults
            for key in ("docker_args", "volumes", "setup", "env"):
                if key in defaults:
                    flat[key] = defaults[key]

            # Job-level overrides
            flat.update(job_cfg)

            config["jobs"][full_name] = flat

    return config


def load_config(path):
    """Load a YAML config file and normalize to flat format."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return normalize_config(raw)


def get_git_commit(ref="HEAD", short=True):
    """Get git commit SHA. Returns 'unknown' on failure."""
    cmd = ["git", "rev-parse"]

    if short:
        cmd.append("--short")

    cmd.append(ref)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return "unknown"

    return result.stdout.strip()
