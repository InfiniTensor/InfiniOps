#!/usr/bin/env python3
"""CI image builder: detect changes, build, tag, and optionally push Docker images."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from utils import get_git_commit, load_config


def has_dockerfile_changed(dockerfile_dir, base_ref="HEAD~1"):
    """Check if any file under `dockerfile_dir` changed since `base_ref`."""
    result = subprocess.run(
        ["git", "diff", "--name-only", base_ref, "--", dockerfile_dir],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(
            "warning: git diff failed (shallow clone or initial commit?);"
            " assuming Dockerfile changed",
            file=sys.stderr,
        )
        return True

    return bool(result.stdout.strip())


def docker_login(registry_cfg, dry_run):
    """Log in to the registry using `credentials_env` token.

    Returns True on success.

    NOTE: Registry support is currently unused (`config.yaml` has no registry
    section). Retained for future integration with an external image management
    system.
    """
    credentials_env = registry_cfg.get("credentials_env")
    registry_url = registry_cfg.get("url", "")

    if not credentials_env or not registry_url:
        return True

    token = os.environ.get(credentials_env)

    if not token:
        print(
            f"error: {credentials_env} not set, cannot login",
            file=sys.stderr,
        )
        return False

    if dry_run:
        print(
            f"[dry-run] echo <token> | docker login {registry_url}"
            " --username token --password-stdin"
        )
        return True

    result = subprocess.run(
        ["docker", "login", registry_url, "--username", "token", "--password-stdin"],
        input=token,
        text=True,
    )

    if result.returncode != 0:
        print("error: docker login failed", file=sys.stderr)
        return False

    return True


def build_image_tag(registry_url, project, platform, tag):
    if registry_url:
        return f"{registry_url}/{project}/{platform}:{tag}"

    return f"{project}-ci/{platform}:{tag}"


def build_image(platform, platform_cfg, registry_cfg, commit, push, dry_run, logged_in):
    """Build a single platform image. Returns True on success."""
    registry_url = registry_cfg.get("url", "")
    project = registry_cfg.get("project", "infiniops")
    dockerfile_dir = platform_cfg["dockerfile"]
    commit_tag = build_image_tag(registry_url, project, platform, commit)
    latest_tag = build_image_tag(registry_url, project, platform, "latest")

    build_args_cfg = platform_cfg.get("build_args", {})
    build_cmd = ["docker", "build", "--network", "host"]

    for key, value in build_args_cfg.items():
        build_cmd.extend(["--build-arg", f"{key}={value}"])

    for proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
        proxy_val = os.environ.get(proxy_var) or os.environ.get(proxy_var.lower())

        if proxy_val:
            build_cmd.extend(["--build-arg", f"{proxy_var}={proxy_val}"])
            build_cmd.extend(["--build-arg", f"{proxy_var.lower()}={proxy_val}"])

    private_sdk = platform_cfg.get("private_sdk", {})

    if private_sdk:
        source_env = private_sdk.get("source_env", "")
        sdk_url = os.environ.get(source_env, "") if source_env else ""

        if sdk_url:
            build_cmd.extend(["--build-arg", f"PRIVATE_SDK_URL={sdk_url}"])

    build_cmd.extend(["-t", commit_tag, "-t", latest_tag, dockerfile_dir])

    if dry_run:
        print(f"[dry-run] {shlex.join(build_cmd)}")

        if push:
            if not logged_in:
                print("[dry-run] (skipping push: docker login failed)")
            else:
                print(f"[dry-run] docker push {commit_tag}")
                print(f"[dry-run] docker push {latest_tag}")

        return True

    print(f"==> building {platform}: {commit_tag}", file=sys.stderr)
    result = subprocess.run(build_cmd)

    if result.returncode != 0:
        error = {
            "stage": "build",
            "platform": platform,
            "tag": commit_tag,
            "exit_code": result.returncode,
        }
        print(json.dumps(error), file=sys.stderr)

        return False

    if push:
        if not logged_in:
            print("error: docker login failed, cannot push", file=sys.stderr)
            return False

        for tag in (commit_tag, latest_tag):
            print(f"==> pushing {tag}", file=sys.stderr)
            push_result = subprocess.run(["docker", "push", tag])

            if push_result.returncode != 0:
                error = {
                    "stage": "push",
                    "platform": platform,
                    "tag": tag,
                    "exit_code": push_result.returncode,
                }
                print(json.dumps(error), file=sys.stderr)

                return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Build CI Docker images")
    parser.add_argument(
        "--platform",
        type=str,
        default="all",
        help="Platform to build (nvidia, iluvatar, metax, moore, cambricon, ascend, or all). Default: all",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default="HEAD",
        help="Git ref for tagging the image (default: HEAD)",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push images to registry after building (requires registry in config)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip change detection and force build",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    registry_cfg = config.get("registry", {})
    images_cfg = config.get("images", {})

    if not images_cfg:
        print("error: no `images` section in config", file=sys.stderr)
        sys.exit(1)

    if args.platform == "all":
        platforms = list(images_cfg.keys())
    else:
        if args.platform not in images_cfg:
            print(
                f"error: platform `{args.platform}` not found in config",
                file=sys.stderr,
            )
            sys.exit(1)
        platforms = [args.platform]

    commit = get_git_commit(args.commit)
    logged_in = docker_login(registry_cfg, args.dry_run) if args.push else True
    failed = False

    for platform in platforms:
        platform_cfg = images_cfg[platform]
        dockerfile_dir = platform_cfg["dockerfile"]

        if not Path(dockerfile_dir).is_dir():
            print(
                f"warning: dockerfile directory `{dockerfile_dir}` does not exist,"
                f" skipping {platform}",
                file=sys.stderr,
            )
            continue

        if not args.force and not has_dockerfile_changed(dockerfile_dir):
            print(f"==> {platform}: no changes detected, skipping", file=sys.stderr)
            continue

        ok = build_image(
            platform,
            platform_cfg,
            registry_cfg,
            commit,
            args.push,
            args.dry_run,
            logged_in=logged_in,
        )

        if not ok:
            failed = True

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
