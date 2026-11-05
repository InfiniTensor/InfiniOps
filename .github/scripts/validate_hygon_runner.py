"""One-runner acceptance using the configured Hygon setup and smoke suite.

The host has Python 3.6; build/test tools run inside the platform image.
"""

import fcntl
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import subprocess
import time

import yaml


SMI = "/usr/local/hyhal/bin/hy-smi"
LOCKS = Path("/tmp/infinitensor-ci-resource-locks")


def detect_devices():
    output = subprocess.check_output(
        [SMI, "--showmeminfo", "vram", "--showuse", "--json"],
        universal_newlines=True,
        timeout=20,
    )
    devices = []
    for name, info in json.loads(output).items():
        if not re.fullmatch(r"card[0-9]+", name):
            continue
        devices.append(
            (
                int(name[4:]),
                float(info["vram Total Used Memory (MiB)"]),
                float(info["HCU use (%)"]),
            )
        )
    if not devices:
        raise RuntimeError("hy-smi returned no devices; no test was started")
    return devices


def acquire_device(queue_seconds):
    deadline = time.monotonic() + queue_seconds
    LOCKS.mkdir(parents=True, exist_ok=True)
    while True:
        with (LOCKS / "hygon.allocate.lock").open("a+") as allocation:
            fcntl.flock(allocation, fcntl.LOCK_EX)
            for index, used_mb, utilization in sorted(detect_devices()):
                if used_mb >= 100 or utilization >= 10:
                    continue
                lease = (LOCKS / ("hygon.device.%d.lock" % index)).open("a+")
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    continue
                lease.seek(0)
                lease.truncate()
                lease.write(
                    json.dumps(
                        {
                            "device_id": index,
                            "pid": os.getpid(),
                            "run_id": os.environ.get("GITHUB_RUN_ID"),
                        }
                    )
                )
                lease.flush()
                print(
                    "Allocated Hygon device %d (%.1f MiB, %.1f%% use)"
                    % (index, used_mb, utilization),
                    flush=True,
                )
                return index, lease
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "GPU resource wait timed out; no build or test was started"
            )
        print("Waiting for an idle Hygon device", flush=True)
        time.sleep(min(10, max(0, deadline - time.monotonic())))


def main():
    assert pwd.getpwuid(os.getuid()).pw_name == "infiniops"
    assert os.environ.get("RUNNER_NAME") == "Hygon-Server-000"
    repo = Path.cwd().resolve()
    with (repo / ".github/ci_config.yml").open() as source:
        config = yaml.safe_load(source)["platforms"]["hygon"]
    resources = config["jobs"]["gpu"]["resources"]
    assert resources["gpu_ids"] == "auto" and resources["ngpus"] == 1
    image = config["image"]["build_args"]["BASE_IMAGE"]
    subprocess.run(["docker", "info"], check=True, timeout=20)
    subprocess.run(["docker", "pull", image], check=True)
    results = repo / "ci-results/hygon-runner"
    results.mkdir(parents=True, exist_ok=True)
    device, lease = acquire_device(int(resources["queue_timeout"]))
    name = "infiniops-hygon-validation-" + os.environ["GITHUB_RUN_ID"]
    preflight = """import torch
print('Torch:', torch.__version__, 'HIP:', torch.version.hip, flush=True)
assert torch.cuda.is_available(), 'DCU runtime is unavailable'
assert torch.cuda.device_count() == 1, 'Assigned-device isolation was not honored'
print('Assigned GPU:', torch.cuda.get_device_name(0), flush=True)
a=torch.ones((16,16),device='cuda')
torch.testing.assert_close(a@a,torch.full((16,16),16.,device='cuda'))
torch.cuda.synchronize()
"""
    test = config["jobs"]["gpu"]["stages"][0]["run"]
    setup = config["setup"]
    script = (
        "set -euo pipefail\ngit config --global --add safe.directory /workspace\npython3 - <<'GPU_CHECK'\n"
        + preflight
        + "GPU_CHECK\n"
        + "bash -euo pipefail -c "
        + shlex.quote(setup)
        + "\n"
        + test
        + "\n"
    )
    args = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--memory",
        resources["memory"],
        "--shm-size",
        resources["shm_size"],
    ]
    args.extend(config["docker_args"])
    for volume in config["volumes"]:
        args.extend(["-v", volume])
    args.extend(
        [
            "-v",
            str(repo) + ":/workspace",
            "-v",
            str(results) + ":/workspace/results",
            "-w",
            "/workspace",
            "-e",
            "HIP_VISIBLE_DEVICES=" + str(device),
            "--entrypoint",
            "bash",
            image,
            "-lc",
            script,
        ]
    )
    try:
        with (results / "lease.json").open("w") as output:
            json.dump(
                {
                    "physical_device": device,
                    "visible_device_count_required": 1,
                    "image": image,
                },
                output,
            )
        try:
            result = subprocess.run(args, timeout=int(resources["timeout"]))
        except subprocess.TimeoutExpired:
            raise RuntimeError("The configured 3600-second build/test budget expired")
        if result.returncode:
            raise SystemExit(result.returncode)
    finally:
        try:
            subprocess.run(
                ["docker", "stop", "--time", "30", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=40,
            )
        finally:
            lease.close()


if __name__ == "__main__":
    main()
