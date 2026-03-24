#!/usr/bin/env python3
"""Resource detection and allocation for CI Runner Agent."""

import json
import operator
import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass

# GPU passthrough styles
GPU_STYLE_NVIDIA = "nvidia"
GPU_STYLE_NONE = "none"
GPU_STYLE_MLU = "mlu"


@dataclass
class GpuInfo:
    index: int
    memory_used_mb: float
    memory_total_mb: float
    utilization_pct: float


@dataclass
class SystemResources:
    total_memory_mb: float
    available_memory_mb: float
    cpu_count: int


class ResourcePool:
    """Thread-safe GPU and system resource manager.

    Detects available GPUs via platform-specific tools (nvidia-smi, ixsmi, mx-smi, mthreads-gmi)
    and tracks allocations to enable dynamic parallel scheduling.
    """

    GPU_QUERY_TOOLS = {
        "nvidia": "nvidia-smi",
        "iluvatar": "ixsmi",
        "metax": "mx-smi",
        "moore": "mthreads-gmi",
        "cambricon": "cnmon",
    }

    def __init__(self, platform, utilization_threshold=10):
        self._platform = platform
        self._utilization_threshold = utilization_threshold
        self._allocated: set[int] = set()
        self._lock = threading.Lock()

    @property
    def platform(self):
        return self._platform

    @property
    def allocated(self):
        with self._lock:
            return set(self._allocated)

    def detect_gpus(self) -> list[GpuInfo]:
        """Query GPU status via platform-specific CLI tool."""
        if self._platform == "metax":
            return self._detect_gpus_metax()

        if self._platform == "moore":
            return self._detect_gpus_moore()

        if self._platform == "cambricon":
            return self._detect_gpus_cambricon()

        tool = self.GPU_QUERY_TOOLS.get(self._platform)

        if not tool:
            return []

        try:
            result = subprocess.run(
                [
                    tool,
                    "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

        if result.returncode != 0:
            return []

        gpus = []

        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]

            if len(parts) < 4:
                continue

            try:
                gpus.append(
                    GpuInfo(
                        index=int(parts[0]),
                        memory_used_mb=float(parts[1]),
                        memory_total_mb=float(parts[2]),
                        utilization_pct=float(parts[3]),
                    )
                )
            except (ValueError, IndexError):
                continue

        return gpus

    def _detect_gpus_metax(self) -> list[GpuInfo]:
        """Parse mx-smi output for MetaX GPUs.

        Runs --show-memory and --show-usage separately and merges results.
        Output format example:
            GPU#0  MXC550  0000:1a:00.0
                Memory
                    vis_vram total  : 67108864 KB
                    vis_vram used   : 879032 KB
                Utilization
                    GPU             : 0 %
        """

        def run_mxsmi(flag):
            try:
                r = subprocess.run(
                    ["mx-smi", flag],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                return r.stdout if r.returncode == 0 else ""
            except (FileNotFoundError, subprocess.TimeoutExpired):
                return ""

        mem_out = run_mxsmi("--show-memory")
        util_out = run_mxsmi("--show-usage")

        # Parse memory: collect {index: (used_kb, total_kb)}
        mem = {}
        current = None
        for line in mem_out.splitlines():
            m = re.match(r"GPU#(\d+)", line.strip())
            if m:
                current = int(m.group(1))
                mem[current] = [0.0, 0.0]
                continue
            if current is None:
                continue
            m = re.search(r"vis_vram total\s*:\s*([\d.]+)\s*KB", line)
            if m:
                mem[current][1] = float(m.group(1)) / 1024  # KB -> MB
            m = re.search(r"vis_vram used\s*:\s*([\d.]+)\s*KB", line)
            if m:
                mem[current][0] = float(m.group(1)) / 1024  # KB -> MB

        # Parse utilization: collect {index: utilization_pct}
        util = {}
        current = None
        in_util = False
        for line in util_out.splitlines():
            m = re.match(r"GPU#(\d+)", line.strip())
            if m:
                current = int(m.group(1))
                in_util = False
                continue
            if current is None:
                continue
            if "Utilization" in line:
                in_util = True
                continue
            if in_util:
                m = re.match(r"\s*GPU\s*:\s*([\d.]+)\s*%", line)
                if m:
                    util[current] = float(m.group(1))
                    in_util = False

        gpus = []
        for idx in sorted(mem):
            used_mb, total_mb = mem[idx]
            gpus.append(
                GpuInfo(
                    index=idx,
                    memory_used_mb=used_mb,
                    memory_total_mb=total_mb,
                    utilization_pct=util.get(idx, 0.0),
                )
            )
        return gpus

    def _detect_gpus_moore(self) -> list[GpuInfo]:
        """Parse mthreads-gmi JSON output for Moore Threads GPUs.

        Uses: mthreads-gmi -q --json
        Expected JSON structure:
            {
              "Attached GPUs": {
                "GPU 00000000:3B:00.0": {
                  "Minor Number": "0",
                  "Memory Usage": {
                    "Total": "24576 MiB",
                    "Used": "512 MiB"
                  },
                  "Utilization": {
                    "Gpu": "5 %"
                  }
                }
              }
            }
        """

        def extract_number(s):
            m = re.search(r"([\d.]+)", str(s))
            return float(m.group(1)) if m else 0.0

        try:
            result = subprocess.run(
                ["mthreads-gmi", "-q", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

        if result.returncode != 0:
            return []

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return []

        gpus = []
        attached = data.get("Attached GPUs", {})

        for gpu_data in attached.values():
            try:
                index = int(gpu_data.get("Minor Number", len(gpus)))

                mem = gpu_data.get("Memory Usage", {})
                total_mb = extract_number(mem.get("Total", "0 MiB"))
                used_mb = extract_number(mem.get("Used", "0 MiB"))
                util_pct = extract_number(
                    gpu_data.get("Utilization", {}).get("Gpu", "0 %")
                )

                gpus.append(
                    GpuInfo(
                        index=index,
                        memory_used_mb=used_mb,
                        memory_total_mb=total_mb,
                        utilization_pct=util_pct,
                    )
                )
            except (ValueError, AttributeError):
                continue

        return sorted(gpus, key=operator.attrgetter("index"))

    def _detect_gpus_cambricon(self) -> list[GpuInfo]:
        """Parse cnmon output for Cambricon MLU cards.

        Each card appears as two consecutive data rows:
            Row 1: | {card}  {vf}  {name}  {fw} | {bus_id} | {util}%  {ecc} |
            Row 2: | {fan}%  {temp}  {pwr} | {mem_used} MiB/ {mem_total} MiB | ... |
        """
        try:
            result = subprocess.run(
                ["cnmon"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

        if result.returncode != 0:
            return []

        gpus = []
        lines = result.stdout.splitlines()
        i = 0

        while i < len(lines):
            line = lines[i]
            # Row 1: "| {index} ... | {bus_id} | {util}%  {ecc} |"
            m1 = re.match(r"^\|\s+(\d+)\s+.*\|\s*([\d.]+)%", line)

            if m1 and i + 1 < len(lines):
                try:
                    card_index = int(m1.group(1))
                    util_pct = float(m1.group(2))
                    row2 = lines[i + 1]
                    mem_m = re.search(r"([\d.]+)\s+MiB/\s*([\d.]+)\s+MiB", row2)

                    if mem_m:
                        used_mb = float(mem_m.group(1))
                        total_mb = float(mem_m.group(2))
                    else:
                        used_mb, total_mb = 0.0, 0.0

                    gpus.append(
                        GpuInfo(
                            index=card_index,
                            memory_used_mb=used_mb,
                            memory_total_mb=total_mb,
                            utilization_pct=util_pct,
                        )
                    )
                except (ValueError, AttributeError):
                    pass
                i += 2
                continue

            i += 1

        return sorted(gpus, key=operator.attrgetter("index"))

    def detect_system_resources(self) -> SystemResources:
        """Read system memory from /proc/meminfo and CPU count."""
        total_mb = 0.0
        available_mb = 0.0

        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_mb = float(line.split()[1]) / 1024
                    elif line.startswith("MemAvailable:"):
                        available_mb = float(line.split()[1]) / 1024
        except OSError:
            pass

        return SystemResources(
            total_memory_mb=total_mb,
            available_memory_mb=available_mb,
            cpu_count=os.cpu_count() or 1,
        )

    def get_free_gpus(self) -> list[int]:
        """Return GPU indices with utilization below threshold."""
        gpus = self.detect_gpus()
        return [
            g.index for g in gpus if g.utilization_pct < self._utilization_threshold
        ]

    def allocate(self, gpu_count, memory_mb=0) -> tuple[list[int], bool]:
        """Try to allocate GPUs and check memory.

        Returns (allocated_gpu_ids, success). On failure returns ([], False).
        GPU detection and memory checks run outside the lock to avoid blocking
        other threads while subprocess.run (nvidia-smi) executes.
        """
        if gpu_count <= 0:
            if memory_mb > 0:
                sys_res = self.detect_system_resources()

                if sys_res.available_memory_mb < memory_mb:
                    return ([], False)

            return ([], True)

        # Detect GPUs and memory outside the lock (subprocess.run can block)
        free_gpus = set(self.get_free_gpus())
        sys_res = self.detect_system_resources() if memory_mb > 0 else None

        with self._lock:
            available = free_gpus - self._allocated

            if len(available) < gpu_count:
                return ([], False)

            if sys_res is not None and sys_res.available_memory_mb < memory_mb:
                return ([], False)

            selected = sorted(available)[:gpu_count]
            self._allocated.update(selected)
            return (selected, True)

    def release(self, gpu_ids):
        """Return GPUs to the free pool."""
        with self._lock:
            self._allocated -= set(gpu_ids)

    def get_status(self) -> dict:
        """Return current resource status for API endpoints."""
        gpus = self.detect_gpus()
        sys_res = self.detect_system_resources()

        with self._lock:
            allocated = sorted(self._allocated)

        return {
            "platform": self._platform,
            "gpus": [
                {
                    "index": g.index,
                    "memory_used_mb": g.memory_used_mb,
                    "memory_total_mb": g.memory_total_mb,
                    "utilization_pct": g.utilization_pct,
                    "allocated_by_agent": g.index in allocated,
                }
                for g in gpus
            ],
            "allocated_gpu_ids": allocated,
            "system": {
                "total_memory_mb": round(sys_res.total_memory_mb, 1),
                "available_memory_mb": round(sys_res.available_memory_mb, 1),
                "cpu_count": sys_res.cpu_count,
            },
            "utilization_threshold": self._utilization_threshold,
        }


def parse_gpu_requirement(job_config) -> int:
    """Extract GPU count requirement from a job config."""
    resources = job_config.get("resources", {})
    gpu_style = resources.get("gpu_style", GPU_STYLE_NVIDIA)

    if gpu_style == GPU_STYLE_NONE:
        return 0

    gpu_ids = str(resources.get("gpu_ids", ""))

    if not gpu_ids:
        return resources.get("gpu_count", 0)

    if gpu_ids == "all":
        return 0  # "all" means use all available, don't reserve specific count

    return len(gpu_ids.split(","))


def parse_memory_requirement(job_config) -> float:
    """Extract memory requirement in MB from a job config."""
    resources = job_config.get("resources", {})
    memory = str(resources.get("memory", ""))

    if not memory:
        return 0

    memory = memory.lower().strip()

    if memory.endswith("gb"):
        return float(memory[:-2]) * 1024
    elif memory.endswith("g"):
        return float(memory[:-1]) * 1024
    elif memory.endswith("mb"):
        return float(memory[:-2])
    elif memory.endswith("m"):
        return float(memory[:-1])

    try:
        return float(memory) * 1024  # Default: GB
    except ValueError:
        return 0


def detect_platform():
    """Auto-detect the current platform by probing GPU query tools on PATH."""
    for platform, tool in ResourcePool.GPU_QUERY_TOOLS.items():
        if shutil.which(tool):
            return platform

    return None
