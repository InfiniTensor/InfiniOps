import threading


import ci_resource as res


# ---------------------------------------------------------------------------
# Tests for `GpuInfo` and `SystemResources`.
# ---------------------------------------------------------------------------


def test_gpu_info_fields():
    g = res.GpuInfo(
        index=0, memory_used_mb=1000, memory_total_mb=8000, utilization_pct=50
    )
    assert g.index == 0
    assert g.memory_total_mb == 8000


def test_system_resources_fields():
    s = res.SystemResources(
        total_memory_mb=32000, available_memory_mb=16000, cpu_count=8
    )
    assert s.cpu_count == 8


# ---------------------------------------------------------------------------
# Tests for `detect_gpus`.
# ---------------------------------------------------------------------------


def test_detect_gpus_nvidia_parses_csv(monkeypatch):
    csv_output = "0, 512, 8192, 5\n1, 1024, 8192, 80\n"

    def mock_run(cmd, **kwargs):
        class R:
            returncode = 0
            stdout = csv_output

        return R()

    monkeypatch.setattr("subprocess.run", mock_run)

    pool = res.ResourcePool("nvidia")
    gpus = pool.detect_gpus()
    assert len(gpus) == 2
    assert gpus[0].index == 0
    assert gpus[0].memory_used_mb == 512
    assert gpus[0].utilization_pct == 5
    assert gpus[1].index == 1
    assert gpus[1].utilization_pct == 80


def test_detect_gpus_empty_on_failure(monkeypatch):
    def mock_run(cmd, **kwargs):
        class R:
            returncode = 1
            stdout = ""

        return R()

    monkeypatch.setattr("subprocess.run", mock_run)

    pool = res.ResourcePool("nvidia")
    assert pool.detect_gpus() == []


def test_detect_gpus_unknown_platform():
    pool = res.ResourcePool("unknown_platform")
    assert pool.detect_gpus() == []


def test_detect_gpus_file_not_found(monkeypatch):
    def mock_run(cmd, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr("subprocess.run", mock_run)

    pool = res.ResourcePool("nvidia")
    assert pool.detect_gpus() == []


# ---------------------------------------------------------------------------
# Tests for `detect_system_resources`.
# ---------------------------------------------------------------------------


def test_detect_system_resources(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       32000000 kB\n"
        "MemFree:        10000000 kB\n"
        "MemAvailable:   20000000 kB\n"
    )

    _real_open = open

    def fake_open(path, **kw):
        if str(path) == "/proc/meminfo":
            return _real_open(str(meminfo), **kw)
        return _real_open(path, **kw)

    monkeypatch.setattr("builtins.open", fake_open)

    pool = res.ResourcePool("nvidia")
    sys_res = pool.detect_system_resources()
    assert abs(sys_res.total_memory_mb - 32000000 / 1024) < 1
    assert abs(sys_res.available_memory_mb - 20000000 / 1024) < 1
    assert sys_res.cpu_count > 0


# ---------------------------------------------------------------------------
# Tests for `allocate` picking least-loaded GPUs.
# ---------------------------------------------------------------------------


def test_allocate_picks_least_loaded(monkeypatch):
    csv_output = "0, 100, 8192, 8\n1, 200, 8192, 2\n2, 300, 8192, 5\n"

    def mock_run(cmd, **kwargs):
        class R:
            returncode = 0
            stdout = csv_output

        return R()

    monkeypatch.setattr("subprocess.run", mock_run)

    pool = res.ResourcePool("nvidia", utilization_threshold=10)
    gpu_ids, ok = pool.allocate(1)
    assert ok is True
    assert gpu_ids == [1]  # GPU 1 has lowest utilization (2%).


def test_allocate_picks_two_least_loaded(monkeypatch):
    csv_output = "0, 100, 8192, 8\n1, 200, 8192, 2\n2, 300, 8192, 5\n"

    def mock_run(cmd, **kwargs):
        class R:
            returncode = 0
            stdout = csv_output

        return R()

    monkeypatch.setattr("subprocess.run", mock_run)

    pool = res.ResourcePool("nvidia", utilization_threshold=10)
    gpu_ids, ok = pool.allocate(2)
    assert ok is True
    assert gpu_ids == [1, 2]  # Sorted by utilization: 2% then 5%.


def test_allocate_skips_busy_gpus(monkeypatch):
    csv_output = "0, 100, 8192, 5\n1, 4000, 8192, 95\n2, 200, 8192, 8\n"

    def mock_run(cmd, **kwargs):
        class R:
            returncode = 0
            stdout = csv_output

        return R()

    monkeypatch.setattr("subprocess.run", mock_run)

    pool = res.ResourcePool("nvidia", utilization_threshold=10)
    gpu_ids, ok = pool.allocate(2)
    assert ok is True
    assert set(gpu_ids) == {0, 2}
    assert 1 not in gpu_ids  # GPU 1 at 95% is above threshold


# ---------------------------------------------------------------------------
# Tests for `allocate` and `release`.
# ---------------------------------------------------------------------------


def test_allocate_success(monkeypatch):
    csv_output = "0, 100, 8192, 5\n1, 200, 8192, 3\n"

    def mock_run(cmd, **kwargs):
        class R:
            returncode = 0
            stdout = csv_output

        return R()

    monkeypatch.setattr("subprocess.run", mock_run)

    pool = res.ResourcePool("nvidia", utilization_threshold=10)
    gpu_ids, ok = pool.allocate(1)
    assert ok is True
    assert len(gpu_ids) == 1
    assert gpu_ids[0] in (0, 1)


def test_allocate_insufficient_gpus(monkeypatch):
    csv_output = "0, 100, 8192, 5\n"

    def mock_run(cmd, **kwargs):
        class R:
            returncode = 0
            stdout = csv_output

        return R()

    monkeypatch.setattr("subprocess.run", mock_run)

    pool = res.ResourcePool("nvidia", utilization_threshold=10)
    gpu_ids, ok = pool.allocate(3)
    assert ok is False
    assert gpu_ids == []


def test_allocate_zero_gpus():
    pool = res.ResourcePool("unknown")
    gpu_ids, ok = pool.allocate(0)
    assert ok is True
    assert gpu_ids == []


def test_release_frees_gpus(monkeypatch):
    csv_output = "0, 100, 8192, 5\n1, 200, 8192, 3\n"

    def mock_run(cmd, **kwargs):
        class R:
            returncode = 0
            stdout = csv_output

        return R()

    monkeypatch.setattr("subprocess.run", mock_run)

    pool = res.ResourcePool("nvidia", utilization_threshold=10)
    gpu_ids, ok = pool.allocate(2)
    assert ok is True
    assert len(gpu_ids) == 2

    # All GPUs allocated; next allocation should fail.
    _, ok2 = pool.allocate(1)
    assert ok2 is False

    # Release one GPU.
    pool.release([gpu_ids[0]])
    gpu_ids2, ok3 = pool.allocate(1)
    assert ok3 is True
    assert gpu_ids2 == [gpu_ids[0]]


def test_allocate_excludes_allocated(monkeypatch):
    csv_output = "0, 100, 8192, 5\n1, 200, 8192, 3\n"

    def mock_run(cmd, **kwargs):
        class R:
            returncode = 0
            stdout = csv_output

        return R()

    monkeypatch.setattr("subprocess.run", mock_run)

    pool = res.ResourcePool("nvidia", utilization_threshold=10)
    gpu_ids1, _ = pool.allocate(1)
    gpu_ids2, _ = pool.allocate(1)

    assert gpu_ids1 != gpu_ids2
    assert set(gpu_ids1 + gpu_ids2) == {0, 1}


def test_thread_safety(monkeypatch):
    csv_output = "0, 0, 8192, 0\n1, 0, 8192, 0\n2, 0, 8192, 0\n3, 0, 8192, 0\n"

    def mock_run(cmd, **kwargs):
        class R:
            returncode = 0
            stdout = csv_output

        return R()

    monkeypatch.setattr("subprocess.run", mock_run)

    pool = res.ResourcePool("nvidia", utilization_threshold=50)
    allocated_all = []
    lock = threading.Lock()

    def allocate_one():
        ids, ok = pool.allocate(1)

        if ok:
            with lock:
                allocated_all.extend(ids)

    threads = [threading.Thread(target=allocate_one) for _ in range(4)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    assert len(allocated_all) == 4
    assert len(set(allocated_all)) == 4


# ---------------------------------------------------------------------------
# Tests for `get_status`.
# ---------------------------------------------------------------------------


def test_get_status(monkeypatch):
    csv_output = "0, 512, 8192, 5\n"

    def mock_run(cmd, **kwargs):
        class R:
            returncode = 0
            stdout = csv_output

        return R()

    monkeypatch.setattr("subprocess.run", mock_run)

    pool = res.ResourcePool("nvidia")
    status = pool.get_status()
    assert status["platform"] == "nvidia"
    assert len(status["gpus"]) == 1
    assert "system" in status


# ---------------------------------------------------------------------------
# Tests for `parse_gpu_requirement` and `parse_memory_requirement`.
# ---------------------------------------------------------------------------


def test_parse_gpu_requirement_auto_default():
    """`gpu_ids` omitted (defaults to `auto`) with `ngpus=1`."""
    job = {"resources": {"ngpus": 1}}
    assert res.parse_gpu_requirement(job) == 1


def test_parse_gpu_requirement_auto_explicit():
    """`gpu_ids=auto` with `ngpus=2`."""
    job = {"resources": {"gpu_ids": "auto", "ngpus": 2}}
    assert res.parse_gpu_requirement(job) == 2


def test_parse_gpu_requirement_auto_no_ngpus():
    """`gpu_ids=auto` without `ngpus` defaults to 1."""
    job = {"resources": {"gpu_ids": "auto"}}
    assert res.parse_gpu_requirement(job) == 1


def test_parse_gpu_requirement_auto_implicit_no_ngpus():
    """No `gpu_ids` and no `ngpus` defaults to 1."""
    job = {"resources": {}}
    assert res.parse_gpu_requirement(job) == 1


def test_parse_gpu_requirement_static_pinning():
    """Static `gpu_ids` counts explicit device IDs."""
    job = {"resources": {"gpu_ids": "0,1"}}
    assert res.parse_gpu_requirement(job) == 2


def test_parse_gpu_requirement_static_single():
    job = {"resources": {"gpu_ids": "0"}}
    assert res.parse_gpu_requirement(job) == 1


def test_parse_gpu_requirement_all():
    job = {"resources": {"gpu_ids": "all"}}
    assert res.parse_gpu_requirement(job) == 0


def test_parse_gpu_requirement_ngpus_mismatch_warns(capsys):
    """Warn when static `gpu_ids` count differs from `ngpus`."""
    job = {"resources": {"gpu_ids": "0,1", "ngpus": 3}}
    assert res.parse_gpu_requirement(job) == 2

    captured = capsys.readouterr()
    assert "warning:" in captured.err
    assert "ngpus=3" in captured.err


def test_parse_gpu_requirement_ignores_unknown_keys():
    """Unknown keys in resources do not affect GPU counting."""
    job = {"resources": {"gpu_ids": "0", "extra_key": "value"}}
    assert res.parse_gpu_requirement(job) == 1


def test_detect_gpus_ascend_hbm_parsing(monkeypatch):
    """`npu-smi` row 2 has DDR (0/0) and HBM (2789/32768); we want HBM."""
    npu_output = (
        "+---------------------------+---------------+-------------------------------+\n"
        "| 0     910B4               | OK            | 86.5  41                      |\n"
        "| 0                         | 0000:c1:00.0  | 5     0 / 0   2789 / 32768    |\n"
        "+---------------------------+---------------+-------------------------------+\n"
    )

    def mock_run(cmd, **kwargs):
        class R:
            returncode = 0
            stdout = npu_output

        return R()

    monkeypatch.setattr("subprocess.run", mock_run)

    pool = res.ResourcePool("ascend")
    gpus = pool.detect_gpus()
    assert len(gpus) == 1
    assert gpus[0].index == 0
    assert gpus[0].utilization_pct == 5.0
    assert gpus[0].memory_used_mb == 2789.0
    assert gpus[0].memory_total_mb == 32768.0


def test_parse_memory_requirement_gb():
    assert res.parse_memory_requirement({"resources": {"memory": "32GB"}}) == 32 * 1024


def test_parse_memory_requirement_mb():
    assert res.parse_memory_requirement({"resources": {"memory": "512MB"}}) == 512


def test_parse_memory_requirement_empty():
    assert res.parse_memory_requirement({"resources": {}}) == 0


def test_parse_memory_requirement_invalid_warns(capsys):
    result = res.parse_memory_requirement({"resources": {"memory": "abc xyz"}})
    assert result == 0

    captured = capsys.readouterr()
    assert "warning:" in captured.err
    assert "abc xyz" in captured.err
