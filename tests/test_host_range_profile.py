import math
import os

import pytest

import infini.ops as ops


_PROFILE_API = (
    "_host_range_profile_compiled",
    "_host_range_profile_start",
    "_host_range_profile_stop",
    "_host_range_profile_calibrate",
)
_SUMMARY_KEYS = {
    "range",
    "count",
    "unit",
    "inclusive_mean",
    "inclusive_median",
    "self_mean",
    "self_median",
}
_CALIBRATION_RANGES = (
    "calibration.depth1",
    "calibration.depth2",
    "calibration.depth3",
)
_TIMING_KEYS = (
    "inclusive_mean",
    "inclusive_median",
    "self_mean",
    "self_median",
)
_EXPECT_PROFILE_ENV = "INFINI_OPS_EXPECT_HOST_RANGE_PROFILING"
_COLLECTOR_DRAIN_LIMIT = 8


def _restore_inactive_profile_collector():
    for _ in range(_COLLECTOR_DRAIN_LIMIT):
        try:
            ops._host_range_profile_stop()
        except RuntimeError:
            try:
                ops._host_range_profile_start()
            except RuntimeError:
                continue

            try:
                ops._host_range_profile_stop()
            except RuntimeError as error:
                pytest.fail(f"host-range profile collector cleanup failed: {error}")

            try:
                ops._host_range_profile_stop()
            except RuntimeError:
                return

    pytest.fail(
        "host-range profile collector did not become inactive after "
        f"{_COLLECTOR_DRAIN_LIMIT} cleanup attempts"
    )


@pytest.fixture(autouse=True)
def _isolate_profile_collector():
    if any(not callable(getattr(ops, name, None)) for name in _PROFILE_API):
        yield

        return

    compiled = getattr(ops, "_host_range_profile_compiled", None)
    if compiled() is not True:
        yield

        return

    _restore_inactive_profile_collector()
    yield
    _restore_inactive_profile_collector()


def _require_profile_build():
    missing = [name for name in _PROFILE_API if not callable(getattr(ops, name, None))]
    assert not missing, "infini.ops is missing host-range profiling API: " + ", ".join(
        missing
    )

    compiled = ops._host_range_profile_compiled()
    assert isinstance(compiled, bool)
    if not compiled:
        if os.environ.get(_EXPECT_PROFILE_ENV) == "1":
            pytest.fail(
                f"{_EXPECT_PROFILE_ENV}=1 but InfiniOps reports host-range "
                "profiling is not compiled"
            )

        pytest.skip("InfiniOps was built without host-range profiling")


def test_host_range_profile_start_rejects_an_active_collector():
    _require_profile_build()

    ops._host_range_profile_start()
    try:
        with pytest.raises(RuntimeError):
            ops._host_range_profile_start()
    finally:
        ops._host_range_profile_stop()


def test_host_range_profile_stop_rejects_an_inactive_collector():
    _require_profile_build()

    with pytest.raises(RuntimeError):
        ops._host_range_profile_stop()


def test_host_range_profile_calibration_reports_nested_summary_schema():
    _require_profile_build()
    iterations = 7

    rows = ops._host_range_profile_calibrate(iterations)

    assert [row["range"] for row in rows] == list(_CALIBRATION_RANGES)
    assert all(set(row) == _SUMMARY_KEYS for row in rows)
    assert all(row["count"] == iterations for row in rows)
    assert all(row["unit"] == "ns" for row in rows)

    timing_values = []
    for row in rows:
        for key in _TIMING_KEYS:
            value = row[key]
            assert isinstance(value, (int, float)) and not isinstance(value, bool)
            assert math.isfinite(value)
            assert value >= 0
            timing_values.append(value)

        assert row["inclusive_mean"] >= row["self_mean"] >= 0
        assert row["inclusive_median"] >= row["self_median"] >= 0

    assert any(value > 0 for value in timing_values)

    depth1, depth2, depth3 = rows
    assert depth1["inclusive_mean"] == pytest.approx(
        depth1["self_mean"] + depth2["inclusive_mean"]
    )
    assert depth2["inclusive_mean"] == pytest.approx(
        depth2["self_mean"] + depth3["inclusive_mean"]
    )
    assert depth3["inclusive_mean"] == depth3["self_mean"]
    assert depth3["inclusive_median"] == depth3["self_median"]

    inclusive_medians = [row["inclusive_median"] for row in rows]
    assert inclusive_medians == sorted(inclusive_medians, reverse=True)
