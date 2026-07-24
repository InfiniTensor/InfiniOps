# Host Range Profiling Experiment Design

## Goal

Determine whether explicit host-side ranges can explain InfiniOps operator-library
overhead during the existing pytest benchmark flow without sampling profilers or
device execution timing.

## Scope

The experiment covers two NVIDIA cases:

- Add: contiguous FP32 `(13, 4)`, native implementation 0.
- GEMM: FP32 `(4, 48, 64) x (4, 64, 6)`, cuBLASLt implementation 1.

It adds an opt-in host range collector and instruments only the common path plus
the two selected backend submission paths. It does not add CI, performance
gates, a new C++ test framework, synchronization inside measured windows,
kernel timing, or all-operator/backend coverage.

## Architecture

`INFINI_OPS_ENABLE_HOST_RANGE_PROFILING` is disabled by default. When disabled,
the range macro expands to nothing. When enabled, a `thread_local` collector
uses `std::chrono::steady_clock` and a nested stack to record inclusive duration
and direct-child duration for each fixed range layer. Self duration is inclusive
duration minus direct-child duration.

The hot path uses a fixed enum rather than dynamic strings or hash-table lookups.
The collector stores per-event integer nanosecond samples so it can compute an
exact count, arithmetic mean, and median when a profiling window ends.

The generated Python extension exposes private experiment controls:

```text
_host_range_profile_compiled()
_host_range_profile_start()
_host_range_profile_stop()
_host_range_profile_calibrate(iterations)
```

The pytest hook starts and stops the collector around InfiniOps work only. It
does not collect the reference implementation. The hook writes one JSON-lines
record per `(pytest case, phase, range, metric)`.

## Range Hierarchy

```text
binding.body
|-- binding.tensor_conversion
|-- binding.device_conversion
`-- dispatch.call
    `-- operator.call
        |-- cache.key
        |-- cache.lookup
        |-- cache.construct       # cache miss only
        `-- operator.invoke
            `-- backend.submit    # host API return is the range end
```

`backend.submit` includes CPU-side CUDA/cuBLASLt setup, submission, and
mandatory cleanup through host API return. It does not synchronize the stream
or directly time kernel execution. A host API may still block because of queue
backpressure or device progress, so the range is host-side API latency rather
than a device-independent CPU-only quantity.

The generated pybind lambda begins after part of pybind11's own function-dispatch
machinery. Therefore `binding.body` is not labelled as total pybind overhead.
The profiling path passes `time.perf_counter` to
`torch.utils.benchmark.Timer`, avoiding its accelerator-synchronizing default
timer. The resulting `end_to_end` row is host call/submission time rather than
direct kernel execution time. It and the replayed C++ ranges are collected from
separate populations, so their difference remains unattributed and must not be
assigned to Python, pybind11, or another layer by subtraction.

## Cold And Warm Measurements

The existing correctness call warms the operator cache before benchmarking. For
profiling, pytest performs two separate windows:

1. Clear the selected operator cache, start profiling, invoke once, and stop.
   This is the cold phase and includes one cache construction.
2. Run `blocked_autorange()` with the host-only timer while the collector is
   inactive, synchronize outside the measured window, then replay exactly its
   formal call count inside one profiling window. Stop the collector before a
   second outside-window synchronization. This keeps estimation calls and
   pending device work out of the warm range population while retaining many
   calls suitable for mean/median statistics.

The reference timer uses the same host-only timer after the collector is stopped.
Cache invalidation is lazy, so destruction of an old cached operator can affect
the enclosing cold `binding.body` and `dispatch.call` inclusive samples. The
single cold sample is diagnostic and is not used as a stable numeric baseline.

## Output

Each JSON-lines row has stable unit-independent metric names:

```json
{
  "nodeid": "tests/test_add.py::test_add[...]",
  "operator": "add",
  "backend": "nvidia",
  "phase": "warm",
  "range": "cache.lookup",
  "metric": "self",
  "count": 1000,
  "unit": "ns",
  "mean": 83.2,
  "median": 81.0
}
```

Inclusive and self measurements are separate rows. This avoids encoding the
unit or metric kind into `mean` and `median` field names.

## Observer-Effect Calibration

The experiment records three controls:

- profiling disabled: a host-only control script using `time.perf_counter`;
- profiling enabled but inactive: the same host-only control script;
- profiling enabled with fixed empty nested ranges.

The final report distinguishes the duration recorded inside an empty leaf from
the complete observer cost visible to its parent and from end-to-end active
collector perturbation. A layer is not treated as a stable numeric baseline
when the conservative complete-scope cost is at least 10% of that layer's
median. Parent inclusive ranges additionally contain the observer cost of all
children, so the end-to-end active control remains the primary perturbation
bound. Observer-sensitive layers must be merged with their parent or treated
only as diagnostic trace data.

## Error Handling

- Requesting `--host-range-profile` from a build compiled without the option is
  a pytest usage error.
- Starting an already-active collector or stopping an inactive collector is an
  error surfaced through the private Python binding.
- Stopping with unclosed ranges is an error; incomplete data is not emitted.
- The first experiment supports only the pytest main thread. Events from other
  threads are not silently merged into the case.

## Acceptance Criteria

- Default builds compile with the range macro erased from instrumented paths.
- Existing generator and report tests pass.
- A profiling-enabled NVIDIA build passes the exact Add and GEMM selectors.
- The JSON-lines report contains cold and warm data with count, unit, mean, and
  median for the expected nested host layers.
- `cache.construct` appears in cold data and not in warm data.
- Device synchronization is used only outside profiling/timing windows to
  isolate pending work; none is added inside a measured path.
- The report quantifies empty-range and end-to-end measurement perturbation.
