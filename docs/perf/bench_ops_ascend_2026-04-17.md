# Ascend Operator Correctness Verification — 2026-04-17

## Environment

| Item | Value |
|------|-------|
| Commit | `64c367c` — fix(ascend): prevent double-free in operator destructors at process exit |
| Branch | `feat/ascend-operators` (with unstaged style/format changes on `src/ascend/*/kernel*.h`) |
| Platform | Ascend 910B4 |
| Device | `davinci1` (via `ASCEND_RT_VISIBLE_DEVICES=0` in container) |
| Container | `infiniops-bench-ascend-1` (image `infiniops-ci/ascend:latest`) |
| npu-smi | 25.5.1 |
| Install | `infini` pre-installed at `/usr/local/python3.11.14/lib/python3.11/site-packages/infini` |

## Command

```bash
docker exec -e ASCEND_RT_VISIBLE_DEVICES=0 infiniops-bench-ascend-1 bash -lc \
  "cd /workspace && pytest tests/ --devices ascend --tb=short -q"
```

## Result

| Metric | Value |
|--------|-------|
| Passed | 2159 |
| Skipped | 1628 |
| Failed | 0 |
| Warnings | 2 (pytest cache on read-only `/workspace`, harmless) |
| Wall time | 19.39s |

**All Ascend operator correctness tests pass.** No failures across the full
parametrized matrix (operators × implementations × dtypes × shapes).

## Notes

- Performance benchmarks were intentionally skipped (user requested
  correctness only).
- Workspace was mounted read-only; pytest cache warnings are expected and
  do not affect results.
