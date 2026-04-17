torch: 2.9.0+cpu
device: npu:0  dtype: torch.float16

=== Probe: linear (gemm) ===
  replay#1 (same as warm inputs):  ok=True  max_abs_diff=0
  replay#2 (different inputs):      ok=True  max_abs_diff=0
  RESULT: PASS

=== Probe: rms_norm ===
  replay#1 (same as warm inputs):  ok=True  max_abs_diff=0.0004883
  replay#2 (different inputs):      ok=True  max_abs_diff=0
  RESULT: PASS

=== Probe: silu_and_mul ===
  replay#1 (same as warm inputs):  ok=True  max_abs_diff=0.007812
  replay#2 (different inputs):      ok=True  max_abs_diff=0.003906
  RESULT: PASS

=== Probe: apply_rotary_pos_emb ===
  replay#1 (same as warm inputs):  ok=True  max_abs_diff=0
  replay#2 (different inputs):      ok=True  max_abs_diff=0
  RESULT: PASS

=== SUMMARY ===
  linear                        PASS
  rms_norm                      PASS
  swiglu                        PASS
  apply_rotary_pos_emb          PASS
EXIT=0
