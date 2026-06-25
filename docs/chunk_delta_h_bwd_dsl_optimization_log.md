# Chunk Delta H BWD DSL Optimization Log

This log tracks CuTeDSL `chunk_gated_delta_rule_bwd_dhu` optimization attempts.
The comparison target is FLA Triton; C++ CUTLASS/CuTe is used as a structural
reference when useful. Scope is SM90, fixed length, K=V=128.

## Baseline

- Branch: `delta_dsl`
- Baseline commit: `21aabab align dsl bwd bv64 prefetch pipeline`
- Kernel: `cula/ops/chunk_delta_h_bwd_dsl.py`
- Tests: `pytest -q tests/test_chunk_delta_h_bwd_dsl.py -s --tb=short`

Representative H800 timings from commit `21aabab`:

| Case | BV | FLA Triton ms | C++ ms | DSL ms | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| B=1,H=32,T=4096 | 64 | 0.2034 | 0.1781 | 0.1853 | BV64 rolling prefetch improved DSL from ~0.2118 |
| B=1,H=32,T=16384 | 64 | 0.7814 | 0.6890 | 0.7205 | DSL faster than Triton, still behind C++ |
| B=1,H=64,T=4096 | 64 | 0.3792 | 0.3528 | 0.3731 | H64 regressed versus pre-rolling-prefetch DSL |
| B=1,H=64,T=16384 | 64 | 1.5080 | 1.3305 | 1.3787 | H64 still faster than Triton but slower than C++ |

## Attempt Log

### BV64 2-stage rolling prefetch

- Change: added `_prefetch_bv64` and 2-stage K/Q/W/do/dv prefetch for BV64.
- Result: clear H=32 improvement.
- SASS effect: H=32 `LDGSTS` changed from 32 to 64, matching C++/Triton load count.
- Regression: H=64 worsened relative to the previous single-stage inline-load DSL.
- Next action: specialize BV64 by head count: keep 2-stage rolling prefetch for low-CTA
  cases such as H=32; use the older single-stage inline-load path for H=64.

### BV64 head-count-specialized prefetch

- Change: keep BV64 2-stage rolling prefetch for `H < 64`, but restore the single-stage
  inline cp.async path for `H == 64`.
- Rationale: H=32 is low-CTA enough to benefit from deeper rolling prefetch; H=64 has
  enough CTAs and the staged path's larger smem/indexing overhead hurts.
- Result: H=64 recovers while H=32 keeps the staged-prefetch benefit.

H800 timings:

| Case | BV | FLA Triton ms | C++ ms | DSL ms | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| B=1,H=32,T=4096 | 64 | 0.2041 | 0.1737 | 0.1768 | Close to C++; 1.15x Triton |
| B=1,H=32,T=16384 | 64 | 0.7747 | 0.6890 | 0.7170 | 1.08x Triton |
| B=1,H=32,T=32768 | 64 | 1.5201 | 1.3620 | 1.4289 | Long-T accuracy columns can be NaN due random-input overflow; timing valid |
| B=1,H=64,T=4096 | 64 | 0.3737 | 0.3578 | 0.3396 | Faster than C++ and Triton |
| B=1,H=64,T=16384 | 64 | 1.5100 | 1.3876 | 1.3365 | Faster than C++ and Triton |
| B=1,H=64,T=32768 | 64 | 3.0063 | 2.6218 | 2.6590 | Near C++; 1.13x Triton |

### Rejected: BV32 launch occupancy min_blocks_per_mp=1

- Change: tried replacing `min_blocks_per_mp=4 if BV==32 else 1` with `1`.
- Result: H=16/BV32 got slower, especially long T.
- Decision: keep `min_blocks_per_mp=4` for BV32.
