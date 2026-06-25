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
- Result: smaller-head BV32 cases got slower, especially long T.
- Decision: keep `min_blocks_per_mp=4` for BV32. This was only a BV32 boundary
  check; current Triton-alignment work for `B=1,H=32,K=V=128` remains BV64.

### Rejected: BV64 FTZ update/accumulate path

- Change: used `add.rn.ftz.f32` / `fma.rn.ftz.f32` for BV64 `dv2` accumulation
  and recurrent state update.
- Tests: `pytest -q tests/test_chunk_delta_h_bwd_dsl.py -s --tb=short` passed.
- Result: no stable H=32 speedup. Representative H800 timings:

| Case | BV | FLA Triton ms | C++ ms | DSL ms | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| B=1,H=32,T=4096 | 64 | 0.2029 | 0.1748 | 0.1855 | No improvement over baseline |
| B=1,H=32,T=16384 | 64 | 0.7740 | 0.6863 | 0.7191 | Same as baseline |
| B=1,H=64,T=4096 | 64 | 0.3735 | 0.3569 | 0.3395 | Same as baseline |
| B=1,H=64,T=16384 | 64 | 1.5090 | 1.3299 | 1.3378 | Slightly worse than baseline |

- Decision: reject the FTZ change and keep the Triton-like non-FTZ FP32 update path.

### Rejected: remove BV64 dv2 store postbar

- Change: removed the final `bar.sync 0` from the 64x64 `dv2` store bridge.
- Tests: `pytest -q tests/test_chunk_delta_h_bwd_dsl.py -s --tb=short` passed.
- Result: `B=1,H=32,T=4096` showed one fast run around `0.1765 ms`, but a repeat
  with longer warmup/iters returned to `0.1863 ms`; long T regressed
  (`T=16384` around `0.7247 ms`, `T=32768` around `1.4373 ms`).
- Decision: reject. The post-store barrier appears useful for long-T scheduling
  stability even though correctness does not require it in short tests.

### BV64 dh store 64x64 bridge merge

- Change: for BV64 `dh` state stores, replace two 64x32 bridge-store calls with one
  64x64 bridge-store call. The dataflow is unchanged: the state is still staged
  for the following `k @ dh` MMA and also written to global `dh`.
- Rationale: current DSL bottleneck is not BV selection. For `B=1,H=32,K=V=128`,
  BV32 doubles CTAs but measured slower (`~0.242 ms` vs `~0.182 ms` at T=4096).
  The profitable local target is per-chunk store/barrier/address overhead inside
  the BV64 CTA.
- Tests: `pytest -q tests/test_chunk_delta_h_bwd_dsl.py -s --tb=short` passed.
- Result: clear H=32 long-T improvement and no material H=64 regression.

H800 timings after the change:

| Case | BV | FLA Triton ms | C++ ms | DSL ms | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| B=1,H=32,T=4096 | 64 | 0.2031 | 0.1755 | 0.1789 | Faster than baseline ~0.185 |
| B=1,H=32,T=16384 | 64 | 0.7748 | 0.6864 | 0.6878 | Near C++; baseline ~0.717 |
| B=1,H=32,T=32768 | 64 | 1.5279 | 1.3623 | 1.3637 | Near C++; baseline ~1.423 |
| B=1,H=64,T=4096 | 64 | 0.3783 | 0.3701 | 0.3407 | No material regression |
| B=1,H=64,T=16384 | 64 | 1.5114 | 1.3723 | 1.3403 | No material regression |
| B=1,H=64,T=32768 | 64 | 3.0099 | 2.6244 | 2.6670 | In previous noise range |

## Current Target

Optimize the current CuTeDSL kernel directly. For `B=1,H=32,K=V=128`, profiling
counters are unavailable in this environment due `ERR_NVGPUCTRPERM`, so use stable
timing plus SASS/resource inspection. Current evidence points to BV64 per-CTA
store/barrier overhead as the best local optimization area; BV32 is not profitable
for H=32 in the current DSL implementation.
