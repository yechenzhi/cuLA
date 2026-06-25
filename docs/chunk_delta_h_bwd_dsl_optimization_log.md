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

### Rejected: separate BV64 dh bridge smem plus no-postbar

- Change: for BV64, allocated a separate bridge smem tile for the second K block
  `dh` store, removed the final postbar from the 64x64 bridge asm, and preserved
  `dv2` store synchronization with an explicit `sync_threads()`.
- Rationale: try reducing CTA barrier count in the two `dh` store bridge calls.
- Tests: `pytest -q tests/test_chunk_delta_h_bwd_dsl.py -s --tb=short` passed.
- Result: no speedup; H32 was slightly worse or noise-level equal:
  `T=4096/16384/32768` DSL around `0.1793/0.6901/1.3695 ms`.
- Decision: reject. The extra BV64 shared memory is not worth the removed postbar.

### Rejected: BV64 H<64 prefetch distance 3

- Change: for BV64 with `H < 64`, changed the rolling K/Q/W/do/dv prefetch ring
  from 2 stages to 3 stages and adjusted `cp.async.wait_group`.
- Tests: `pytest -q tests/test_chunk_delta_h_bwd_dsl.py -s --tb=short` passed.
- Result: mixed. H32 long T improved slightly (`T=16384/32768` around
  `0.6803/1.3477 ms`), but short T regressed (`T=4096` around `0.1810 ms`).
  H64 was effectively unchanged because it uses the single-stage inline path.
- Decision: reject as a standalone change. Revisit stage depth only inside the
  planned warp-specialized TMA block, where producer/consumer overlap is
  structurally different.

### Rejected: standalone TMA-only RMEM-state probe

- Change: added an isolated experimental TMA-only BV64 path in
  `cula/ops/chunk_delta_h_bwd_dsl_tma.py`. It kept a 128-thread CTA and no
  cluster/warp-specialized load-store warps, but borrowed the `feat_bwd_delta`
  RMEM-state KDH orientation and TMA S2G store structure.
- Fixes made during the probe:
  - GMEM TMA V dimensions must use global `V=128`, not tile `BV=64`.
  - `dht` reusing `sDh` requires enough backing storage; a single bf16 state
    tile is too small for fp32 dht.
  - TMA load pipeline consumer counts must follow the reference pattern
    (`num_compute_warps`, not `num_compute_threads`), otherwise stage reuse
    deadlocks after the first two chunks.
  - K/Q/W must be split into BK64 halves to match the current DSL accumulation
    structure.
- Correctness result: short cases pass or nearly pass, but long T diverges.
  Representative H32 max errors against FLA with h0+dht:
  `T=512 -> dh/dh0/dv2 max 0.0156/0.0278/0.0169`,
  `T=1024 -> 0.375/0.550/0.438`,
  `T=4096 -> O(1e6)`.
- Performance result before the BK64 correction, for H32/H64 T=4096, was also
  slower than the current DSL (`~0.239/0.509 ms` vs `~0.178/0.341 ms`).
- Decision: reject this as a valid TMA-only optimization. It changed more than
  the transport path: the KDH operand orientation/store-read path no longer
  matched the current DSL/Triton-sensitive recurrence. The next TMA-only attempt
  must preserve the current DSL math path: R2S state into the existing
  state-read layout, KDH as `K @ dh`, and only replace the global-memory
  movement/store mechanism.

### Rejected: current-math BV64 TMA-only without warp specialization

- Change: kept the current DSL math path and 128-thread CTA shape, but replaced
  BV64 global load/store paths with TMA G2S/S2G in
  `cula/ops/chunk_delta_h_bwd_dsl_tma.py`.
- Correctness fixes:
  - `dv2` global TMA view must be `(T,V)` and the store coordinate is
    `(chunk_idx, v_tile_idx)`.
  - State and `dv2` shared-memory write/read paths must preserve the current
    K_SW128 store plus MN_SW128 read pairing.
  - QDO/WDV WGMMA issue order must match the default DSL exactly.
  - The recurrent update must use the same in-place `+=` expression as the
    default DSL; an expanded assignment produced 1-ulp differences that were
    amplified by long-T recurrence.
- Correctness result: after those fixes, H32/H64 with h0+dht are bitwise equal
  to the default DSL and FLA through `T=4096`.
- Performance result: slower than the current cp.async/bridge DSL.

H800 timings, h0+dht, `B=1,K=V=128`:

| Case | Current DSL ms | TMA full ms | TMA load-only ms | Finding |
| --- | ---: | ---: | ---: | --- |
| H=32,T=4096 | 0.1734 | 0.2304 | 0.1934 | S2G store/wait costs ~0.037 ms |
| H=32,T=16384 | 0.6820 | 0.8881 | 0.7229 | S2G store/wait costs ~0.165 ms |
| H=64,T=4096 | 0.3332 | 0.5011 | 0.3895 | S2G store/wait costs ~0.112 ms |
| H=64,T=16384 | 1.3398 | 2.0679 | 1.4832 | S2G store/wait costs ~0.585 ms |

SASS/resource evidence:

| Kernel | REG | Key static instructions |
| --- | ---: | --- |
| DSL H32 BV64 | 224 | `LDGSTS=64`, `BAR.SYNC=11`, `BRA=8` |
| TMA full H32 BV64 | 200 | `UTMALDG=10`, `UTMASTG=5`, `SYNCS.PHASECHK=36`, `BAR.SYNC=12`, `BRA=50` |
| TMA load-only H32 BV64 | 212 | `UTMALDG=10`, `SYNCS.PHASECHK=36`, `BAR.SYNC=8`, `BRA=46` |
| DSL H64 BV64 | 224 | `LDGSTS=32`, `BAR.SYNC=11`, `BRA=4` |
| TMA full H64 BV64 | 200 | same TMA control structure as H32 |

- Diagnosis: TMA itself is not the issue. The issue is using TMA inside the same
  4 compute warps with stage-1 immediate waits. The TMA path replaces many
  `LDGSTS`/store instructions, but introduces mbarrier wait loops
  (`SYNCS.PHASECHK`) and extra branches. The S2G stores are especially expensive
  because every chunk does `dh` and `dv2` TMA stores followed by
  `UTMACMDFLUSH/DEPBAR/BAR.SYNC` before compute continues.
- Decision: do not roll back the TMA experiment; keep it as the validated layout
  reference. Do not wire this 128-thread TMA-only path into the public kernel.
  The next TMA work must add a real store/load producer role so G2S/S2G waits are
  overlapped, otherwise TMA cannot beat the current cp.async bridge path.

### Rejected: BV64 TMA store-warp attribution probe

- Change: added an isolated 5-warp attribution kernel in
  `cula/ops/chunk_delta_h_bwd_dsl_tma_store_warp.py`. It kept the validated
  current-math TMA load/store layouts, kept the four compute warps, and moved
  TMA S2G stores to a dedicated store warp. Loads still come from compute warp
  0, so this is not the final 7-warp design.
- Correctness result: H32 short h0+dht is bitwise equal to the default DSL for
  `dh`, `dh0`, and `dv2`.
- Store staging result:
  - single store stage was effectively the same as TMA full, because the next
    chunk immediately reacquired the same smem stage and waited for store-warp
    release;
  - the first two-stage probe only checked `T=64`; longer sequences were wrong
    because KDH/WDV still read fixed stage `0`;
  - after fixing KDH/WDV to read the dynamic store stage, `T=4096` is bitwise
    equal to the default DSL;
  - the correct two-stage chunk-level handoff did not improve speed;
  - replacing hand-written `cp_async_bulk_commit_group/wait_group` with
    `PipelineTmaStore` helped some store-warp timings, but still left the kernel
    far slower than the cp.async DSL.

H800 timings, h0+dht, `B=1,K=V=128`, `PipelineTmaStore` variant:

| Case | Current DSL ms | TMA load-only ms | TMA full ms | Store-warp ms |
| --- | ---: | ---: | ---: | ---: |
| H=32,T=4096 | 0.1789 | 0.1931 | 0.2333 | 0.2217 |
| H=32,T=16384 | 0.6792 | 0.7256 | 0.8972 | 0.8527 |
| H=64,T=4096 | 0.3352 | 0.3868 | 0.5057 | 0.4738 |
| H=64,T=16384 | 1.3373 | 1.4811 | 2.0698 | 1.9341 |

SASS/resource evidence for H32 BV64:

| Kernel | REG | Key static instructions |
| --- | ---: | --- |
| DSL | 224 | `LDGSTS=64`, `BAR.SYNC=11`, `BRA=8`, `DEPBAR=10` |
| TMA full | 200 | `UTMALDG=10`, `UTMASTG=5`, `SYNCS.PHASECHK=36`, `BRA=50`, `UTMACMDFLUSH=3` |
| TMA load-only | 212 | `UTMALDG=10`, `SYNCS.PHASECHK=36`, `BRA=46` |
| TMA store-warp | 224 | `UTMALDG=10`, `UTMASTG=17`, `SYNCS.PHASECHK=78`, `BRA=99`, `UTMACMDFLUSH=11` |

Correct chunk-level store handoff follow-up:

| Case | Current DSL ms | TMA load-only ms | TMA full ms | Chunk store-warp ms |
| --- | ---: | ---: | ---: | ---: |
| H=32,T=4096 | 0.1730 | 0.1932 | 0.2327 | 0.2358 |
| H=32,T=16384 | 0.6786 | 0.7245 | 0.8990 | 0.9052 |
| H=64,T=4096 | 0.3361 | 0.3861 | 0.5099 | 0.4923 |
| H=64,T=16384 | 1.3392 | 1.4815 | 2.0708 | 2.0102 |

SASS for the correct chunk-level handoff improves but remains too heavy:
`SYNCS.PHASECHK=60`, `BRA=80`, `UTMACMDFLUSH=6`, `UTMASTG=17`.
That confirms the coarse handoff reduces some mbarrier/control overhead, but the
combined TMA-store path is still not close to the cp.async bridge or even to
TMA load-only.

- Diagnosis: this probe did not create the Hopper-style lightweight producer
  warp. It moved the TMA S2G instructions out of the compute branch, but added
  three generic compute-to-store `PipelineAsync` pipelines (`dh`, `dv2`, `dh0`)
  plus TMA-store queue management. The generated control flow has much more
  `SYNCS.PHASECHK`, branch, and TMA-store command/flush code than both the
  128-thread TMA-only probe and the cp.async DSL. Therefore the current gap is
  not because TMA descriptors or smem layouts are wrong; it is because the TMA
  schedule is not actually hiding mbarrier/store waits.
- Decision: keep this file only as an attribution reference. The next TMA work
  should not add more independent per-output store pipelines. It should use one
  coarse chunk-level handoff, `PipelineTmaStore` for S2G completion, and separate
  load producer warps so G2S mbarrier waits are overlapped with WGMMA.

### Attribution: BV64 TMA load warp and sync store warp

- Change: added isolated attribution kernels:
  - `cula/ops/chunk_delta_h_bwd_dsl_tma_load_warp.py`: 4 compute warps plus a
    dedicated load warp. The load warp issues all TMA G2S loads and participates
    in the existing CTA sync points. Stores remain inline TMA S2G from compute
    warp 0.
  - `cula/ops/chunk_delta_h_bwd_dsl_tma_ldst_warp.py`: adds a dedicated store
    warp that issues TMA S2G between the same CTA sync points. This avoids the
    earlier generic store `PipelineAsync` handoff.
- Correctness: both full variants are bitwise equal to the default DSL for H32
  with h0+dht through `T=4096`.
- Result: moving G2S issue out of compute warp 0 is useful. Moving S2G issue out
  of compute warp 0 without hiding the S2G wait is only a small improvement.

H800 timings, h0+dht, `B=1,K=V=128`:

| Case | DSL ms | TMA load-only ms | LD/ST load-only ms | TMA full ms | Load-warp full ms | LD/ST full ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H=32,T=4096 | 0.1784 | 0.1932 | 0.1788 | 0.2333 | 0.2214 | 0.2189 |
| H=32,T=16384 | 0.6825 | 0.7242 | 0.6698 | 0.8984 | 0.8460 | 0.8350 |
| H=64,T=4096 | 0.3353 | 0.3845 | 0.3519 | 0.5070 | 0.4833 | 0.4757 |
| H=64,T=16384 | 1.3388 | 1.4786 | 1.3511 | 2.0701 | 1.9640 | 1.9328 |

SASS/resource, H32 BV64:

| Kernel | REG | Key static instructions |
| --- | ---: | --- |
| TMA load-only | 212 | `UTMALDG=10`, `SYNCS.PHASECHK=36`, `BRA=46` |
| Load-warp load-only | 230 | `UTMALDG=26`, `SYNCS.PHASECHK=66`, `BRA=77`, `BAR.SYNC=14` |
| Load-warp full | 205 | `UTMALDG=26`, `UTMASTG=5`, `SYNCS.PHASECHK=66`, `BAR.SYNC=26` |
| LD/ST full | 206 | `UTMALDG=26`, `UTMASTG=17`, `SYNCS.PHASECHK=66`, `BAR.SYNC=48`, `UTMACMDFLUSH=11` |

- Diagnosis:
  - Static SASS looks heavier for the load-warp variants, but timing improves
    because the four compute warps no longer spend their hot path issuing G2S
    TMA copies.
  - The remaining full-kernel gap is dominated by S2G store wait/flush. The
    LD/ST sync-warp version moves store issue to a separate warp but still waits
    at the same CTA syncs, so it only improves full TMA by a few percent.
  - The earlier store `PipelineAsync` version was worse because it added mbarrier
    state machines. The sync store-warp version proves that merely changing the
    issuing warp is not enough either.
- Decision: keep the load-warp structure as the next useful base. The next
  prototype should preserve the dedicated load warp, then redesign store
  scheduling so `dh/dv2` S2G waits overlap with later WGMMA work. Do not spend
  more time on store-warp variants that keep a blocking sync immediately after
  every TMA S2G.

### Attribution: BV64 delayed TMA S2G wait

- Change: added isolated LD/ST-overlap probes:
  - `cula/ops/chunk_delta_h_bwd_dsl_tma_ldst_overlap.py`: 4 compute warps plus
    dedicated load/store warps, with two staged `sDh/sDv2` buffers. The store
    warp issues `dh/dv2` TMA S2G and delays `cp_async_bulk_wait_group` until the
    same store stage is about to be reused.
  - `cula/ops/chunk_delta_h_bwd_dsl_tma_ldst_overlap_s3.py`: same design with
    three store stages.
- Correctness: both full variants are bitwise equal to the default DSL for H32
  with h0+dht through `T=4096`.
- Result: delayed S2G wait helps, especially at H64 long T, but it does not fix
  the core gap because the generated TMA-store command/flush path remains large.

H800 timings, h0+dht, `B=1,K=V=128`:

| Case | DSL ms | TMA full ms | LD/ST sync ms | LD/ST overlap s2 ms | LD/ST overlap s3 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| H=32,T=4096 | 0.1732 | 0.2332 | 0.2197 | 0.2183 | 0.2171 |
| H=32,T=16384 | 0.6823 | 0.8982 | 0.8342 | 0.8268 | 0.8278 |
| H=64,T=4096 | 0.3330 | 0.5040 | 0.4697 | 0.4614 | 0.4563 |
| H=64,T=16384 | 1.3382 | 2.0669 | 1.9342 | 1.8737 | 1.8755 |

SASS/resource, H32 BV64:

| Kernel | REG | Key static instructions |
| --- | ---: | --- |
| LD/ST sync full | 206 | `BAR.SYNC=48`, `BRA=86`, `DEPBAR=15`, `SYNCS.PHASECHK=66`, `UTMACMDFLUSH=11`, `UTMALDG=26`, `UTMASTG=17` |
| LD/ST overlap s2 full | 206 | `BAR.SYNC=39`, `BRA=89`, `DEPBAR=11`, `SYNCS.PHASECHK=66`, `UTMACMDFLUSH=11`, `UTMALDG=26`, `UTMASTG=17` |
| LD/ST overlap s3 full | 206 | `BAR.SYNC=39`, `BRA=90`, `DEPBAR=11`, `SYNCS.PHASECHK=66`, `UTMACMDFLUSH=11`, `UTMALDG=26`, `UTMASTG=17` |

- Diagnosis:
  - The stage-2 delayed wait reduces blocking barriers/dependency barriers and
    improves H64,T=16384 by about 3% versus the sync LD/ST store-warp path.
  - Stage 3 is not materially better than stage 2, so the remaining time is not
    just "need a deeper store buffer".
  - `UTMASTG=17` and `UTMACMDFLUSH=11` remain unchanged across sync and overlap
    variants. That points at the S2G TMA store mechanism and command/flush
    granularity itself, not only the exact wait placement.
- Decision: keep the two-stage delayed-wait file as the best TMA-S2G attribution
  reference. Do not wire it into the public kernel because it is still slower
  than the cp.async/bridge DSL. The next useful probe should keep the successful
  dedicated TMA load warp but replace only the output store path with the current
  cp/bridge store path, or otherwise reduce the number of S2G commands/flushes.

### Attribution: BV64 TMA load plus bridge store

- Change: tested a temporary hybrid probe that kept the dedicated TMA load warp
  but used the current DSL bridge/global-store path for `dh/dv2/dh0`.
- Correctness: bitwise equal to the default DSL for H32/H64 with h0+dht through
  `T=4096`; also bitwise for all h0/dht enable combinations at `T=512`.
- Result: this confirmed that TMA S2G is the largest gap, but the hybrid is not
  the main implementation direction because it copies inline bridge-store asm and
  violates the current preference for a TMA-first, simpler kernel.

H800 timings, h0+dht, `B=1,K=V=128`:

| Case | DSL ms | TMA load-warp full ms | TMA-load + bridge-store ms | TMA-load s2 + compute-bar bridge ms |
| --- | ---: | ---: | ---: | ---: |
| H=32,T=4096 | 0.1764 | 0.2231 | 0.1877 | 0.1600 |
| H=32,T=16384 | 0.6836 | 0.8371 | 0.7091 | 0.6315 |
| H=64,T=4096 | 0.3321 | 0.4763 | 0.4166 | 0.3688 |
| H=64,T=16384 | 1.3380 | 1.9637 | 1.6723 | 1.3861 |

SASS/resource for the best H32 hybrid:

| Kernel | REG | Key static instructions |
| --- | ---: | --- |
| TMA-load s2 + compute-bar bridge | 222 | `UTMALDG=26`, `UTMASTG=0`, `UTMACMDFLUSH=0`, `SYNCS.PHASECHK=66`, `BAR.SYNC=16`, `BRA=78`, `STG=28` |

- Diagnosis:
  - Removing TMA S2G removes `UTMASTG/UTMACMDFLUSH` and makes H32 faster than
    the default DSL.
  - H64 remains slower because TMA G2S still carries `SYNCS.PHASECHK=66` and a
    much heavier control path than the cp.async DSL.
  - The compute-bar bridge version needs a private copy of the inline asm and is
    therefore rejected as the production direction despite the H32 speedup.
- Decision: do not wire this into the public DSL path. Keep the result as
  attribution only: it proves the performance issue is concentrated in TMA S2G
  command/flush plus TMA load control overhead, not in math or layout.

### Rejected: BV64 all-TMA input load stage 2

- Change: tested a TMA-first load-warp variant with two G2S stages for
  `K/Q/W/Do/Dv`; output stores remained TMA S2G.
- Correctness: bitwise equal to the default DSL for H32/H64 with h0+dht through
  `T=4096`.
- Result: slower than the one-stage load-warp and slower than the delayed-wait
  LD/ST overlap path.

H800 timings, h0+dht, `B=1,K=V=128`:

| Case | DSL ms | TMA load-warp s1 ms | TMA load-warp s2 ms | LD/ST overlap ms |
| --- | ---: | ---: | ---: | ---: |
| H=32,T=4096 | 0.1740 | 0.2209 | 0.2288 | 0.2164 |
| H=32,T=16384 | 0.6823 | 0.8366 | 0.8670 | 0.8283 |
| H=64,T=4096 | 0.3316 | 0.4788 | 0.4980 | 0.4581 |
| H=64,T=16384 | 1.3364 | 1.9613 | 2.0227 | 1.8764 |

- Diagnosis: adding input stages without fixing TMA S2G increases shared-memory
  footprint and mbarrier/control pressure but does not hide the dominant store
  cost.
- Decision: reject all-TMA input stage 2 as a standalone optimization. The
  TMA-first path should next simplify/reduce the S2G store command path or find
  a cleaner way to overlap store completion; do not keep adding input stages.

### Rejected: BV64 K128 dh/dh0 TMA stores

- Change: tested a TMA-first load-warp variant that stores `dh` and `dh0` as one
  K128-by-BV TMA S2G tile instead of two BK64-by-BV TMA stores. The math path and
  WGMMA read views still use two BK64 halves backed by one contiguous shared
  buffer.
- Correctness: bitwise equal to the default DSL for H32/H64 with h0+dht through
  `T=4096`.
- Result: static SASS confirms fewer TMA store instructions, but timing is
  unchanged versus the original TMA load-warp full kernel.

H800 timings, h0+dht, `B=1,K=V=128`:

| Case | DSL ms | TMA load-warp ms | LD/ST overlap ms | K128-store ms |
| --- | ---: | ---: | ---: | ---: |
| H=32,T=4096 | 0.1770 | 0.2219 | 0.2183 | 0.2229 |
| H=32,T=16384 | 0.6835 | 0.8371 | 0.8280 | 0.8372 |
| H=64,T=4096 | 0.3340 | 0.4836 | 0.4591 | 0.4855 |
| H=64,T=16384 | 1.3434 | 1.9631 | 1.8728 | 1.9642 |

SASS/resource, H32 BV64:

| Kernel | REG | Key static instructions |
| --- | ---: | --- |
| K128-store full | 205 | `UTMALDG=26`, `UTMASTG=3`, `UTMACMDFLUSH=3`, `SYNCS.PHASECHK=66`, `BAR.SYNC=26`, `BRA=84` |

- Diagnosis:
  - Combining `dh/dh0` reduces static `UTMASTG` from the load-warp full path, so
    the descriptor/layout is valid.
  - Runtime does not improve because the loop still blocks on the same TMA S2G
    completion points, and `dv2` remains a per-chunk TMA store required before
    WDV.
  - The remaining hot cost is synchronization/wait placement around `dh/dv2`,
    not the number of `dh/dh0` descriptors alone.
- Decision: reject as a standalone optimization. Keep the idea only if combined
  with a real delayed-wait/store-overlap schedule; do not wire K128 stores alone
  into the public path.

### Rejected: BV64 TMA cluster multicast for K/Q/W

- Change: added an attribution-only `chunk_delta_h_bwd_dsl_tma_load_warp_mcast.py`
  probe. It keeps the existing TMA load-warp math/store path, but launches the
  two BV64 V tiles as a `(2,1,1)` CTA cluster and uses
  `CopyBulkTensorTileG2SMulticastOp` for `K/Q/W`. `dv/do/dht/dh/dv2/dh0` remain
  non-multicast so the measurement isolates shared `K/Q/W` G2S traffic.
- Correctness: bitwise equal to the default DSL for H32/H64 with h0+dht through
  `T=16384`. A cluster-exit `cluster_arrive/cluster_wait` is required; without
  it the load-only variant can hit an async launch failure after return.
- Result: multicast gives a small full-path improvement only at long T versus
  the non-multicast TMA load-warp, but it is still far slower than the public
  cp.async/bridge DSL. In load-only mode multicast is consistently slower, which
  means the extra cluster/mcast pipeline control cost is larger than the removed
  duplicate `K/Q/W` G2S issue in this block design.

H800 timings, h0+dht, `B=1,K=V=128`, warmup=3, iters=20:

| Case | DSL ms | TMA load-warp ms | TMA mcast ms | TMA load-only ms | TMA mcast load-only ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| H=32,T=4096 | 0.1787 | 0.2148 | 0.2169 | 0.1697 | 0.1788 |
| H=32,T=16384 | 0.6879 | 0.8227 | 0.8170 | 0.6396 | 0.6692 |
| H=64,T=4096 | 0.3436 | 0.4770 | 0.4806 | 0.3449 | 0.3571 |
| H=64,T=16384 | 1.3361 | 1.9425 | 1.9086 | 1.2833 | 1.3222 |

- Diagnosis:
  - TMA multicast itself is valid and can reduce enough work to slightly improve
    the long-T full TMA path.
  - The load-only regression shows `K/Q/W` duplicate G2S traffic is not the
    dominant issue. The dominant cost remains TMA pipeline/cluster control and
    the TMA S2G store/wait path.
  - This also explains why "TMA has no benefit" is too broad: TMA descriptors and
    multicast work, but the current fine-grained per-chunk usage does not earn
    back its synchronization cost.
- Decision: do not wire multicast into the public DSL. Keep the file as an
  attribution probe only if it helps compare against a future simpler
  warp-specialized TMA block; otherwise remove before a production cleanup.

### Rejected: BV64 split cp.async wait for the second load group

- Change: tested a public-DSL BV64 wait-placement change. The original kernel
  issues two `cp.async` groups per chunk and waits for both before KDH. The probe
  used `wait_group(1)` before KDH so only the first group (`K0/K1/Q0/W0`) must be
  complete, then waited for the second group (`Q1/W1/Do/Dv`) after KDH and before
  reading `dv/do/q/w`.
- Correctness: bitwise equal to the original DSL for H32/H64 with h0+dht at
  `T=512`.
- Result:
  - H32/BV64 staged-prefetch variant regressed long T and was rejected
    immediately.
  - H64-only variant gave small wins at `T=4096/16384`, but consistently
    regressed `T=32768`, which is explicitly in the target range.
  - A dynamic `T <= 16384` gate removed most short-T benefit and still left a
    small long-T regression from the extra branch/control path.

H800 repeated timings, h0+dht, `B=1,H=64,K=V=128`:

| Case | Baseline ms | Split wait ms | Speedup |
| --- | ---: | ---: | ---: |
| T=4096 rep0/1/2 | 0.3407 / 0.3394 / 0.3415 | 0.3360 / 0.3381 / 0.3385 | 1.014 / 1.004 / 1.009 |
| T=16384 rep0/1/2 | 1.3381 / 1.3426 / 1.3407 | 1.3359 / 1.3346 / 1.3327 | 1.002 / 1.006 / 1.006 |
| T=32768 rep0/1/2 | 2.6557 / 2.6658 / 2.6631 | 2.6633 / 2.6825 / 2.6777 | 0.997 / 0.994 / 0.995 |

Dynamic gate timings:

| Case | Baseline ms | Gated split ms | Speedup |
| --- | ---: | ---: | ---: |
| T=4096 rep0/1/2 | 0.3384 / 0.3370 / 0.3406 | 0.3355 / 0.3381 / 0.3399 | 1.009 / 0.997 / 1.002 |
| T=16384 rep0/1/2 | 1.3375 / 1.3421 / 1.3378 | 1.3366 / 1.3406 / 1.3428 | 1.001 / 1.001 / 0.996 |
| T=32768 rep0/1/2 | 2.6901 / 2.6855 / 2.6930 | 2.7057 / 2.6911 / 2.6993 | 0.994 / 0.998 / 0.998 |

- Diagnosis: the second group can overlap with KDH for short sequences, but the
  extra mid-loop wait/sync and dynamic control path are not free. At long T the
  kernel is dominated by store/update work, and the additional synchronization
  loses more than the load overlap saves.
- Decision: reject and leave the public kernel unchanged. Do not retry this
  exact split-wait schedule unless the block is redesigned to avoid the extra
  CTA sync before `dv/do/q/w` consumption.

### Accepted: BV64 WGMMA group merge for KDH and update

- Change: reduced unnecessary WGMMA group waits in the public DSL kernel.
  - KDH previously issued `K0 @ dh0`, committed/waited, then issued
    `K1 @ dh1`. The two BK64 halves now issue into the same `acc_dv2` group and
    wait once.
  - QDO and WDV previously used separate WGMMA groups even though they write
    independent accumulators. They now issue in one group and wait once before
    the recurrent state update.
- Correctness: bitwise equal to the original DSL for H32/H64 with h0+dht at
  `T=512`; `pytest -q tests/test_chunk_delta_h_bwd_dsl.py -s --tb=short`
  passes (`34 passed`).
- Result: stable small speedup on BV64 without changing math order or stores.
  This is not enough for the 1.5x Triton target, but it is a real baseline
  improvement and aligns with the structure seen in `feat_bwd_delta`, where the
  hot loop avoids extra WGMMA waits.

Repeated baseline-vs-new timings, h0+dht, `B=1,K=V=128`:

| Case | Baseline ms | New ms | Speedup |
| --- | ---: | ---: | ---: |
| H=32,T=4096 rep0/1/2 | 0.1773 / 0.1751 / 0.1754 | 0.1739 / 0.1728 / 0.1729 | 1.019 / 1.013 / 1.014 |
| H=32,T=16384 rep0/1/2 | 0.6874 / 0.6902 / 0.6891 | 0.6799 / 0.6798 / 0.6800 | 1.011 / 1.015 / 1.013 |
| H=32,T=32768 rep0/1/2 | 1.3679 / 1.3691 / 1.3689 | 1.3475 / 1.3513 / 1.3512 | 1.015 / 1.013 / 1.013 |
| H=64,T=4096 rep0/1/2 | 0.3409 / 0.3403 / 0.3415 | 0.3373 / 0.3376 / 0.3380 | 1.011 / 1.008 / 1.010 |
| H=64,T=16384 rep0/1/2 | 1.3359 / 1.3397 / 1.3402 | 1.3271 / 1.3278 / 1.3298 | 1.007 / 1.009 / 1.008 |
| H=64,T=32768 rep0/1/2 | 2.6706 / 2.6757 / 2.6684 | 2.6597 / 2.6577 / 2.6650 | 1.004 / 1.007 / 1.001 |

Benchmark script result after the change (`benchmarks/bench_chunk_delta_h_bwd_dsl.py`,
warmup=5, iters=50):

| Case | FLA ms | C++ ms | DSL ms | C++/DSL |
| --- | ---: | ---: | ---: | ---: |
| H=32,T=4096 | 0.2055 | 0.1779 | 0.1756 | 1.013 |
| H=32,T=16384 | 0.7742 | 0.6761 | 0.6755 | 1.001 |
| H=32,T=32768 | 1.5317 | 1.3617 | 1.3473 | 1.011 |
| H=64,T=4096 | 0.3775 | 0.3656 | 0.3332 | 1.097 |
| H=64,T=16384 | 1.4952 | 1.3279 | 1.3284 | 1.000 |
| H=64,T=32768 | 3.0055 | 2.6260 | 2.6620 | 0.986 |

- Diagnosis:
  - The previous extra WGMMA waits were conservative but unnecessary. Removing
    them saves control overhead without exposing shared-memory or accumulator
    hazards.
  - The gain is larger for H32 because the kernel has less aggregate parallelism
    and wait overhead is more visible.
  - H64 long T remains limited by store/update throughput; this change does not
    address the remaining C++ gap at `T=32768`.
- Decision: keep the WGMMA group merge in the public DSL. Next BV64 work should
  target the remaining H64 long-T gap or bring over the useful `feat_bwd_delta`
  structure only where it clearly helps H32 without regressing H64.

## Current Target

Optimize the current CuTeDSL kernel directly. For `B=1,H=32,K=V=128`, profiling
counters are unavailable in this environment due `ERR_NVGPUCTRPERM`, so use stable
timing plus SASS/resource inspection. Current evidence points to BV64 per-CTA
store/barrier overlap as the best local optimization area; BV32 is not profitable
for H=32 in the current DSL implementation, and TMA-only without warp
specialization is slower than cp.async.

The next implementation target is documented in
`docs/chunk_delta_h_bwd_dsl_block_design.md`. The route is now split for
attribution: TMA-only has validated descriptors/layouts but showed the bottleneck
is unhidden mbarrier/S2G wait overhead. Cluster multicast has now been tested
and is not a standalone fix. The next useful TMA-first step has to simplify or
hide the store/wait path rather than adding more multicast/input stages.
