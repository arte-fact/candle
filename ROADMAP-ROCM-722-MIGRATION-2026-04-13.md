# ROCm 7.2.1 + Tensile gfx906 Migration Roadmap — 2026-04-13

## 2026-04-14 update — measured, no benefit on gfx906

**TL;DR**: ROCm 7.2.1 HIP runtime installed and benched against 7.1.1 on
Gemma4-E4B Q4_0. **No improvement**; the G3 patch loop is actually
slower (547-661 µs vs 430 µs / 218 ops). The AQL packet-batching
improvement claimed in arxiv 2511.11581 (which powers the Triton
"launch-overhead" 1.99× win) is **MI200/MI300-specific**, not gfx906.
This migration is **closed as negative result**.

| bench | ROCm 7.1.1 | ROCm 7.2.1 HIP + 7.1.1 rocBLAS | Δ |
|---|---|---|---|
| Prefill (178 tok) | 644 t/s | 643 t/s | 0.0 % |
| Default decode (100 tok gen) | 65.35 t/s | 65.15 t/s | -0.3 % |
| G3 decode (100 tok gen) | 60.40 t/s | 59.71 t/s | **-1.1 %** |
| G3 patch loop (218 ops) | 430 µs | ~600 µs | **-40 %** (slower) |

**Install config benched**: `LD_PRELOAD=/opt/rocm-7.2.1/lib/libamdhip64.so.7 LD_LIBRARY_PATH=/opt/rocm-7.1.1/core-7.13/lib`
(HIP 7.2.1 + rocBLAS 7.1.1 because 7.2.1 ships **zero** gfx906 Tensile
kernels — native rocBLAS calls crash with "Cannot read
`TensileLibrary.dat` for GPU arch : gfx906"). Pure-7.2.1 would need
mixa3607's `tensile-files-7.2.1.tgz` supplement.

**Conclusion**: the gfx906 graph-node overhead ceiling is **not an API
version problem**, it's a silicon-generation limitation (pre-CDNA2, no
AQL packet batching support). There is no userspace path to close the
G3 launch-overhead gap on this hardware.

**Action**: keep ROCm 7.1.1 as the build target. G3 stays opt-in
(`CANDLE_G3_GRAPH=1`) and remains a regression on gfx906 — use
default decode unless benchmarking.

---

## Starting point

| Stack | libamdhip64 | librocblas | Tensile gfx906 coverage | Used by |
|---|---|---|---|---|
| **Our candle today** | `.so.7` (ROCm 7.1.1) | `.so.5` | **Partial** — `rocblas_sgemv` falls to comgr JIT → SIGSEGV on gfx906 | Phase O blocked |
| vllm-gfx906-mobydick (`/artefact/vllm-gfx906-mobydick`) | `.so.6` (ROCm 6.3.4) | `.so.4` | Complete — last version before AMD deprecation | Used by `/artefact/mobydick-venv` vLLM build |
| nlzy/vllm-gfx906 (archived Feb 2026) | `.so.6` | `.so.4` | Complete | Reference vLLM fork for gfx906 |
| **Target — mixa3607 ML-gfx906** | `.so.7` (ROCm **7.2.1**) | `.so.5+` | **Complete via re-added `rocm-tensile` package** | mixa3607 Docker `rocm-gfx906:7.2.1-complete` |

**Key fact**: AMD dropped complete gfx906 Tensile kernels from rocBLAS starting
~5.7. ROCm 6.3.x was the last version to ship them natively. ROCm 7.x gives us
newer rocBLAS API but *incomplete* gfx906 kernel coverage — which is why
`rocblas_sgemv_strided_batched` silently falls back to a comgr JIT path that
crashes on this specific driver/kernel combo (Phase O postmortem: `2123e649`).

[mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906) fixes this by
maintaining a `rocm-tensile` package that re-packages the missing gfx906
Tensile kernel blobs and drops them into ROCm 7.2.1's `lib/rocblas/library/`
tree. They've done the work the wider community needs; migrating our candle
dev environment to their stack is low-effort and high-leverage.

## Why migrate

1. ~~**Unblocks Phase O revival.**~~ **WITHDRAWN 2026-04-13 after tensile
   bundle inspection.** The rocBLAS `sgemv_strided_batched` we tried to use
   for GEMV-based decode attention (commits `d1307b92`, `2123e649`,
   `56309b69`) does **NOT** crash due to a missing Tensile kernel file.
   Direct comparison of `/opt/rocm-7.1.1/lib/rocblas/library/` (255 gfx906
   kernels) against mixa3607's `tensile-files-7.1.1.tgz` (211 files) shows
   our installed set is a strict **superset** of the community bundle —
   zero files missing. Additionally, no file in either set has `sgemv` or
   `gemv` in its name; rocBLAS's sgemv path is generated at runtime via
   `comgr` JIT (observed in strace: `/tmp/comgr-XXX/output/hipfatbin-*.o`),
   so the problem is in the JIT codegen path itself, not a missing
   pre-compiled kernel. Migrating to 7.2.1 may help indirectly via a
   newer `hipcc`/comgr, but it's not the targeted fix I assumed.
   **Phase O revival remains blocked until we dig into comgr JIT
   behavior on gfx906 + ROCm 7.1.1 specifically.**

   **Further confirmed 2026-04-13 (Phase R planning)**: pulled mixa3607's
   `tensile-files-7.2.1.tgz` (`https://static.arkprojects.space/
   public-data/wiki/AMD-GFX906/rocm-tensile/tensile-files-7.2.1.tgz`,
   39 MB, 156 gfx906 kernel files). Diff vs `/opt/rocm-7.1.1/lib/
   rocblas/library/` (255 gfx906 files):
     - 7.2.1 bundle → 7.1.1 install: **0 new files**
     - 7.1.1 install → 7.2.1 bundle: 99 extra
   So the pre-compiled kernel coverage on disk does not improve at all
   by migrating. The only potential migration benefit left is speculative:
   a newer `hipcc`/`comgr` might fix the DPP warp-reduce miscompile and
   the K-quant warp-coop crash we observed on ROCm 7.1.1 (Phase J / B3).
   That isn't demonstrable without actually migrating.

2. **May naturally help Phase Q2 (kernel tuning).** The `gqa_decode_mv_d256`
   kernel we landed in Phase P Stage 1 (`ee83df34`) runs at **73 μs/call**
   vs turbo's `mul_mat_vec_f` at **~9 μs/call**. Our kernel isn't dramatically
   different in algorithm — the per-call gap suggests instruction scheduling,
   VMEM latency hiding, or register allocation differences. ROCm 7.2.1 ships
   a newer `hipcc` (LLVM-based) with more mature gfx906 codegen, particularly
   in the v_dot and LDS schedule passes. Not a guaranteed win but worth
   measuring before investing in manual kernel rewrites.

## 2026-04-13 priority reassessment

With justification #1 withdrawn twice (sgemv is runtime codegen, and the
7.2.1 bundle brings no new pre-compiled kernels) and the remaining
benefits purely speculative, this migration is **demoted from STEP 1 to
a contingent follow-up**. The concrete Phase R (LDS `+1` padding) win is
a better next move:

| action | cost | concrete benefit |
|---|---|---|
| ROCm 7.2.1 migration (this doc) | ~1 day op + bench | speculative (DPP, hipcc codegen) |
| **Phase R — LDS `+1` padding** | 2-4 h per kernel | 2× LDS bandwidth (skyne98 wiki) |
| Phase Q2 — mat-vec kernel tuning | 1-2 days | 73 → ~30 μs per d=256 call |

Proceed with Phase R + Q2 on the existing ROCm 7.1.1 stack first. Only
return to this migration if one of:
- Phase J DPP reductions become the bottleneck and we need to test if
  newer hipcc fixes the MI50 miscompile.
- We ship a Docker image / product bundle and need to align with the
  community ROCm 7.2.1 default.
- Third-party vLLM / PyTorch / ComfyUI integration forces the version.

Until then, keep `/opt/rocm-7.1.1` as the build target. The `tensile-
files-7.2.1.tgz` tarball was inspected but not installed.

3. **Aligns with the community ecosystem.** mixa3607's ROCm 7.2.1 image is
   the current community default for gfx906 ML work. vLLM 0.19.1,
   PyTorch 2.11, ComfyUI — all ship against it. If we ever want to mix our
   candle binary with those artifacts (shared Python env, shared CUDA
   context for co-located inference), binary ABI needs to match.

4. **No regressions expected on the working paths.** The gfx906 Tensile
   kernels mixa3607 re-adds are the *same* blobs that ship in ROCm 6.3.4.
   The newer rocBLAS API wraps them correctly. Our flash-attn paths compile
   against HIP headers, not rocBLAS internals, so they're unaffected by
   rocBLAS version.

## What this roadmap does NOT do

- Does **not** rebuild candle against ROCm 7.2.1 in this session. That's a
  follow-up after driver install + sanity check. It's a full `cargo clean
  && cargo build --release --features hip` (~2 min) across hipdarc,
  candle-core, candle-hip-kernels. Destructive of all existing artifacts.
- Does **not** remove `/opt/rocm-7.1.1` or `/opt/rocm-6.3.4` from the sandbox.
  Both stay intact for fallback. Version selection is via `ROCM_PATH` env
  var at build time + `LD_LIBRARY_PATH` at runtime.
- Does **not** block Phase P Stage 2 (wire Phase P default-on) or Phase Q1
  (G2 + Phase P compatibility). Those are on the existing ROCm 7.1.1 stack
  and can proceed independently.
- Does **not** touch the `vllm-gfx906-mobydick` install. That lives on 6.3.4
  and is unaffected by adding 7.2.1 alongside.

## Execution plan

### Step 1 — Install mixa3607 ROCm 7.2.1 + Tensile gfx906 package

Distribution options (easiest first):

**Option A — Docker Hub registry HTTP API** (no docker daemon needed):
```
# Manifest for mixa3607/rocm-gfx906:7.2.1-complete
curl -s "https://registry-1.docker.io/v2/mixa3607/rocm-gfx906/manifests/7.2.1-complete" \
     -H "Accept: application/vnd.docker.distribution.manifest.v2+json"
# Iterate over layer blobs, fetch each as a tarball, extract
# layer_n.tar.gz → / in the target rootfs
```
Layers extract directly to `/opt/rocm-7.2.1/`. Known to contain:
- `/opt/rocm-7.2.1/lib/librocblas.so.5*`, `libhipblaslt.so.1*`, `libamdhip64.so.7*`
- `/opt/rocm-7.2.1/lib/rocblas/library/Kernels.so-000-gfx906*.hsaco` (the
  re-added kernels)
- `/opt/rocm-7.2.1/bin/hipcc`, `rocminfo`, `rocm-smi`
- Headers under `/opt/rocm-7.2.1/include/`

**Option B — GitHub release tarballs.** mixa3607 pins releases at
`https://github.com/mixa3607/ML-gfx906/releases` (tag `20260412003929` as of
this writing). These ship scripts + small artifacts; the bulk of the ROCm
install is sourced via their Dockerfile instructions. Not standalone
installable without the Docker pull.

**Option C — Reconstruct from rocm-tensile + AMD official ROCm 7.2.1 debs.**
AMD ships `amdgpu-install --rocmrelease=7.2.1` (public); then
`mixa3607/rocm-tensile` supplies just the gfx906 `.hsaco` files to drop into
`/opt/rocm-7.2.1/lib/rocblas/library/`. Two-step but cleanest if Option A
fails (e.g. Docker Hub rate-limits unauthenticated pulls).

**Chosen path**: Try Option A first (single-source, guaranteed identical to
mixa3607's tested stack). Fall back to Option C on any failure.

### Step 2 — Validation checklist

After install:
- [ ] `/opt/rocm-7.2.1/lib/librocblas.so.5` present and non-empty
- [ ] `ls /opt/rocm-7.2.1/lib/rocblas/library | grep gfx906 | wc -l` ≥ the
      count in `/opt/rocm-6.3.4/lib/rocblas/library/`
- [ ] `nm -D /opt/rocm-7.2.1/lib/librocblas.so | grep sgemv_strided_batched`
      — symbol present
- [ ] `readelf -d /opt/rocm-7.2.1/lib/librocblas.so` — no missing DT_NEEDED
      entries that aren't either in `/opt/rocm-7.2.1/lib/` or standard
      system locations
- [ ] `LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib /opt/rocm-7.2.1/bin/rocminfo |
      grep gfx906` — all 4 MI50s enumerate
- [ ] `LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib /opt/rocm-7.2.1/bin/rocm-smi` —
      runs without error

### Step 3 — Rebuild candle against 7.2.1 (follow-up session)

Not in this roadmap's scope, but for continuity the steps are:
```
cd /artefact/candle
export ROCM_PATH=/opt/rocm-7.2.1
export HIP_PATH=/opt/rocm-7.2.1
export PATH=/opt/rocm-7.2.1/bin:$PATH
cargo clean -p hipdarc -p candle-core -p candle-hip-kernels
touch candle-hip-kernels/build.rs
cargo build --release --example quantized-gemma4 --features hip
# then run with LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib
```

Regression bench against current baseline (on `/opt/rocm-7.1.1`):
- TinyLlama Q4_0 decode: ≥ 260 t/s (current)
- Qwen3.5-9B Q4_1 decode: ≥ 40 t/s
- Gemma4-E4B Q4_0 decode, default: ≥ 51 t/s
- Gemma4-E4B Q4_0 decode, `CANDLE_KV_TMAJOR=1`: ≥ 53 t/s
- Gemma4-E4B Q4_0 decode, `CANDLE_G2_REPLAY=1`: ≥ 46 t/s
- Phase O sgemv smoke test: no crash (regression check vs the
  documented ROCm 7.1.1 crash)

### Step 4 — Optional Phase O revival

If Step 3 confirms no regressions, un-cfg-gate the Phase O dispatch in
`candle-transformers/src/models/quantized_blocks/attention.rs` (currently
`#[cfg(any())]`) and benchmark rocBLAS-gemv decode attention vs our
Phase P `gqa_decode_mv` kernel. Per the ROCm 6.3.4 rebuild experiment we
ran (11.6% faster gemv on same rocm), gemv could beat our hand-written
kernel — but we need same-stack comparison to confirm.

## Interactions with existing phases

| Phase | Status | ROCm 7.2.1 migration impact |
|---|---|---|
| Phase H (gemma4 attention port) | Done | Unaffected |
| Phase I (MMQ rewrite) | Ongoing | Possibly faster under newer hipcc; measure |
| Phase J (Qwen correctness) | Done | Unaffected |
| Phase K (gemma4 G2/G3 correctness) | Done | Captured plans may need re-recording once after migration |
| Phase L (MMVQ tune) | Pending | Newer hipcc could change the tune targets; defer until after migration |
| Phase M (MoE FFN) | Deferred | Unaffected |
| Phase N (prefill) | Pending | rocBLAS gemm improvements may help here directly |
| Phase O (rocBLAS gemv decode) | **Blocked on current ROCm** | **UNBLOCKED by migration** — high-priority follow-up |
| Phase P Stage 1 (T-major KvCache) | Landed (`ee83df34`) | Unaffected; opt-in via `CANDLE_KV_TMAJOR=1` |
| Phase P Stage 2 (default-on) | Pending | Unaffected |
| Phase Q1 (G2 + P compat) | Pending | Unaffected |
| Phase Q2 (kernel tuning) | Pending | **May be partly addressed by newer hipcc codegen** |
| Phase Q3 (skip `.contiguous()`) | Pending | Unaffected |
| Phase Q4 (d=512 prefill kernel) | Pending | Unaffected |

## Risks

1. **Docker Hub registry throttling.** Unauthenticated pulls from
   `registry-1.docker.io` are rate-limited (100 pulls / 6h per IP). If we
   hit it, fall back to Option C (AMD official 7.2.1 debs + mixa3607's
   tensile-only package from their release artifacts).
2. **ABI drift.** If mixa3607's 7.2.1 build uses different soname
   structure than AMD's 7.2.1, our rebuilt candle might link against
   symbols mixa3607's lib doesn't provide. Mitigation: verify
   `readelf -d` early, before rebuilding candle.
3. **Disk space.** `/opt/rocm-7.2.1` full install is ~15 GiB. Current
   `/opt/rocm-7.1.1` is ~12 GiB, 6.3.4 is ~10 GiB. If disk is tight we
   can prune obsolete kernels for gfx1030/gfx1100/etc — but that's a
   tightening step, not a blocker.
4. **HIP runtime conflicts.** If a `LD_LIBRARY_PATH` leak from another
   tool puts /opt/rocm-7.1.1/lib ahead of /opt/rocm-7.2.1/lib, the
   candle binary will partially link against the wrong version and
   crash in opaque ways. Mitigation: prepend `/opt/rocm-7.2.1/lib`
   explicitly in our bench scripts.
5. **Incompatibility with existing vLLM install.** `vllm-gfx906-mobydick`
   is linked against 6.3.4. If we later want both candle and vllm in the
   same process (e.g. for comparative profiling), one will need to
   rebuild. Not a blocker now; defer to if/when we need that integration.

## Expected outcomes

- Phase O (rocBLAS gemv decode) becomes measurable → may unlock an
  additional +5-10% on top of Phase P Stage 1.
- Phase Q2 (kernel tuning) baseline may improve naturally from newer
  codegen; if so, Q2 scope shrinks.
- Alignment with community stack → easier to consume third-party
  builds (vLLM, ComfyUI) side-by-side with our candle.
- **No regression budget**: every benchmark in the regression list must
  be ≥ current numbers on `/opt/rocm-7.1.1`. If any regresses > 5%,
  roll back to `ROCM_PATH=/opt/rocm-7.1.1` until investigated.

## References

- Existing roadmap (Phase P/Q context): `ROADMAP-MULTI-MODEL-2026-04-13.md`
- Phase O postmortem commits: `d1307b92`, `2123e649`, `56309b69`
- Phase P Stage 1: commit `ee83df34`
- Upstream mixa3607: https://github.com/mixa3607/ML-gfx906
- mixa3607 rocm-tensile: https://github.com/mixa3607/ML-gfx906/tree/master/rocm-tensile
- Community docs referenced: https://arkprojects.space/wiki/AMD_GFX906
- gfx906 wiki KV-cache study (validates Phase P layout choice):
  https://skyne98.github.io/wiki-gfx906/studies/2026-02-21/gfx906-kv-cache-read-write-study.html
