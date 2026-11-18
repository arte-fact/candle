# Roadmap: Expert Parallelism (EP) on 4× MI50 PCIe

**Date:** 2026-04-17
**Target topology:** 4× AMD MI50 (gfx906), PCIe Gen3 x16 (~16 GB/s usable per-link per-direction), no NVLink / xGMI.
**Primary target models:** Qwen3-Coder-30B-A3B (O4 tuned), Gemma-4-26B-A4B (E/A-branch), DeepSeek-V3 / Kimi-K2.5-class (currently OOM on 1× MI50).
**Status:** design only — no code landed. Supersedes the TP/PP notes in `ROADMAP-MULTI-MODEL-2026-04-13.md` for MoE paths.

---

## 1. Why EP, not TP or PP

Survey of PCIe-only multi-GPU scaling (2026-04-17 research round) converges on one conclusion for this topology and this workload mix:

| Strategy | Per-token link traffic (order) | Scales past 2 GPUs on PCIe? | Fits our model list? |
|---|---|---|---|
| Tensor Parallel (TP) | dense all-reduce per layer | **No** — vLLM disables fast all-reduce >2 PCIe GPUs; 0.75× scaling at best | Works for dense only |
| Pipeline Parallel (PP) | point-to-point activation at stage boundary | **Yes**, but needs async batching to overlap | Mostly dense; wasted on MoE |
| **Expert Parallel (EP)** | **sparse all-to-all (topk experts only)** | **Yes** — payload ~10× smaller than TP all-reduce | **Native fit for every MoE target** |

Our hot list is overwhelmingly MoE: Qwen3-Coder-Next, Gemma-4-*-A*B, GLM-4.5, Kimi-K2.5, DeepSeek-V3. EP is the one strategy that (a) reduces per-token link traffic vs. TP, (b) unlocks models that won't fit on 1× 16 GB MI50, and (c) maps cleanly onto the existing `indexed_moe_forward[_preq8]` kernels — the per-expert gather is already in-kernel.

Dense models will get PP later (separate phase). TP=4 is removed from the roadmap for PCIe; TP=2 stays available as a fallback for a small dense class.

## 2. What exists today

- `candle-core/src/hip_backend/device.rs:29` — `HipDevice { ordinal: usize, ... }`. Single-device abstraction.
- `candle-core/src/hip_backend/decode_cache.rs:1225` — K9 multi-device replay plumbing: every `RecordedOp` carries a `device_ordinal` and `hipSetDevice` is issued at device boundaries during replay. Proves we can cross devices inside a captured plan.
- `candle-transformers/src/models/quantized_blocks/ffn.rs:286` — `MoeExperts`. Router replicated, experts held as 3-D `QTensor`s, hot path dispatches to `indexed_moe_forward_preq8` (gate/up) + `indexed_moe_forward` (down).
- `candle-core/src/quantized/hip.rs:1520` — `launch_indexed_moe_forward_with_q8`. Kernel already takes `topk_ids` and computes per-expert output rows. Exactly the primitive EP wants — we just have to feed it the *local* expert subset.
- **Collective primitives: none.** No `hipMemcpyPeer`, no RCCL wrapper, no multi-stream coordination. This is the gap to fill.

## 3. Architecture

### 3.1 Cluster object

**New file:** `candle-core/src/hip_backend/cluster.rs`

```rust
pub struct HipCluster {
    pub devices: Vec<HipDevice>,          // index = rank
    pub rank: usize,                       // this process's rank
    pub world_size: usize,
    peer_streams: Vec<HipStream>,          // one per (rank, other_rank) pair, for overlap
}

impl HipCluster {
    pub fn new(ordinals: &[usize], rank: usize) -> Result<Self>;

    /// Issues hipDeviceEnablePeerAccess for every (i, j) pair. Idempotent.
    pub fn enable_peer_access(&self) -> Result<()>;

    /// Single peer copy on `stream`. Thin wrapper over hipMemcpyPeerAsync.
    pub fn peer_copy_async(
        &self,
        src_rank: usize, src: *const u8,
        dst_rank: usize, dst: *mut u8,
        bytes: usize, stream: &HipStream,
    ) -> Result<()>;

    /// Variable-length all-to-all: each rank sends `send[r]` bytes to rank r.
    /// Returns owned receive buffers on the local device.
    /// v1 impl: naive loop of peer copies + per-dst stream. v2: RCCL.
    pub fn all_to_all_varlen(
        &self,
        send: &[HipSlice<u8>],          // world_size entries, indexed by dst rank
        recv_bytes_per_rank: &[usize],  // known from pre-exchanged header
    ) -> Result<Vec<HipSlice<u8>>>;
}
```

Rationale for peer-copy-first (not RCCL immediately):
- RCCL on gfx906 + ROCm 7.1.1 is not our happy path (not tested, versioning pain).
- Peer copies give us the minimum viable correct path in ~2 days of work.
- The trait shape lets us swap RCCL in later without touching `MoeExperts`.
- For Qwen3-Coder-Next at `topk=10, hidden=2048` the per-token dispatch payload is ~23 KB. At 16 GB/s that's 1.5 µs of raw transfer — launch overhead dominates either way.

### 3.2 Expert placement

**Extend `MoeExperts`** at `candle-transformers/src/models/quantized_blocks/ffn.rs:286`:

```rust
pub struct MoeExperts {
    // existing fields unchanged
    ...
    // EP additions, all Option for single-device back-compat
    cluster: Option<Arc<HipCluster>>,
    local_experts: Option<Range<usize>>,  // e.g. 0..32 for rank 0 of 128/4
    expert_to_rank: Option<Vec<u8>>,       // num_experts entries; only populated when cluster.is_some()
}
```

Placement policy v1: contiguous block partition (`ceil(num_experts / world_size)` experts per rank). Each rank uploads *only* its local slice of `{gate,up,down}_exps` / `gate_up_exps`. Router stays replicated. Shared expert stays replicated.

`MoeExperts::load` grows a new overload:

```rust
pub fn load_ep(gg: &Gguf, prefix: &str, cfg: &GgufConfig,
               cluster: &Arc<HipCluster>) -> Result<Self>;
```

Uses `QTensor::narrow_on_first_axis(local_experts)` (needs to be added — it's an expert-axis slice before GPU upload) to avoid holding all expert weights on rank 0 then scattering.

### 3.3 Hot path: `forward_shared_q8_decode` with EP

Today's decode path (for reference): `ffn.rs:423–586`. Router → topk → `indexed_moe_forward_preq8(gate)` → `indexed_moe_forward_preq8(up)` → `silu_mul` → `indexed_moe_forward(down)` → weighted sum → shared-expert add.

**New EP path (rank r out of W):**

```
[all ranks] input x on local device, already F32 contiguous

[replicated]  router_logits = router.forward(x)
              (topk_weights, topk_ids) = topk_fused(logits, k)
              // every rank has identical topk_ids — router is replicated

[replicated]  q8_x = quantize_q8_1(x)   // needed locally anyway

─── DISPATCH ───
For each token t, for each slot s ∈ 0..k:
    dst_rank = expert_to_rank[topk_ids[t,s]]
    enqueue (t, s, topk_ids[t,s]_local, q8_x[t]) into send_bucket[dst_rank]

pack send_bucket[r'] = [(header: t,s,local_exp) | q8_payload] × n_r'
all_to_all_varlen(send_bucket) → recv_tokens[]

─── LOCAL EXPERT COMPUTE ───
(on rank r, over recv_tokens with LOCAL expert ids only)
  gate_out = indexed_moe_forward_preq8(gate_exps_local, recv_q8, recv_local_ids)
  up_out   = indexed_moe_forward_preq8(up_exps_local,   recv_q8, recv_local_ids)
  act      = silu_mul(gate_out, up_out)
  down_out = indexed_moe_forward_preq8(down_exps_local, q8(act), recv_local_ids)

─── COMBINE ───
pack (t, s, down_out[i]) back into return_bucket[src_rank]
all_to_all_varlen(return_bucket) → origin rank

[replicated]  scatter into moe_out[t, s, :]   // hole-fill via id → slot map
              weighted = broadcast_mul(moe_out, topk_weights)
              result   = weighted.sum(1)
              result  += shared_mlp(x) * sigmoid(shared_gate · x)
```

**Critical path budget per MoE layer per token (Qwen3-Coder-Next class):**

| stage | bytes | bw cost | launch/sync cost | total est. |
|---|---:|---:|---:|---:|
| bucket + pack dispatch | — | — | 1 launch | ~2 µs |
| dispatch all-to-all | 23 KB/rank pair × 3 peers | 1.5 µs | stream sync | ~5 µs |
| local expert compute (¼ of k experts) | — | — | 3 MMVQs | ~25 µs |
| combine all-to-all | 8 KB/rank pair × 3 peers | 0.5 µs | stream sync | ~3 µs |
| scatter + reduce | — | — | 2 launches | ~2 µs |
| **total per MoE layer** | | | | **~37 µs** |

Current 1-GPU baseline Qwen3-Coder-Next MoE layer (decode, post-O4): ~130 µs/layer (experts only; router etc. on top). So per-layer ideal saving ≈ 90 µs × 48 layers = **4.3 ms / token**. At 33.7 ms/token today (29.7 t/s) that's 34 t/s ideal → **~+15 % decode**, before accounting for load imbalance and real-world sync costs.

### 3.4 Prefill and G2/G3

- **Prefill v1: OFF.** Fall through to non-EP single-device path when `b*m > decode_threshold`. Prefill all-to-all payloads scale linearly with prompt length (at pp512 with topk=10 → 12 MB/peer — hits 1 ms+ per layer on Gen3). Proper prefill EP needs microbatch pipelining; separate phase.
- **G2/G3 v1: OFF in EP mode.** Cross-device replay with peer-copy ops in captured plans needs `RecordedOp::PeerCopy { src_rank, dst_rank, bytes }` and matching replay logic. K9 done the hard part (device ordinal per op); adding peer copies is a contained extension but not v1-critical.

## 4. Milestones

### Phase X1 — peer-copy primitive (est. 2 days)

- `HipCluster::{new, enable_peer_access, peer_copy_async}` landed.
- Standalone test `candle-core/examples/hip_peer_copy_test.rs`: write pattern on device 0, peer-copy to 1/2/3, verify.
- **Exit criterion:** 4-way peer copy of 1 MiB at steady ~12 GB/s aggregate (close to Gen3 x16 ceiling).

### Phase X2 — `all_to_all_varlen` + bench (est. 2 days)

- Naive impl: W separate peer copies on W streams, 1 sync barrier.
- Header-first protocol: exchange byte counts via pinned host memory or device-resident scratch.
- **Exit criterion:** all-to-all of random-size buckets (~25 KB/rank pair) at <= 15 µs end-to-end.

### Phase X3 — Gemma-4-26B-A4B EP proof on 2 GPUs (est. 3 days)

Why this model first: current 4-GPU Gemma-4-26B-A4B is 0.50× turbo (memory note: `project_moe_comparative_bench_2026_04_15.md`). It's the worst-performing MoE in our tree, so it has the most headroom and the biggest risk of broken baselines. W=2 first to keep placement trivial.

- `QTensor::narrow_on_first_axis` added (slices expert axis before upload).
- `MoeExperts::load_ep` implemented.
- `MoeExperts::forward_shared_q8_decode_ep` implemented alongside existing (non-EP) path; gated by `cluster.is_some()`.
- **Exit criterion:** correctness (logit equality vs. 1-GPU to < 5e-3 rel) + decode t/s ≥ current 4-GPU (0.50× turbo).

### Phase X4 — W=4 full ladder (est. 3 days)

- All 4 MI50s occupied.
- **Exit criterion:** Gemma-4-26B-A4B decode ≥ 0.75× turbo. Qwen3-Coder-Next decode ≥ 1-GPU baseline (no regression from added dispatch cost) ideally +15 %.

### Phase X5 — OOM-class models (est. 4 days)

Targets: DeepSeek-V3-Lite-class (~40 GB MoE), Kimi-K2.5 (~55 GB).

- Expert weight streaming on load (cannot fit full model on rank 0 then scatter).
- Per-rank GGUF tensor range reads (extend `Gguf::tensor_range` to accept an expert-axis span).
- **Exit criterion:** model inference runs end-to-end; ≥ 15 t/s decode target.

### Phase X6 — RCCL swap (est. 2 days, optional)

Only if peer-copy all-to-all shows up in profiles as a hot path. Swap `HipCluster::all_to_all_varlen` implementation; API unchanged.

### Phase X7 — prefill EP with microbatching (separate roadmap)

Deferred — likely its own 2-week roadmap once X3–X5 land.

## 5. Theoretical speedup projection

### 5.1 Decode (primary goal)

| model | current best | EP=4 ideal | EP=4 realistic (after overhead, imbalance, sync) | gain |
|---|---:|---:|---:|---:|
| Qwen3-Coder-30B-A3B Q4_0 | 29.7 t/s (1-GPU, post-O4) | ~40 t/s | **34–37 t/s** | **+15–25 %** |
| Gemma-4-26B-A4B Q8_0 | ~33 t/s (4-GPU, 0.50× turbo) | ~70 t/s | **50–60 t/s (0.75–0.90× turbo)** | **+50–80 %** |
| DeepSeek-V3-Lite / Kimi-K2.5 | **OOM** | 20–25 t/s | **15–25 t/s** | **∞ (runs at all)** |

### 5.2 Prefill

Untouched in X3–X5 (falls back to 1-GPU path). When X7 lands, expect +30–50 % on MoE prefill at pp512+ sizes; <pp64 likely regresses due to dispatch overhead amortising poorly.

### 5.3 Honest caveats

- The headline "0.75× PCIe scaling" number in the survey assumed **Gen5 x16 at ~64 GB/s**. MI50 is **Gen3 x16 at ~16 GB/s**, which is 4× worse. Our Qwen3-Coder-Next decode projection already bakes this in (why it's only +15 %, not +34 %). Gemma-4-26B-A4B has more room because its current 4-GPU path is broken, not bandwidth-limited.
- **Load imbalance:** naive contiguous-block placement + uniform-ish routing will leave 1–2 ranks with ~20 % more tokens on pathological batches. DeepSeek's group-limited gating fixes this; we won't have it in v1. Expect ~20 % tail latency inflation until X8 (not scheduled).
- **Shared-expert cost is replicated, not parallelised.** For models where the shared expert is a large fraction of per-token FLOPs (Qwen3-Coder-Next: ~20 %), EP doesn't help it. Cap on decode speedup reflects this.
- **Launch latency dominates small transfers.** At Gen3, dispatching 23 KB takes ~1.5 µs of bandwidth time but ~4–8 µs of launch+sync. Below topk=4 or hidden<1024 EP might not be worth it on this topology.
- **Router replication is wasteful at scale.** For very large num_experts (>256) the router gemv starts to matter; a follow-up could shard it too. Not in scope.
- **Contention with existing multi-device paths.** ROADMAP-DECODE-GAP-2026-04-16 assumes 4-GPU via layer-split (tensor_split `1/1/1/1`). EP replaces that for MoE. Dense layers (attention, shared expert) still need a placement decision — v1 keeps them replicated on every rank.

## 6. Open questions

1. **Replicated attention, or EP attention?** v1 replicates attention across ranks (cheap, wastes VRAM). Alternative: co-locate attention with its originating token's rank (requires KV cache partition by token-rank). Decide after X4 profile.
2. **Pinned host staging for dispatch?** On Gen3, device→device peer copy bandwidth depends on the PCIe root complex. Some topologies are faster through pinned host bounce. Measure in X1.
3. **Single process, multi-device vs. multi-process?** v1 is single-process (`Vec<HipDevice>` in one `HipCluster`). Multi-process adds IPC but could relieve the 31-thread CPU leak called out in Phase O5. Revisit after X4.
4. **Interaction with G2/G3.** Turning on EP forces G2/G3 off in v1. The Gemma-4-E4B G2 correctness win (Phase K13) would need re-proving on the EP path; that model is dense + MoE hybrid and is its own can of worms.
5. **Router requantise interaction.** Phase O4 requantises `ffn_gate_inp.weight` on load. That happens per-rank under EP — harmless but redundant. Consider load-time broadcast from rank 0 once X5 streaming weights arrives.

## 7. Out of scope

- TP=2 / TP=4 for dense models (separate path; TP=4 on PCIe is explicitly rejected by this doc).
- PP for dense models (future ROADMAP-PIPELINE-PARALLEL).
- NVLink / xGMI ports (no hardware).
- Cross-host / RDMA (one-box only).
- Dynamic expert migration, LRU caching, expert offload to CPU (Phase Y+).
- Quantization of activations during dispatch below Q8_1 (possible future: Q4_1 dispatch payload, ~50 % bandwidth saving, adds dequant cost on receiver — benchmark when X4 profiles say bandwidth matters).

## 8. References

- expert-kit/expert-kit — Rust E/A-separation scaffolding, closest prior art (LGPL-3).
- DeepSeek-V3 paper §3.2 (group-limited gating) — load balancing pattern for X8.
- DeepEP — reference CUDA kernels for EP dispatch/combine; patterns to port.
- NVIDIA Llama-3.1-405B PP vs TP analysis — justifies PP > TP even *with* NVLink.
- vLLM `_init_custom_all_reduce` — documents the PCIe>2-GPU TP disable.
- Internal: `ROADMAP-DECODE-GAP-2026-04-16.md`, `project_moe_comparative_bench_2026_04_15.md` (memory), `project_q8_kv_port_phases_1_5.md` (memory), Phase K K9 (memory).
