# Candle Model Load-Time Analysis — 2026-04-13

Measured on RTX 3090 (Ampere, PCIe Gen4 x16, CUDA 12.8 driver 570.211.01), 6-core sandbox, 17.17 GB Qwen3.5-27B-Q4_1.gguf.

## Observed wall time

| Phase | Time |
|---|---|
| `gguf_file::Content::read_mmap` (metadata parse) | 0.11 s |
| `ModelWeights::from_gguf` (weights → GPU) | **31.7 s** |
| First-token prefill | 0.3 s |
| Decode t/s (stable) | 29–41 t/s |

Effective bandwidth: **17.17 GB / 31.7 s = 540 MB/s** — roughly 45× below the PCIe Gen4 x16 pinned ceiling (~24 GB/s) and 15× below the pageable-memory practical ceiling (~8 GB/s).

## What the wall time is actually spent on

`strace -c` (summary over the full load + 1 decode token):

| syscall | calls | time (s) | % |
|---|---:|---:|---:|
| `futex` (thread sync) | 29 | **13.31** | **87.5%** |
| `pread64` | 6 | 0.97 | 6.4 |
| `ioctl` (nvidia driver) | 2 286 | 0.76 | 5.0 |
| `sched_yield` (spin) | 18 941 | 0.034 | 0.2 |

`/usr/bin/time -v`:
- User: 3.7 s · System: 15.2 s · Wall: 34 s · CPU: 55 %
- Minor page faults: 5.5 M (→ 21 GB of first-touch)
- **Cold vs warm page cache: identical** (no major faults either way) → not disk-I/O bound
- Maximum RSS: 2.0 GB (host buffers ride a sliding window, good)

## Interpretation

1. **Not I/O-bound**: cold vs warm shows zero delta. The file is served from page cache or prefetched by SSD controller.
2. **Not CPU-bound**: 55 % utilization.
3. **Not GPU-compute-bound**: no kernels run during build.
4. **Bottleneck is thread synchronization**: 13.3 s in 29 futex calls (avg 460 ms each). Combined with 18 941 `sched_yield` calls, this is classic "CPU threads waiting for something" behavior.

Most likely source of the waits (in decreasing probability):
- **Rayon workers contending on the CUDA stream / context**: the per-layer `.into_par_iter()` path has up to 62 threads calling `memcpy_htod` on the same `CudaDevice`. Each `cuMemcpyHtoDAsync` from pageable memory internally stages through a driver-managed pinned bounce buffer that's guarded by a context-wide lock. Workers queue up and sleep on futexes.
- **Per-tensor allocator churn**: `GgufBlob::read_to_vec` `vec![0u8; len]` + drop, 851 times. The code comments note mmap was tried and rejected due to page-fault cost on AMD — pread into a fresh `Vec<u8>` was the chosen alternative. But it still pays a `vec![0; N]` zero-fill per tensor.
- **`load_quantized`'s `alloc_zeros`**: I fixed this to `alloc` + tail-only zero (saves ~400 ms but barely moves the needle — that was the small change, not the dominant one).

Thread-count sweep confirms #1:

| `RAYON_NUM_THREADS` | wall | sys | CPU% |
|---:|---:|---:|---:|
| 1 | 39.6 s | 13.5 s | 42 % |
| 2 | 33.5 s | 14.2 s | 53 % |
| 6 (default) | 34.0 s | 15.2 s | 55 % |

Going 1→2 threads saves 6 s; 2→6 saves nothing. Only ~2 threads' worth of work can actually run in parallel because of the GPU-stream serialization point.

## How other frameworks avoid this

### llama.cpp (CUDA backend)
- **mmap the GGUF** + `ggml_backend_tensor_set` per tensor → `cuMemcpyHtoDAsync`.
- Key trick: the mmap'd region is **registered with `cuMemHostRegister(..., CU_MEMHOSTREGISTER_PORTABLE)`** once at load. After registration, the same virtual pages are treated as pinned for CUDA — no bounce buffer, direct DMA, no page-fault-per-page pattern.
- A single copy path, no rayon/threadpool fighting for a stream.
- Published numbers for 30B Q4 on an RTX 3090: **2–4 s warm**, 4–8 s cold (SSD-dependent).

### vLLM (safetensors → PyTorch)
- `safetensors.torch.load_file` mmaps the file, hands each tensor's byte range as a CPU tensor, `.to(device, non_blocking=True)` does the H→D copy.
- PyTorch's allocator uses CUDA's pinned memory pool — transfers hit ~22 GB/s.
- Workers (one per GPU in tensor-parallel mode) load shards in parallel on **separate CUDA contexts**, so no stream contention.
- Published numbers: **3–6 s warm**, 10–20 s cold for 30B including safetensors parse.

### Candle (current fork)
- `GgufBlob::pread` into fresh `Vec<u8>` → `CudaDevice::memcpy_htod` (pageable).
- Rayon per-layer parallelism that hurts more than it helps because of stream contention.
- **Measured: 32 s** for 17 GB (3× slower than llama.cpp, 6× slower than PCIe ceiling).

## Ranked optimization recommendations

Listed by (expected speedup) / (engineering hours). Numbers assume single-GPU 30B-class load.

### L1. `cuMemHostRegister` the `GgufBlob` pages → true pinned H→D *(1 day, estimated 5–10× speedup)*

Once per `GgufBlob::open`, call `cuMemHostRegister(blob_ptr, blob_len, CU_MEMHOSTREGISTER_PORTABLE)`. This makes the pageable file-backed memory *or* a block-read buffer look pinned to CUDA, enabling DMA without a bounce buffer. This is exactly what llama.cpp does.

- **Requires** either switching back to mmap (blob becomes a long-lived stable pointer) *or* allocating one big scratch buffer at open time and reading into it once.
- Needs a cudarc patch (or `unsafe extern "C"` call) for `cuMemHostRegister` — cudarc 0.19 only exposes `alloc_pinned` via `cuMemHostAlloc`. `cuMemHostRegister` is adjacent CUDA driver API.
- Expected bandwidth: 20 GB/s pinned DMA → 17 GB load in ~0.9 s.

### L2. One pinned staging buffer + streamed per-tensor upload *(4 h, ~3–5× speedup)*

Alternative to L1 when you can't register the GGUF blob:

- Allocate one `PinnedHostSlice<u8>` of max-tensor size (~512 MB).
- Per tensor: `pread` into the pinned buffer, then `memcpy_htod` from pinned → full DMA bandwidth.
- No rayon — serial is fine when you're bandwidth-bound.
- Expected: 17 GB × (1/CPU_BW + 1/PCIe_BW) ≈ 17 × (1/15 + 1/24) ≈ 1.8 s.

### L3. Kill rayon parallelism on the upload path *(30 min, 0–10% speedup)*

The `into_par_iter()` over 62 layers actively hurts when there's only one GPU. Benchmark shows `RAYON_NUM_THREADS=1` is only 6 s slower than default — all the "parallelism" is fighting the stream lock.

- For single-device loads, use plain `.iter()`.
- For multi-device, scope rayon to one worker per physical device (not per layer).
- Zero risk, small win.

### L4. Batch-read contiguous tensor groups *(1 day, 1.5–2× speedup)*

GGUF stores tensors back-to-back on disk. Instead of 851 × `pread` + 851 × `memcpy_htod`, group N tensors that live in the same disk region, do **one** big `pread` + **one** big `memcpy_htod`, then index into the on-GPU buffer to get each tensor's storage.

- Works beautifully if we slightly refactor `QCudaStorage::from_data` to accept a "view into a shared upload buffer" instead of owning its own padded allocation.
- Cuts per-tensor overhead from 851 × (alloc + launch + bounce) down to a constant.

### L5. Skip `vec![0; N]` zero-fill in `GgufBlob::read_to_vec` *(2 h, 5–10% speedup)*

Rust's `vec![0u8; len]` for large N usually comes from an anonymous mmap that's already zeroed — but the subsequent `pread` still dirties each page (minor fault per 4 KB). Use `Vec::with_capacity` + `MaybeUninit` + `read_exact_at` on the uninit slice via `slice::from_raw_parts_mut` → avoids the kernel's COW break on every page.

Worth maybe 1–2 s out of 15 s of system time.

### L6. Profile with Nsight Systems to confirm L1's assumption *(1 h, instrumentation)*

Before investing in L1/L2, run `nsys profile` for one load to see where the 13.3 s of futex time originates. If it's not the CUDA stream lock, the priorities change.

### L7. Async alloc + double-buffered streams *(1 day, 1.5× on top of L1/L2)*

Once L1 or L2 is in place and copies are bandwidth-bound, overlap the next tensor's disk read with the current tensor's H→D DMA using two pinned buffers ping-pong style. Peaks the disk and PCIe at the same time.

## Immediate cheap wins already landed

- `alloc_zeros` → `alloc` in `quantized/cuda.rs::load_quantized` (saves ~400 ms of wasted `cudaMemset`).

## Projected result if L1 + L3 + L5 land

17 GB in 1–2 s — roughly **15–30× faster**. That puts candle at parity with llama.cpp-cuda and faster than vLLM's cold path.

## Target model expectation

| Model | Disk size | Current load | After L1+L3+L5 |
|---|---:|---:|---:|
| gemma-4-E4B-Q8_0 | 8.2 GB | 15 s | ~0.7 s |
| Qwen3.5-27B-Q4_1 | 17.2 GB | 32 s | ~1.5 s |
| Qwen3.5-35B-A3B-MXFP4 (if C2 lands) | 21.6 GB | (n/a today) | ~2 s |
| gemma-4-31B-Q4_K_M | 18.3 GB | 39 s (then OOM) | ~1.7 s load (still OOMs on forward) |

---

## Session results — what landed

| Optimization | File(s) | Status | Measured |
|---|---|---|---|
| L3 — scoped rayon pool (`devices.len()×2` threads) | `quantized_qwen35.rs` | landed | wall ~33s, neutral; principled (1=39s, 2=33s, 6=34s) |
| L5 — uninit `Vec` in `read_to_vec` | `gguf_file.rs` | landed | -200k page faults; wall flat (kernel COW from zero pages was already ~free) |
| L2-pool — thread-local pinned staging in `cuda::load_quantized` | `cuda.rs` | landed | inner GPU upload 30s → ~2s |
| L2-fused — `load_quantized_from_blob` (pread → pinned → GPU) | `cuda.rs`, `gguf_file.rs` | landed | bypasses intermediate `Vec<u8>` |
| L2-concat — `load_quantized_concat_from_blob` (fused QKV/gate-up) | `cuda.rs`, `gguf_loader.rs` | landed | eliminates CPU-side concat assembly for fused matmuls |
| stream sync after each pinned H→D | `cuda.rs` | landed | corrects pinned-buffer reuse race (`&[u8]` source skips cudarc event tracking) |

### Final measurements (Qwen3.5-27B-Q4_1, 17.17 GB, single RTX 3090)

|  | Baseline | After L3+L5+L2 | Δ |
|---|---:|---:|---:|
| Wall | 34.5 s | 32.0 s | -7 % |
| User time | 3.7 s | 1.0 s | **-73 %** |
| System time | 15.2 s | 5.7 s | **-62 %** |
| Minor page faults | 5.47 M | 3.91 M | -29 % |
| Peak RSS | 2.0 GB | 1.9 GB | flat |

The CPU/kernel work dropped sharply (sys+user 19s → 6.7s), proving the pipeline is now lean. Wall didn't drop in proportion because we now serialize on the per-tensor `stream.synchronize()` and on disk pread time — we traded one bottleneck for two smaller ones.

### What's left (next session)

To break below 20 s wall, the per-tensor stream sync needs to go away. The clean design:

- **Double-buffered pinned**: maintain two pinned buffers per thread, ping-pong. While buffer A drains H→D, buffer B starts pread for the next tensor. Only sync when wrapping back to A.
- This requires either: (a) cudarc PR for owned PinnedHostSlice slicing so the safe `memcpy_htod` tracks events, or (b) calling raw `cudarc::driver::result::memcpy_htod_async` + manual `CudaEvent::record`/`event.synchronize()`.

Estimated effort: 1 day. Projected: 17 GB load in ~5–7 s.

### Why pinned-staging didn't 5–10× wall time as projected

The L1 projection assumed pageable H→D was the bottleneck (which it was, accounting for ~30s of the 34s baseline). After pinning, that 30s collapses to ~2s — but two new bottlenecks emerge that were hidden underneath:

1. **Per-tensor stream sync** (~5–8 s for 851 tensors × ~6 ms each) — added by us to fix the pinned-buffer reuse race.
2. **Disk pread time** (~17 s, file isn't in page cache for first read; ~3 s when cached) — was previously overlapped with the slow GPU upload, now dominates because GPU upload is fast.

The forecast was correct about the GPU upload bottleneck; it just underestimated what would emerge once that ceiling was lifted.

---

## Double-buffering update — 2026-04-13 evening

### What landed
- `PinnedSlot { buf, in_flight: Option<CudaEvent> }` × 2 per thread, ping-pong via `NEXT_IDX` cell.
- `next_pinned_slot(stream, n)` syncs the slot's prior event (if any), grows the buffer geometrically, returns its index.
- `record_in_flight_event(stream, idx)` lazily allocates and recycles a single event per slot via `cuEventRecord`.
- Default event flags (no `BLOCKING_SYNC`) — busy-spin sync since most events are already complete by the next reuse.

### Result on Qwen3.5-27B-Q4_1 (17.17 GB)

| Metric | Single-buffered | Double-buffered | Δ |
|---|---:|---:|---|
| Wall | 32 s | 33 s | flat |
| Minor page faults | 3.91 M | **0.092 M** | **-98 %** |
| Voluntary ctx switches | 14 k | 14 k | flat |
| futex (strace -f) | 31 s | **11 s** | -65 % |
| pread (strace -f) | 20 s | **7.4 s** | -63 % |
| Total syscall time | 51 s | 20 s | -60 % |

The CPU-side and kernel-side metrics all improved sharply. Wall didn't move because we now hit the **disk read floor**.

### The real floor: SATA SSD bandwidth

```
$ dd if=Qwen3.5-27B-Q4_1.gguf of=/dev/null bs=64M count=64
4294967296 bytes (4.3 GB, 4.0 GiB) copied, 6.0 s, 713 MB/s
```

`/candle` is on `/dev/sda2` — a SATA SSD, not NVMe. 713 MB/s × 24 s = 17 GB. Disk reads alone account for 73 % of wall time on this machine. No amount of host-side or GPU-side optimization changes that floor.

### Where this win actually shows up

The work shipped here pays off the moment the disk stops being the bottleneck:

| Scenario | Disk BW | 17 GB read time | Total wall (with our pipeline) |
|---|---:|---:|---:|
| This machine (SATA SSD) | 0.7 GB/s | 24 s | **33 s** (measured) |
| Typical NVMe Gen3 | 3.5 GB/s | 4.9 s | ~7 s (projected) |
| NVMe Gen4 / RAID | 7 GB/s | 2.4 s | ~5 s (projected) |
| Hot page cache (no disk) | ∞ | 0 s | ~3 s (projected) |

The original "32 s → 1.5 s" projection was implicit-cache; with cold SATA reads the floor is 24 s and we're at 33 s — leaving ~9 s of overhead from CPU/sync/cudarc to attack only when disk stops dominating.

### Verifying page cache is the actual bottleneck

After `dd` pre-warms the file:
- `buff/cache` only rose from 7.1 → 7.1 GiB — **the kernel did not retain the file**
- Next candle run reports `File system inputs: 33647312` blocks ≈ 17 GB freshly read

Sandbox memory pressure (or limited cgroup) is evicting the file pages between reads. On a production box with ample RAM, this would not happen; the second-and-onwards loads would hit the cache and approach the projected ~3 s.

### Code: what's new and reusable

- `cudarc::driver::{CudaEvent, PinnedHostSlice}` API surface used directly.
- `PINNED_BUFS: thread_local![Option<[PinnedSlot; 2]>]` + `NEXT_IDX: thread_local![Cell<usize>]`.
- `next_pinned_slot` / `record_in_flight_event` helpers (private to `quantized/cuda.rs`).
- All three load-time paths (`pinned_staged_copy`, `load_quantized_from_blob`, `load_quantized_concat_from_blob`) routed through the ping-pong.

### Verified correctness

`Qwen3.5-27B-Q4_1` `--prompt "2+2="` → `"2 + 2 = **4**"` at 47 t/s prefill, 24 t/s decode. End-to-end matches the pre-optimization baseline.
