# Roadmap: CPU Overhead Elimination for Candle Decode

**Date:** 2026-04-12
**Goal:** Reduce CPU-side overhead from 46% to <10% of decode wall time
**Projected decode improvement:** TinyLlama 203 → 400+ t/s (surpass turbo's 212)

---

## Phase G1: KV Cache Migration — Tensor::cat → KvCache::append

**Estimated savings:** 660 μs/token on TinyLlama (14% of decode time)
**Complexity:** M (~200 LOC across 2 files)
**Dependencies:** None — pure Rust refactor

### Problem

`quantized_llama.rs` lines 228-229 allocate and copy the entire KV cache every decode step:
```rust
let k_t = Tensor::cat(&[k_t_cache, &k_t_new], 3)?;  // O(T) alloc + copy
let v = Tensor::cat(&[v_cache, &v], 2)?;              // O(T) alloc + copy
```

This does:
1. `alloc_uninit` for new (T+1)-sized buffer — pool lookup + potential hipMallocAsync
2. `copy_strided_src` from old cache — ucopy kernel launch
3. `copy_strided_src` from new token — ucopy kernel launch
4. Drop old cache tensor — pool return or hipFree
5. Repeat for V

Total per layer per token: 2 allocs + 4 copies + 2 drops = ~30 μs CPU + GPU copies.

### Solution

Replace with `candle_nn::kv_cache::KvCache` which pre-allocates a fixed buffer and uses `slice_set` for O(1) append (no alloc, no full copy).

### Implementation Plan

#### G1a. Add KvCache field to LayerWeights (~30 LOC)

**File:** `candle-transformers/src/models/quantized_llama.rs`

```rust
// BEFORE (line 160):
kv_cache: Option<(Tensor, Tensor)>,

// AFTER:
kv_cache: Option<candle_nn::kv_cache::KvCache>,
```

Initialize in the model constructor:
```rust
// In ModelWeights::new() or load():
kv_cache: None,  // lazy-init on first forward
```

#### G1b. Replace Tensor::cat with KvCache::append (~80 LOC)

**File:** `candle-transformers/src/models/quantized_llama.rs` lines 210-234

```rust
// BEFORE:
let k_t_new = k.transpose(2, 3)?.contiguous()?;
let (k_t, v) = match &self.kv_cache {
    None => (k_t_new, v),
    Some((k_t_cache, v_cache)) => {
        let k_t = Tensor::cat(&[k_t_cache, &k_t_new], 3)?;
        let v = Tensor::cat(&[v_cache, &v], 2)?;
        (k_t, v)
    }
};
self.kv_cache = Some((k_t.clone(), v.clone()));

// AFTER:
let k_t_new = k.transpose(2, 3)?.contiguous()?;
let cache = self.kv_cache.get_or_insert_with(|| {
    // Attack C: store K pre-transposed
    candle_nn::kv_cache::KvCache::new_k_transposed(2, 4096)
});
if index_pos == 0 {
    cache.reset();
}
cache.append(&k_t_new, &v)?;
let k_t = cache.k()?.unwrap();
let v = cache.v()?.unwrap();
```

#### G1c. Update attention dispatch (~20 LOC)

The attention call already uses `gqa_attention_k_transposed` on the HIP fast path (line 263). No change needed — just ensure the K from KvCache is in the expected `(B, n_kv_head, D, T)` layout (which `new_k_transposed` guarantees).

#### G1d. Handle cache reset on new generation (~10 LOC)

```rust
// At the start of forward(), when index_pos == 0:
if index_pos == 0 {
    for layer in self.layers.iter_mut() {
        if let Some(ref mut c) = layer.kv_cache {
            c.reset();
        }
    }
}
```

#### G1e. Verification

- Run `scripts/test-hip-quantized.sh` — ensure output matches pre-change
- Benchmark TinyLlama decode: expect ~14% improvement (203 → ~232 t/s)
- Profile: `fillBufferAligned` count should drop (fewer Tensor::cat allocs)

---

## Phase G2: Decode Op Cache — Record-and-Replay Kernel Launches

**Estimated savings:** 2000 μs/token on TinyLlama (42% of decode time)
**Complexity:** L (~600 LOC across candle-core + candle-transformers)
**Dependencies:** G1 (stable buffer addresses needed for replay)

### Problem

Every decode token re-executes the full CPU dispatch chain:
1. `storage_and_layout()` — Arc clone + RwLock read lock
2. Shape validation — dims4()?, contiguity checks
3. `func.builder()` — allocate LaunchArgs Vec
4. `builder.arg()` × 7-12 — extract raw pointers
5. `builder.launch()` — hipModuleLaunchKernel syscall
6. `from_storage()` — TensorId atomic, Arc+RwLock alloc, Layout compute
7. `BackpropOp::new()` — closure + parent Arc clones (wasted in inference)

For 155 qmatmul calls + ~100 other ops per token = ~255 full dispatch cycles × ~18 μs each = 4590 μs.

The key insight: **during decode, every token follows the exact same op sequence with the exact same shapes**. Only the data changes (new token embedding, updated KV cache pointer). All shapes, strides, kernel functions, launch configs, and most buffer addresses are identical across tokens.

### Solution

On the first decode token, record a compact "replay plan" of all kernel launches. On subsequent tokens, replay the plan directly, skipping all shape validation, storage borrows, and output allocation.

### Data Structures

#### G2a. DecodeOp — one recorded kernel launch (~40 LOC)

**File:** new `candle-core/src/hip_backend/decode_cache.rs`

```rust
/// A single recorded kernel launch for decode replay.
pub struct DecodeOp {
    /// HIP function handle (stable across tokens).
    func: sys::hipFunction_t,
    /// Launch configuration (grid, block, shared_mem).
    cfg: LaunchConfig,
    /// Arg slots. Each slot is either:
    /// - Fixed: a scalar value (i32, f32) or a stable device pointer (weight buffer)
    /// - Dynamic: an index into the decode state's buffer table
    arg_slots: Vec<ArgSlot>,
    /// Index of the output buffer in the buffer table.
    output_slot: usize,
}

enum ArgSlot {
    /// Value is fixed across tokens (scalar, weight pointer, dimension).
    Fixed(*mut c_void),
    /// Value comes from buffer_table[idx] at replay time.
    Buffer(usize),
}
```

#### G2b. DecodePlan — the full recorded forward pass (~60 LOC)

```rust
/// Recorded decode forward pass for replay.
pub struct DecodePlan {
    /// Sequence of kernel launches.
    ops: Vec<DecodeOp>,
    /// Pre-allocated GPU buffers for all intermediate tensors.
    /// Indexed by slot number. Stable addresses across tokens.
    buffer_table: Vec<HipSlice<u8>>,
    /// Slot indices for the dynamic inputs (token embedding, KV cache pointers).
    input_slots: Vec<usize>,
    /// Slot index for the final output (logits).
    output_slot: usize,
    /// The model's hidden_dim, n_layers, etc. for validation.
    model_hash: u64,
}
```

#### G2c. Recording phase — intercept kernel launches (~200 LOC)

Add a thread-local recording mode to `LaunchArgs::launch()`:

```rust
thread_local! {
    static DECODE_RECORDING: RefCell<Option<Vec<DecodeOp>>> = RefCell::new(None);
}

impl<'a> LaunchArgs<'a> {
    pub unsafe fn launch(self, cfg: LaunchConfig) -> Result<(), DriverError> {
        // Normal launch
        check_hip(sys::hipModuleLaunchKernel(...))?;
        
        // If recording, capture this launch
        DECODE_RECORDING.with(|rec| {
            if let Some(ref mut ops) = *rec.borrow_mut() {
                ops.push(DecodeOp {
                    func: self.func.raw,
                    cfg,
                    arg_slots: self.args.iter().map(|p| ArgSlot::Fixed(*p)).collect(),
                    output_slot: 0, // filled in by the recording wrapper
                });
            }
        });
        Ok(())
    }
}
```

#### G2d. Replay phase — direct kernel launch loop (~100 LOC)

```rust
impl DecodePlan {
    pub fn replay(&self, stream: &HipStream) -> Result<()> {
        for op in &self.ops {
            // Build args array from slots
            let args: Vec<*mut c_void> = op.arg_slots.iter().map(|slot| {
                match slot {
                    ArgSlot::Fixed(ptr) => *ptr,
                    ArgSlot::Buffer(idx) => self.buffer_table[*idx].device_ptr() as *mut c_void,
                }
            }).collect();
            
            // Direct hipModuleLaunchKernel — no builder, no validation
            unsafe {
                check_hip(sys::hipModuleLaunchKernel(
                    op.func,
                    op.cfg.grid_dim.0, op.cfg.grid_dim.1, op.cfg.grid_dim.2,
                    op.cfg.block_dim.0, op.cfg.block_dim.1, op.cfg.block_dim.2,
                    op.cfg.shared_mem_bytes,
                    stream.raw,
                    args.as_ptr() as *mut *mut c_void,
                    std::ptr::null_mut(),
                ))?;
            }
        }
        Ok(())
    }
}
```

#### G2e. Integration with model forward (~100 LOC)

**File:** `candle-transformers/src/models/quantized_llama.rs`

```rust
impl ModelWeights {
    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let is_decode = x.dims2()?.1 == 1 && index_pos > 0;
        
        if is_decode {
            if let Some(ref plan) = self.decode_plan {
                // REPLAY: update dynamic inputs, launch all kernels
                plan.update_input(0, &x)?;  // new token embedding
                plan.replay(&self.stream)?;
                return plan.read_output();
            }
            // RECORD: first decode token, capture the plan
            start_recording();
        }
        
        // ... existing forward pass ...
        
        if is_decode && is_recording() {
            self.decode_plan = Some(stop_recording_and_build_plan()?);
        }
        
        result
    }
}
```

#### G2f. Dynamic input handling (~60 LOC)

The decode plan needs to handle inputs that change per token:
1. **Token embedding** — changes every token (new token ID → different embedding vector)
2. **KV cache pointers** — may change if cache grows (but KvCache::append reuses buffer)
3. **Attention mask** — may change shape for sliding window

For (1): the embedding lookup kernel's output buffer is a dynamic slot.
For (2): with KvCache using pre-allocated buffers, pointers are stable. Only the "current_seq_len" parameter changes — patch this scalar in the plan.
For (3): cache the mask or recompute outside the plan.

### Verification

- Compare replay output vs fresh-forward output for 100 tokens — must match bit-for-bit
- Benchmark: expect ~42% decode improvement (232 → ~380 t/s after G1+G2)
- Edge cases: first token (no plan), sequence length change, multi-GPU

---

## Phase G3: HIP Graph Capture for Decode

**Estimated savings:** additional 500 μs/token (eliminates remaining launch overhead)
**Complexity:** XL (~400 LOC in hipdarc + candle-core)
**Dependencies:** G1 + G2 (stable buffer addresses required for graph capture)

### Problem

Even with the op cache (G2), each kernel launch still goes through `hipModuleLaunchKernel` — a syscall with ~0.5 μs overhead. With ~255 launches per token, that's ~128 μs of irreducible syscall overhead.

HIP graphs can replay all 255 launches from a single `hipGraphLaunch` call, reducing the total syscall overhead to ~1 μs.

### Why HIP Graphs Failed Before

The REVIEW doc notes HIP graphs were "counter-productive" because `hipMallocAsync` graph nodes serialize on the runtime pool lock. This is the key constraint:

**HIP graphs CANNOT contain allocation nodes.** All buffers must be pre-allocated before capture.

With G1 (KvCache pre-alloc) and G2 (decode plan with pre-allocated buffer table), this constraint is satisfied.

### Implementation Plan

#### G3a. Add HIP Graph API bindings to hipdarc (~100 LOC)

**File:** `hipdarc/src/driver.rs` (new section)

```rust
pub struct HipGraph {
    raw: sys::hipGraph_t,
}

pub struct HipGraphExec {
    raw: sys::hipGraphExec_t,
}

impl HipStream {
    /// Begin capturing kernel launches into a graph.
    pub fn begin_capture(&self) -> Result<(), DriverError> {
        unsafe {
            check_hip(sys::hipStreamBeginCapture(
                self.raw,
                sys::hipStreamCaptureMode::hipStreamCaptureModeGlobal,
            ))
        }
    }
    
    /// End capture and return the recorded graph.
    pub fn end_capture(&self) -> Result<HipGraph, DriverError> {
        let mut graph = std::ptr::null_mut();
        unsafe {
            check_hip(sys::hipStreamEndCapture(self.raw, &mut graph))?;
        }
        Ok(HipGraph { raw: graph })
    }
}

impl HipGraph {
    /// Instantiate the graph for execution.
    pub fn instantiate(&self) -> Result<HipGraphExec, DriverError> {
        let mut exec = std::ptr::null_mut();
        unsafe {
            check_hip(sys::hipGraphInstantiate(
                &mut exec,
                self.raw,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            ))?;
        }
        Ok(HipGraphExec { raw: exec })
    }
}

impl HipGraphExec {
    /// Launch the instantiated graph on a stream.
    pub fn launch(&self, stream: &HipStream) -> Result<(), DriverError> {
        unsafe {
            check_hip(sys::hipGraphLaunch(self.raw, stream.raw))
        }
    }
    
    /// Update a kernel node's parameters (for changing scalars between replays).
    pub fn update_kernel_node(
        &self,
        node: sys::hipGraphNode_t,
        params: &sys::hipKernelNodeParams,
    ) -> Result<(), DriverError> {
        unsafe {
            check_hip(sys::hipGraphExecKernelNodeSetParams(
                self.raw, node, params,
            ))
        }
    }
}

// Drop impls for cleanup
impl Drop for HipGraph {
    fn drop(&mut self) {
        unsafe { sys::hipGraphDestroy(self.raw); }
    }
}

impl Drop for HipGraphExec {
    fn drop(&mut self) {
        unsafe { sys::hipGraphExecDestroy(self.raw); }
    }
}
```

#### G3b. Add hipGraph sys bindings (~40 LOC)

**File:** `hipdarc/src/sys.rs` (add extern declarations)

```rust
extern "C" {
    pub fn hipStreamBeginCapture(
        stream: hipStream_t,
        mode: hipStreamCaptureMode,
    ) -> hipError_t;
    
    pub fn hipStreamEndCapture(
        stream: hipStream_t,
        pGraph: *mut hipGraph_t,
    ) -> hipError_t;
    
    pub fn hipGraphInstantiate(
        pGraphExec: *mut hipGraphExec_t,
        graph: hipGraph_t,
        pErrorNode: *mut hipGraphNode_t,
        pLogBuffer: *mut c_char,
        bufferSize: usize,
    ) -> hipError_t;
    
    pub fn hipGraphLaunch(
        graphExec: hipGraphExec_t,
        stream: hipStream_t,
    ) -> hipError_t;
    
    pub fn hipGraphDestroy(graph: hipGraph_t) -> hipError_t;
    pub fn hipGraphExecDestroy(graphExec: hipGraphExec_t) -> hipError_t;
    pub fn hipGraphExecKernelNodeSetParams(
        graphExec: hipGraphExec_t,
        node: hipGraphNode_t,
        pNodeParams: *const hipKernelNodeParams,
    ) -> hipError_t;
}

pub type hipGraph_t = *mut c_void;
pub type hipGraphExec_t = *mut c_void;
pub type hipGraphNode_t = *mut c_void;

#[repr(C)]
pub enum hipStreamCaptureMode {
    hipStreamCaptureModeGlobal = 0,
    hipStreamCaptureModeThreadLocal = 1,
    hipStreamCaptureModeRelaxed = 2,
}
```

#### G3c. Capture decode forward as HIP graph (~100 LOC)

**File:** `candle-core/src/hip_backend/decode_cache.rs`

```rust
impl DecodePlan {
    /// Capture the replay sequence as a HIP graph.
    /// Requires all buffers in buffer_table to be pre-allocated (no alloc during capture).
    pub fn capture_graph(&self, stream: &HipStream) -> Result<HipGraphExec> {
        stream.begin_capture()?;
        
        // Replay all ops — these are captured into the graph
        self.replay(stream)?;
        
        let graph = stream.end_capture()?;
        let exec = graph.instantiate()?;
        Ok(exec)
    }
    
    /// Execute the captured graph (single API call for ALL kernels).
    pub fn launch_graph(&self, exec: &HipGraphExec, stream: &HipStream) -> Result<()> {
        exec.launch(stream)
    }
}
```

#### G3d. Integration with model forward (~60 LOC)

```rust
impl ModelWeights {
    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let is_decode = x.dims2()?.1 == 1 && index_pos > 0;
        
        if is_decode {
            // Phase 1: try graph replay (fastest)
            if let Some(ref graph_exec) = self.graph_exec {
                self.decode_plan.as_ref().unwrap().update_input(0, &x)?;
                self.decode_plan.as_ref().unwrap().launch_graph(graph_exec, &self.stream)?;
                return self.decode_plan.as_ref().unwrap().read_output();
            }
            
            // Phase 2: try plan replay (fast, builds graph on second call)
            if let Some(ref plan) = self.decode_plan {
                plan.update_input(0, &x)?;
                plan.replay(&self.stream)?;
                // Capture graph for next token
                self.graph_exec = Some(plan.capture_graph(&self.stream)?);
                return plan.read_output();
            }
            
            // Phase 3: first decode token — record plan
            start_recording();
        }
        
        // ... existing forward pass ...
    }
}
```

#### G3e. Handling parameter updates between graph replays (~50 LOC)

Between graph replays, only a few values change:
- **Embedding output** — the input token changes, so the embedding lookup output changes
- **KV cache seq_len** — increments by 1

For the embedding: run the embedding lookup OUTSIDE the graph (it's a single kernel), then the graph starts from the first layer.

For the KV seq_len: use `hipGraphExecKernelNodeSetParams` to patch the scalar argument in the relevant kernel nodes.

### Verification

- Graph replay must produce bit-identical output to fresh forward
- Benchmark: expect ~10% additional improvement over G2 (380 → ~420 t/s)
- Edge cases: graph invalidation on sequence length bucket change, multi-GPU graph

---

## Priority Execution Order

| # | Phase | Item | Complexity | Savings/token | Cumulative t/s |
|---|-------|------|------------|---------------|----------------|
| 1 | G1 | KvCache migration (llama) | M (~200 LOC) | 660 μs (14%) | ~232 t/s |
| 2 | G2 | Decode op cache | L (~600 LOC) | 2000 μs (42%) | ~380 t/s |
| 3 | G3 | HIP graph capture | XL (~400 LOC) | 500 μs (10%) | ~420 t/s |

**Target:** TinyLlama decode 420 t/s = **2.0x turbo's 212 t/s**.

---

## Verification Protocol

After each phase:
1. Run `scripts/test-hip-quantized.sh` — correctness check
2. Benchmark TinyLlama + gemma4-E4B + Qwen3.5-9B — decode t/s
3. Profile with `--tracing` — verify CPU self-time reduction
4. Profile with `rocprofv3 --kernel-trace` — verify dispatch count reduction
5. Compare output with pre-change baseline — must match to within f32 tolerance

---

## Risk Analysis

| Risk | Phase | Mitigation |
|------|-------|------------|
| KvCache resize invalidates graph | G3 | Detect resize, re-capture graph |
| Multi-GPU pipeline breaks graph | G3 | Graph per GPU, sync between graphs |
| Dynamic shapes (variable seq_len) | G2/G3 | Bucket by shape, one plan per bucket |
| BackpropOp removal breaks training | G2 | Guard with inference_mode flag |
| Buffer pool miss during record | G2 | Pre-warm pool with one forward pass |
