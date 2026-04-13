//! G2: Decode op cache — record-and-replay kernel launches.
//!
//! Phase 1 records all kernel launches into a `DecodePlan`.
//! Phase 2 (this file) replays the plan directly via
//! `hipModuleLaunchKernel`, skipping all Rust tensor overhead.
//!
//! Dynamic args (token input pointer, index_pos counters) are
//! advanced/patched between replays. Stable buffer addresses are
//! guaranteed by `hipdarc::driver::decode_alloc_*`.

use std::cell::RefCell;
use std::ffi::c_void;

use super::device::HipDevice;
use crate::hip_backend::hipdarc;
use hipdarc::driver::{set_launch_recorder, LaunchRecorderFn};

// ============================================================================
// Op recording (Phase 1 — unchanged)
// ============================================================================

/// A single recorded kernel launch.
#[derive(Clone, Debug)]
pub struct RecordedOp {
    pub func: hipdarc::sys::hipFunction_t,
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem: u32,
    pub stream: hipdarc::sys::hipStream_t,
    /// Raw arg values captured at record time, held in a u64 buffer but
    /// only `arg_sizes[i]` bytes are valid per arg. Pointers use 8 bytes,
    /// int32 args use 4 bytes (low bits). Storing in u64 keeps replay
    /// indirection simple (arg_ptrs point into this vec).
    pub arg_values: Vec<u64>,
    /// Byte size of each argument as declared by the kernel signature.
    pub arg_sizes: Vec<usize>,
}

// Thread-local recording state.
thread_local! {
    static RECORDING: RefCell<Option<Vec<RecordedOp>>> = RefCell::new(None);
}

/// Start recording kernel launches on the current thread.
pub fn start_recording() {
    RECORDING.with(|r| {
        *r.borrow_mut() = Some(Vec::with_capacity(512));
    });

    let recorder: LaunchRecorderFn = Box::new(|func, grid, block, shared, stream, args, sizes| {
        maybe_record(func, grid, block, shared, stream, args, sizes);
    });
    set_launch_recorder(Some(recorder));
}

/// Stop recording and return the captured ops.
pub fn stop_recording() -> Option<Vec<RecordedOp>> {
    set_launch_recorder(None);
    RECORDING.with(|r| r.borrow_mut().take())
}

/// Check if recording is active.
pub fn is_recording() -> bool {
    RECORDING.with(|r| r.borrow().is_some())
}

/// Called from LaunchArgs::launch() to record a kernel launch.
/// Reads each arg according to its actual byte size (from the kernel's
/// arg declaration). Avoids over-reading scalar int32 args into adjacent
/// stack memory (the "fake 2^32 delta" bug).
pub fn maybe_record(
    func: hipdarc::sys::hipFunction_t,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem: u32,
    stream: hipdarc::sys::hipStream_t,
    args: &[*mut c_void],
    sizes: &[usize],
) -> bool {
    RECORDING.with(|r| {
        let mut rec = r.borrow_mut();
        if let Some(ref mut ops) = *rec {
            debug_assert_eq!(args.len(), sizes.len(),
                "maybe_record: args/sizes length mismatch ({} vs {})",
                args.len(), sizes.len());
            let arg_values: Vec<u64> = args.iter().zip(sizes.iter())
                .map(|(p, &sz)| unsafe {
                    match sz {
                        4 => *(*p as *const u32) as u64,
                        8 => *(*p as *const u64),
                        // Fallback for unusual sizes — zero-extend the
                        // low bytes.
                        n if n <= 8 => {
                            let mut buf = [0u8; 8];
                            std::ptr::copy_nonoverlapping(*p as *const u8, buf.as_mut_ptr(), n);
                            u64::from_le_bytes(buf)
                        }
                        _ => 0,
                    }
                })
                .collect();
            ops.push(RecordedOp {
                func,
                grid,
                block,
                shared_mem,
                stream,
                arg_values,
                arg_sizes: sizes.to_vec(),
            });
            true
        } else {
            false
        }
    })
}

// ============================================================================
// DecodePlan — Phase 2: replay with delta-based dynamic arg patching
// ============================================================================

/// How to handle a dynamic arg on each replay tick.
#[derive(Debug, Clone)]
enum DynArgKind {
    /// Increment by a fixed delta each token (e.g. index_pos += 1).
    Counter(i64),
    /// Externally patched pointer (e.g. input token embedding device ptr).
    /// Patched via `DecodePlan::patch_externals()`.
    External,
}

/// A (op_index, arg_index) pair identifying one dynamic arg slot.
#[derive(Debug, Clone, Copy)]
pub struct PatchLoc {
    pub op: usize,
    pub arg: usize,
}

/// A recorded decode forward pass, ready for replay.
#[derive(Debug, Clone)]
pub struct DecodePlan {
    ops: Vec<RecordedOp>,
    /// For each op, which arg indices are "dynamic" (change between tokens).
    dynamic_args: Vec<Vec<usize>>,
    /// For each (op, arg) in dynamic_args, how to advance it.
    dynamic_kinds: Vec<Vec<DynArgKind>>,
    /// Locations of external-patch args (input pointer, etc).
    /// Caller patches these explicitly before replay.
    external_patch_locs: Vec<PatchLoc>,
    /// Total number of fixed (stable) args.
    fixed_count: usize,
    /// Device pointer of the output buffer (last op's result).
    output_ptr: usize,
    /// Number of f32 elements in the output buffer.
    output_f32_count: usize,
    /// Shape of the output tensor (e.g. [1, vocab_size]).
    output_shape: Vec<usize>,
    /// How many times replay has been called (for step limiting).
    replay_count: usize,
    /// Input-tensor device pointer captured at recording time. At replay
    /// time the caller memcpys the new token-id bytes into this address
    /// so downstream ops see fresh indices without pointer patching.
    pub input_anchor_ptr: usize,
    /// Byte size of the input tensor to memcpy on each replay.
    pub input_anchor_bytes: usize,
}

impl DecodePlan {
    /// Build a plan by comparing two recordings from consecutive decode tokens.
    /// `output_ptr` / `output_f32_count` describe the result tensor from the
    /// *second* recording's forward pass (stable-alloc buffer).
    ///
    /// `external_ptrs` is a list of pointer values (e.g. the input-tensor
    /// device pointer for the first and second recordings). Any arg whose
    /// value matches one of these in either recording is marked as an
    /// External patch slot rather than a Counter — the caller must supply
    /// a fresh value via `patch_externals()` before each replay.
    pub fn from_two_recordings(
        first: &[RecordedOp],
        second: &[RecordedOp],
        output_ptr: usize,
        output_f32_count: usize,
        output_shape: Vec<usize>,
    ) -> Option<Self> {
        Self::from_two_recordings_with_externals(
            first, second, output_ptr, output_f32_count, output_shape, &[],
        )
    }

    pub fn from_two_recordings_with_externals(
        first: &[RecordedOp],
        second: &[RecordedOp],
        output_ptr: usize,
        output_f32_count: usize,
        output_shape: Vec<usize>,
        external_ptrs: &[usize],
    ) -> Option<Self> {
        let debug_ext = std::env::var("CANDLE_G2_EXT_DEBUG").is_ok();
        if debug_ext {
            eprintln!("[G2] external candidates: {:?}",
                external_ptrs.iter().map(|p| format!("0x{:x}", p)).collect::<Vec<_>>());
            // Search all ops for args matching any external pointer.
            let mut matches = Vec::new();
            for (op_i, op) in second.iter().enumerate() {
                for (arg_i, v) in op.arg_values.iter().enumerate() {
                    if external_ptrs.iter().any(|p| (*p as u64) == *v) {
                        matches.push((op_i, arg_i, *v));
                    }
                }
            }
            eprintln!("[G2] args matching external_ptrs exactly: {}", matches.len());
            for (op_i, arg_i, v) in matches.iter().take(10) {
                eprintln!("  op[{}] arg[{}] = 0x{:x}", op_i, arg_i, v);
            }
            // Also: find args that differ between a and b, showing delta.
            let mut diffs = Vec::new();
            for (op_i, (a, b)) in first.iter().zip(second.iter()).enumerate() {
                for (arg_i, (va, vb)) in a.arg_values.iter().zip(b.arg_values.iter()).enumerate() {
                    if va != vb {
                        diffs.push((op_i, arg_i, *va, *vb));
                    }
                }
            }
            eprintln!("[G2] dynamic arg summary (all {} diffs):", diffs.len());
            // Show delta histogram
            let mut delta_counts = std::collections::HashMap::<i64, usize>::new();
            for (_, _, va, vb) in &diffs {
                let d = *vb as i64 - *va as i64;
                *delta_counts.entry(d).or_insert(0) += 1;
            }
            let mut dc: Vec<_> = delta_counts.iter().collect();
            dc.sort_by_key(|(_, cnt)| std::cmp::Reverse(**cnt));
            eprintln!("  top delta values (delta -> count):");
            for (d, c) in dc.iter().take(10) {
                eprintln!("    delta={} ({:#x}) → {} args", d, **d as u64, c);
            }
            // Show samples for the largest deltas
            let mut by_abs: Vec<_> = diffs.iter().collect();
            by_abs.sort_by_key(|(_, _, va, vb)| std::cmp::Reverse((*vb as i64 - *va as i64).abs()));
            eprintln!("  largest |delta| args (first 5):");
            for (op_i, arg_i, va, vb) in by_abs.iter().take(5) {
                let delta = *vb as i64 - *va as i64;
                let sz = second[*op_i].arg_sizes.get(*arg_i).copied().unwrap_or(0);
                eprintln!("    op[{}] arg[{}] size={} {:#x} -> {:#x} (delta={})",
                    op_i, arg_i, sz, va, vb, delta);
            }
        }
        if first.len() != second.len() {
            return None;
        }

        let ops = second.to_vec();
        let mut dynamic_args = Vec::with_capacity(ops.len());
        let mut dynamic_kinds = Vec::with_capacity(ops.len());
        let mut external_patch_locs: Vec<PatchLoc> = Vec::new();

        for (i, (a, b)) in first.iter().zip(second.iter()).enumerate() {
            if a.func != b.func || a.arg_values.len() != b.arg_values.len() {
                return None;
            }

            let mut dyn_indices = Vec::new();
            let mut dyn_kinds = Vec::new();
            for (j, (va, vb)) in a.arg_values.iter().zip(b.arg_values.iter()).enumerate() {
                if va != vb {
                    // If either recording's value matches a registered
                    // external pointer, treat this arg as externally
                    // patched rather than a counter.
                    let is_external = external_ptrs.iter().any(|p| (*p as u64) == *va || (*p as u64) == *vb);
                    if is_external {
                        external_patch_locs.push(PatchLoc { op: i, arg: j });
                        dyn_indices.push(j);
                        dyn_kinds.push(DynArgKind::External);
                    } else {
                        let delta = *vb as i64 - *va as i64;
                        dyn_indices.push(j);
                        dyn_kinds.push(DynArgKind::Counter(delta));
                    }
                }
            }
            dynamic_args.push(dyn_indices);
            dynamic_kinds.push(dyn_kinds);
        }

        let dynamic_count: usize = dynamic_args.iter().map(|d| d.len()).sum();
        let total_args: usize = ops.iter().map(|o| o.arg_values.len()).sum();
        let fixed_count = total_args - dynamic_count;

        let counter_count = dynamic_kinds.iter().flat_map(|v| v.iter()).filter(|k| matches!(k, DynArgKind::Counter(_))).count();
        let external_count = external_patch_locs.len();
        let ops_fixed = dynamic_args.iter().filter(|d| d.is_empty()).count();
        let ops_dyn1 = dynamic_args.iter().filter(|d| d.len() == 1).count();
        let ops_dyn2p = dynamic_args.iter().filter(|d| d.len() >= 2).count();

        eprintln!(
            "[G2] plan: {} ops, {}/{} args fixed/dynamic (counters={}, external={})",
            ops.len(), fixed_count, dynamic_count, counter_count, external_count
        );
        eprintln!(
            "[G2] ops: {} fixed + {} with 1 dyn + {} with 2+ dyn",
            ops_fixed, ops_dyn1, ops_dyn2p
        );
        for loc in &external_patch_locs {
            eprintln!(
                "[G2] external patch: op[{}] arg[{}] = 0x{:x}",
                loc.op, loc.arg, ops[loc.op].arg_values[loc.arg]
            );
        }

        // Record-time input pointer: use the second recording's value
        // (that's what's embedded in the captured ops). Bytes = 0 until
        // the caller sets it via set_input_anchor().
        let input_anchor_ptr = external_ptrs.get(1).copied().unwrap_or(0);

        Some(Self {
            ops,
            dynamic_args,
            dynamic_kinds,
            external_patch_locs,
            fixed_count,
            output_ptr,
            output_f32_count,
            output_shape,
            replay_count: 0,
            input_anchor_ptr,
            input_anchor_bytes: 0,
        })
    }

    /// Set the byte size of the input tensor. The caller memcpys this many
    /// bytes from the new x into `input_anchor_ptr` on each replay.
    pub fn set_input_anchor_bytes(&mut self, bytes: usize) {
        self.input_anchor_bytes = bytes;
    }

    /// How many replay steps have been executed.
    pub fn replay_count(&self) -> usize {
        self.replay_count
    }

    /// Advance all counter-type dynamic args by their delta.
    /// Call this once before each replay (tokens 5, 6, 7, ...).
    pub fn advance_counters(&mut self) {
        self.replay_count += 1;
        let skip_ptr = std::env::var("CANDLE_G2_SKIP_PTR_ADVANCE").is_ok();
        let skip_scalar = std::env::var("CANDLE_G2_SKIP_SCALAR_ADVANCE").is_ok();
        let skip_delta: Option<i64> = std::env::var("CANDLE_G2_SKIP_DELTA")
            .ok().and_then(|s| s.parse().ok());
        for (i, dyn_indices) in self.dynamic_args.iter().enumerate() {
            for (k, &arg_idx) in dyn_indices.iter().enumerate() {
                if let DynArgKind::Counter(delta) = self.dynamic_kinds[i][k] {
                    let sz = self.ops[i].arg_sizes.get(arg_idx).copied().unwrap_or(8);
                    if skip_ptr && sz == 8 { continue; }
                    if skip_scalar && sz <= 4 { continue; }
                    if Some(delta) == skip_delta { continue; }
                    let val = self.ops[i].arg_values[arg_idx];
                    self.ops[i].arg_values[arg_idx] = (val as i64 + delta) as u64;
                }
            }
        }
    }

    /// Patch external dynamic args. `values` must have exactly
    /// `self.external_patch_count()` entries, in registration order.
    pub fn patch_externals(&mut self, values: &[usize]) {
        debug_assert_eq!(
            values.len(),
            self.external_patch_locs.len(),
            "patch_externals: expected {} values, got {}",
            self.external_patch_locs.len(),
            values.len()
        );
        for (loc, &val) in self.external_patch_locs.iter().zip(values.iter()) {
            self.ops[loc.op].arg_values[loc.arg] = val as u64;
        }
    }

    /// Patch every external slot to the same pointer. Used when all
    /// externals refer to a single logical input (the decode token id
    /// tensor).
    pub fn patch_all_externals(&mut self, value: usize) {
        for loc in &self.external_patch_locs {
            self.ops[loc.op].arg_values[loc.arg] = value as u64;
        }
    }

    /// Number of external-patch slots the caller must supply.
    pub fn external_patch_count(&self) -> usize {
        self.external_patch_locs.len()
    }

    /// Replay the plan: launch every recorded kernel.
    ///
    /// # Safety
    /// All buffer addresses in the plan must be valid (decode-alloc
    /// sentinel keeps them alive). Dynamic args must have been
    /// advanced/patched before calling this.
    pub unsafe fn replay(&self, dev: &HipDevice) -> crate::Result<()> {
        let stream = dev.stream();
        let raw_stream = stream.raw();
        let debug = std::env::var("CANDLE_G2_DEBUG").is_ok();

        for (idx, op) in self.ops.iter().enumerate() {
            let mut arg_ptrs: Vec<*mut c_void> = op
                .arg_values
                .iter()
                .map(|v| v as *const u64 as *mut c_void)
                .collect();

            let rc = hipdarc::sys::hipModuleLaunchKernel(
                op.func,
                op.grid.0, op.grid.1, op.grid.2,
                op.block.0, op.block.1, op.block.2,
                op.shared_mem,
                raw_stream,
                arg_ptrs.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if rc != hipdarc::sys::hipError_t::hipSuccess {
                crate::bail!("decode_cache replay: op[{}] hipModuleLaunchKernel failed with {:?}", idx, rc);
            }
            if debug {
                let src = stream.synchronize();
                if src.is_err() {
                    crate::bail!(
                        "decode_cache replay: op[{}] sync failed (replay#{}, grid=({},{},{}), block=({},{},{}))",
                        idx, self.replay_count,
                        op.grid.0, op.grid.1, op.grid.2,
                        op.block.0, op.block.1, op.block.2,
                    );
                }
            }
        }

        Ok(())
    }

    /// Capture the replay as a HIP graph for single-call execution (G3).
    ///
    /// # Safety
    /// Same requirements as `replay()`.
    pub unsafe fn capture_graph(&self, dev: &HipDevice) -> crate::Result<hipdarc::driver::HipGraphExec> {
        let (exec, _g, _n) = self.capture_graph_full(dev)?;
        Ok(exec)
    }

    /// Capture + return `(HipGraphExec, HipGraph, nodes)`. Nodes stay valid
    /// as long as `HipGraph` is alive, which is why the caller must hold
    /// all three (`DecodeGraph` does this).
    pub unsafe fn capture_graph_full(
        &self,
        dev: &HipDevice,
    ) -> crate::Result<(
        hipdarc::driver::HipGraphExec,
        hipdarc::driver::HipGraph,
        Vec<hipdarc::sys::hipGraphNode_t>,
    )> {
        let stream = dev.stream();
        let graph = hipdarc::driver::with_capture(stream, || {
            let raw_stream = stream.raw();
            for op in &self.ops {
                let mut arg_ptrs: Vec<*mut c_void> = op
                    .arg_values
                    .iter()
                    .map(|v| v as *const u64 as *mut c_void)
                    .collect();
                let rc = hipdarc::sys::hipModuleLaunchKernel(
                    op.func,
                    op.grid.0, op.grid.1, op.grid.2,
                    op.block.0, op.block.1, op.block.2,
                    op.shared_mem,
                    raw_stream,
                    arg_ptrs.as_mut_ptr(),
                    std::ptr::null_mut(),
                );
                if rc != hipdarc::sys::hipError_t::hipSuccess {
                    return Err(hipdarc::error::DriverError::Hip(rc));
                }
            }
            Ok(())
        }).map_err(|e| crate::Error::wrap(e))?;
        let nodes = graph.nodes().map_err(|e| crate::Error::wrap(e))?;
        if nodes.len() != self.ops.len() {
            crate::bail!(
                "capture_graph: node count {} != op count {}",
                nodes.len(), self.ops.len(),
            );
        }
        let exec = graph.instantiate().map_err(|e| crate::Error::wrap(e))?;
        Ok((exec, graph, nodes))
    }

    /// Advance counter-type dynamic args AND rebuild stable kernelParams
    /// pointer-of-pointers arrays for every dynamic op. Call this before
    /// each graph launch when using `DecodeGraph`. The caller then
    /// dispatches `DecodeGraph::patch_and_launch(self)` using those
    /// pointer arrays.
    ///
    /// The kernel-param pointer arrays must OUTLIVE the graph launch, so
    /// this returns stable borrows owned by `self.ops[op].arg_values`.
    #[doc(hidden)]
    pub fn dynamic_op_indices(&self) -> Vec<usize> {
        self.dynamic_args
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.is_empty())
            .map(|(i, _)| i)
            .collect()
    }

    /// Create a Tensor wrapping the output buffer. The buffer is owned
    /// by the decode allocator (sentinel-marked) so it stays alive.
    /// Content changes on each replay but the metadata is stable.
    pub fn output_tensor(&self, dev: &HipDevice) -> crate::Result<crate::Tensor> {
        use crate::{Shape, Storage, Tensor};
        use super::HipStorage;
        use hipdarc::driver::HipSlice;

        // Create a sentinel-marked HipSlice that won't free on drop.
        let slice: HipSlice<f32> = unsafe {
            HipSlice::decode_view(
                self.output_ptr as hipdarc::sys::hipDeviceptr_t,
                self.output_f32_count,
            )
        };
        let storage = HipStorage::wrap_hip_slice(slice, dev.clone());
        let shape: Shape = Shape::from_dims(&self.output_shape);
        Ok(Tensor::from_storage(
            Storage::Hip(storage),
            shape,
            crate::op::BackpropOp::none(),
            false,
        ))
    }

    /// Number of recorded ops.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Number of dynamic args across all ops.
    pub fn dynamic_arg_count(&self) -> usize {
        self.dynamic_args.iter().map(|d| d.len()).sum()
    }

    /// Number of fixed args (stable across tokens).
    pub fn fixed_arg_count(&self) -> usize {
        self.fixed_count
    }

    /// Get the recorded ops for external analysis.
    pub fn ops(&self) -> &[RecordedOp] {
        &self.ops
    }

    /// Get the dynamic arg indices for each op.
    pub fn dynamic_args(&self) -> &[Vec<usize>] {
        &self.dynamic_args
    }

    /// Device pointer of the output buffer.
    pub fn output_ptr(&self) -> usize {
        self.output_ptr
    }
}

// ============================================================================
// DecodeGraph — captured HIP graph + per-op node handles + stable
// kernelParams storage, so we can update the ~130 dynamic kernel-node
// parameters on every replay tick without re-capturing.
// ============================================================================

/// A captured + instantiated HIP graph plus the info needed to mutate
/// dynamic kernel nodes before each launch.
///
/// We keep `_graph` alive so the node handles remain valid; the nodes in
/// `exec` and `_graph` refer to the same underlying graph instances.
/// `arg_ptr_storage[op_idx]` is a per-op Vec whose entries are pointers
/// into `plan.ops[op_idx].arg_values`. These pointers are stable for the
/// lifetime of the plan (Vec<u64> storage isn't re-allocated once built).
pub struct DecodeGraph {
    pub exec: hipdarc::driver::HipGraphExec,
    _graph: hipdarc::driver::HipGraph,
    nodes: Vec<hipdarc::sys::hipGraphNode_t>,
    /// One entry per op; stores *mut c_void pointing to each of the op's
    /// u64 arg values. Kept alive so `hipGraphExecKernelNodeSetParams` can
    /// dereference the pointers safely when HIP internally reads them.
    arg_ptr_storage: Vec<Vec<*mut c_void>>,
    /// op indices that have at least one dynamic arg (cached).
    dynamic_ops: Vec<usize>,
}

// SAFETY: The internal HIP handles are thread-safe; the arg_ptr_storage
// pointers refer to plan.arg_values which is pinned inside Rc/Arc in the
// caller (DecodeState::Graph holds the plan by value).
unsafe impl Send for DecodeGraph {}
unsafe impl Sync for DecodeGraph {}

impl DecodeGraph {
    /// Build a `DecodeGraph` by capturing the plan. Must be called with
    /// decode-alloc in replay mode so captured pointers match the
    /// subsequent launch-time addresses.
    ///
    /// # Safety
    /// See `DecodePlan::capture_graph_full`.
    pub unsafe fn capture(plan: &DecodePlan, dev: &super::device::HipDevice) -> crate::Result<Self> {
        let (exec, graph, nodes) = plan.capture_graph_full(dev)?;
        // Build stable arg-pointer arrays for every op.
        let arg_ptr_storage: Vec<Vec<*mut c_void>> = plan
            .ops
            .iter()
            .map(|op| {
                op.arg_values
                    .iter()
                    .map(|v| v as *const u64 as *mut c_void)
                    .collect()
            })
            .collect();
        let dynamic_ops = plan.dynamic_op_indices();
        eprintln!(
            "[G3] DecodeGraph: {} nodes, {} dynamic ops to patch per launch",
            nodes.len(), dynamic_ops.len()
        );
        Ok(DecodeGraph {
            exec,
            _graph: graph,
            nodes,
            arg_ptr_storage,
            dynamic_ops,
        })
    }

    /// Patch every dynamic op's kernel-node parameters with the plan's
    /// current arg_values, then launch the graph. Assumes the caller has
    /// already called `plan.advance_counters()` and `plan.patch_all_externals()`.
    ///
    /// # Safety
    /// `plan` must be the same plan that was used in `capture`. The HIP
    /// graph reads from `self.arg_ptr_storage`, which points into
    /// `plan.ops[i].arg_values` — so `plan` must not be dropped or
    /// mutated in ways that re-allocate its `Vec<u64>`s while any graph
    /// launch is in flight on the stream.
    pub unsafe fn patch_and_launch(
        &mut self,
        plan: &DecodePlan,
        dev: &super::device::HipDevice,
    ) -> crate::Result<()> {
        use hipdarc::error::DriverError;
        for &op_i in &self.dynamic_ops {
            let op = &plan.ops[op_i];
            // Refresh the pointer storage — arg_values backing Vec should
            // not have moved (we build the plan once), but sanity-update
            // in case anything re-allocated.
            let storage = &mut self.arg_ptr_storage[op_i];
            if storage.len() != op.arg_values.len() {
                storage.clear();
                storage.extend(op.arg_values.iter().map(|v| v as *const u64 as *mut c_void));
            } else {
                for (slot, v) in storage.iter_mut().zip(op.arg_values.iter()) {
                    *slot = v as *const u64 as *mut c_void;
                }
            }
            let node = self.nodes[op_i];
            self.exec
                .set_kernel_node_params(
                    node,
                    op.func,
                    op.grid,
                    op.block,
                    op.shared_mem,
                    storage.as_mut_ptr(),
                )
                .map_err(|e: DriverError| crate::Error::wrap(e))?;
        }
        self.exec
            .launch(dev.stream())
            .map_err(|e| crate::Error::wrap(e))?;
        Ok(())
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn dynamic_node_count(&self) -> usize {
        self.dynamic_ops.len()
    }
}
