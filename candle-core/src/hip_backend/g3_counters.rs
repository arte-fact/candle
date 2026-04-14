//! Phase T2 — device-resident counter buffer for G3 captured plans.
//!
//! Background: under G3 (HIP graph capture) we used to patch ~326 kernel-
//! node params per replay via hipGraphExecKernelNodeSetParams. Profiling
//! showed this costs 1.1 ms/replay = ~6.5 % of E4B Q4_0 G3 decode wall-
//! clock. AMD performance docs ackowledge this is "excessive overhead"
//! when the same small subset of nodes is updated each launch.
//!
//! The fix: take counter args (L_k_iter, index_pos, etc.) by POINTER
//! rather than by value. The pointer is captured ONCE at recording time
//! (stable device address, baked into the graph node). Per replay we
//! update the underlying value via a single tiny hipMemcpyHtoDAsync.
//!
//! This module owns the device-side counter buffer. Slots are pre-
//! assigned at compile time (one per logical counter type). Any thread/
//! caller can read `slot_ptr(SLOT)` to get the device pointer to use as
//! a kernel arg, and `set(SLOT, value)` to update the value before launch.
//!
//! Lifetime: one counter buffer per HipDevice per process (lazy on first
//! access). Pointer stable for the lifetime of the device.

use crate::hip_backend::hipdarc;
use crate::hip_backend::HipDevice;
use crate::hip_backend::WrapErr;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Counter slot indices. Each unique scalar counter that flows through
/// G3-captured kernels gets its own slot. Add new entries here when
/// expanding T2 to more kernels.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CounterSlot {
    /// `L_k_iter` for `gqa_decode_mv_fast_d{256,512}_f32_ctr`. The
    /// effective number of K positions to iterate (= index_pos + 1
    /// during decode, may be larger when n_kv padding is in effect).
    LkIter = 0,
    /// `offset` (RoPE base position) for `rope_f32_ctr`. Same value as
    /// L_k_iter - 1 in the typical decode case but kept separate so
    /// callers don't have to assume the relationship.
    RopeOffset = 1,
}

/// Number of slots; sized for 16 future counters.
const COUNTER_SLOTS: usize = 16;

/// One counter buffer per device.
struct CounterBuf {
    /// Device buffer of `COUNTER_SLOTS` u32 values. Stable pointer.
    device: hipdarc::driver::HipSlice<u32>,
    /// Host-side mirror; updated before each hipMemcpyHtoDAsync.
    host: [u32; COUNTER_SLOTS],
}

/// Per-device singleton. Indexed by HipDevice id.
static REGISTRY: OnceLock<Mutex<HashMap<usize, Box<CounterBuf>>>> = OnceLock::new();

fn registry() -> &'static Mutex<HashMap<usize, Box<CounterBuf>>> {
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn ensure_buf(dev: &HipDevice) -> *mut u32 {
    let key = dev.ordinal();
    let mut reg = registry().lock().unwrap();
    if !reg.contains_key(&key) {
        // Allocate a zero-initialised buffer.
        let device = dev
            .alloc_zeros::<u32>(COUNTER_SLOTS)
            .expect("g3_counters: alloc_zeros failed");
        reg.insert(
            key,
            Box::new(CounterBuf {
                device,
                host: [0u32; COUNTER_SLOTS],
            }),
        );
    }
    let buf = reg.get(&key).unwrap();
    buf.device.device_ptr() as *mut u32
}

/// Get the device pointer for the given slot. Stable for the lifetime
/// of the device. Pass this as a kernel arg to a `_ctr` variant.
pub fn slot_ptr(dev: &HipDevice, slot: CounterSlot) -> *const u32 {
    let base = ensure_buf(dev);
    unsafe { base.add(slot as usize) }
}

/// Update `slot`'s value and push the host mirror to the device. The
/// update is async on the device's primary stream — kernels launched
/// after this call (on the same stream) will see the new value.
pub fn set(dev: &HipDevice, slot: CounterSlot, value: u32) -> crate::Result<()> {
    let key = dev.ordinal();
    ensure_buf(dev);
    let mut reg = registry().lock().unwrap();
    let buf = reg.get_mut(&key).unwrap();
    if buf.host[slot as usize] == value {
        // No-op: slot already at this value.
        return Ok(());
    }
    buf.host[slot as usize] = value;
    // Push the FULL host mirror in one memcpy. 16 × 4 bytes = 64 bytes —
    // smaller than a single set_kernel_node_params call's overhead.
    let dst = buf.device.device_ptr() as *mut u32;
    let src = buf.host.as_ptr();
    let bytes = COUNTER_SLOTS * std::mem::size_of::<u32>();
    let stream = dev.stream().raw();
    unsafe {
        let rc = hipdarc::sys::hipMemcpyAsync(
            dst as *mut _,
            src as *const _,
            bytes,
            hipdarc::sys::hipMemcpyKind::hipMemcpyHostToDevice,
            stream,
        );
        if rc != hipdarc::sys::hipError_t::hipSuccess {
            crate::bail!("g3_counters::set: hipMemcpyAsync failed with {:?}", rc);
        }
    }
    Ok(())
}
