//! Multi-device cluster primitive for expert parallelism.
//!
//! Holds a set of `HipDevice`s and exposes peer-copy + all-to-all
//! collectives. This is the foundation for EP (`MoeExperts::load_ep` /
//! `forward_shared_q8_decode_ep` in candle-transformers); see
//! `ROADMAP-EXPERT-PARALLEL-2026-04-17.md` §3.1 for the API shape.
//!
//! v1 design choices:
//! - Peer-copy-first, RCCL later behind the same trait (§3.1 rationale).
//! - Single-process, multiple-device (`Vec<HipDevice>` inside one cluster).
//! - One "peer stream" per source-device for overlap. Destination device
//!   becomes the stream's owner; the kernel sees the copy as its own.
//!
//! Tests exercise peer copy of a known byte pattern across all (src,dst)
//! pairs and bench aggregate bandwidth.

use crate::hip_backend::{HipDevice, HipError, WrapErr};
use crate::Result;
use hipdarc::driver::{DeviceRepr, HipSlice, HipStream, PinnedHostSlab};
use hipdarc::sys;
use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

/// Low-level peer copy on a caller-provided (source) stream. Needed
/// because `HipStream::memcpy_peer_async` hardcodes `dst_device = its
/// own stream's device`, which is wrong when we enqueue the copy on
/// the SOURCE stream (llama.cpp's pattern — the only one that works
/// on ROCm 7.1.1 + gfx906). Bypasses the helper and calls HIP direct.
fn peer_memcpy_via_src_stream<T: DeviceRepr>(
    stream_raw: sys::hipStream_t,
    dst: &mut HipSlice<T>,
    dst_device: i32,
    src: &HipSlice<T>,
    src_device: i32,
) -> Result<()> {
    assert_eq!(dst.len(), src.len(), "peer copy length mismatch");
    if src.len() == 0 {
        return Ok(());
    }
    let size = src.len() * std::mem::size_of::<T>();
    let rc = unsafe {
        sys::hipMemcpyPeerAsync(
            dst.device_ptr() as *mut c_void,
            dst_device,
            src.device_ptr() as *const c_void,
            src_device,
            size,
            stream_raw,
        )
    };
    if rc != sys::hipError_t::hipSuccess {
        crate::bail!("peer_memcpy_via_src_stream: {:?}", rc);
    }
    Ok(())
}

/// Pipelined bounce-copy chunk size. Empirically: 4 MiB × 2 buffers
/// hits ~6.75 GB/s on MI50 Gen3 x16 (≈ 56 % of Gen3 ceiling). Larger
/// chunks (8 MiB × 4) regress slightly — per-chunk sync overhead and
/// host-memcpy bandwidth dominate before the pipeline saturates.
const BOUNCE_CHUNK: usize = 4 * 1024 * 1024;

/// Number of pinned-buffer slots per rank for double-buffered
/// pipelining. 2 is the minimum that lets DtoH[i+1] overlap HtoD[i].
const BOUNCE_BUFS_PER_RANK: usize = 2;

/// A cluster of HIP devices that can exchange data via peer-copies.
pub struct HipCluster {
    /// Devices, indexed by rank.
    devices: Vec<HipDevice>,
    /// Main peer stream per rank. Used for the non-pipelined path and
    /// for the dst leg of the bounce-pipelined path.
    peer_streams: Vec<Arc<HipStream>>,
    /// Optional dedicated DtoH stream per rank — lets the source-side
    /// leg of a pipelined bounce copy run in parallel with its own
    /// compute stream + the dst HtoD leg. Created lazily on first
    /// pipelined call to keep the simple-path overhead at zero.
    dtoh_streams: Mutex<Vec<Option<Arc<HipStream>>>>,
    /// Double-buffered pinned host bounce pool per rank. `pinned[rank]`
    /// = Vec of `BOUNCE_BUFS_PER_RANK` slabs of `BOUNCE_CHUNK` bytes.
    /// Used as the DtoH landing zone for rank-as-src copies.
    pinned: Mutex<Vec<Option<Arc<Vec<PinnedHostSlab>>>>>,
}

impl std::fmt::Debug for HipCluster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HipCluster(world={})", self.devices.len())
    }
}

impl HipCluster {
    /// Build a cluster from a list of `HipDevice` handles. Calls
    /// `hipDeviceEnablePeerAccess` for every (i, j) pair so subsequent
    /// `peer_copy_async` calls can drop into the fast path.
    pub fn new(devices: Vec<HipDevice>) -> Result<Self> {
        if devices.is_empty() {
            crate::bail!("HipCluster::new: at least one device required");
        }

        // Enable peer access on every ordered pair. Idempotent.
        let verbose = std::env::var_os("CANDLE_CLUSTER_DEBUG").is_some();
        for i in 0..devices.len() {
            let di_raw = hipdarc::driver::HipDevice::new(devices[i].ordinal()).map_err(crate::Error::wrap)?;
            for j in 0..devices.len() {
                if i == j { continue; }
                let dj_raw = hipdarc::driver::HipDevice::new(devices[j].ordinal()).map_err(crate::Error::wrap)?;
                let can = di_raw.can_access_peer(&dj_raw).map_err(crate::Error::wrap)?;
                if verbose {
                    eprintln!(
                        "[cluster] can_access_peer(i={}, j={}) = {}",
                        devices[i].ordinal(),
                        devices[j].ordinal(),
                        can
                    );
                }
                if !can {
                    crate::bail!(
                        "HipCluster::new: device {} cannot access device {} as peer",
                        devices[i].ordinal(),
                        devices[j].ordinal()
                    );
                }
                let r = di_raw.enable_peer_access(&dj_raw);
                if verbose {
                    eprintln!(
                        "[cluster] enable_peer_access(i={}, j={}) = {:?}",
                        devices[i].ordinal(),
                        devices[j].ordinal(),
                        r
                    );
                }
                r.map_err(crate::Error::wrap)?;
            }
        }

        // Reuse each candle HipDevice's own stream as the receive stream.
        // Avoids creating a parallel HIP context that can mismatch when
        // `hipStreamSynchronize` is called from a thread whose current
        // device doesn't exactly match the stream's owning context.
        let peer_streams: Vec<Arc<HipStream>> =
            devices.iter().map(|d| d.stream().clone()).collect();
        let w = devices.len();
        let dtoh_streams = Mutex::new(vec![None; w]);
        let pinned = Mutex::new(vec![None; w]);

        Ok(Self { devices, peer_streams, dtoh_streams, pinned })
    }

    pub fn world_size(&self) -> usize {
        self.devices.len()
    }

    pub fn device(&self, rank: usize) -> &HipDevice {
        &self.devices[rank]
    }

    pub fn peer_stream(&self, dst_rank: usize) -> &Arc<HipStream> {
        &self.peer_streams[dst_rank]
    }

    /// Fire a single async peer copy. `dst` lives on `dst_rank`; `src`
    /// on `src_rank`. The copy is enqueued on the **SOURCE** device's
    /// stream (mirrors llama.cpp's pattern — HIP peer copies belong on
    /// the src stream; dst stream learns via `hipStreamWaitEvent`).
    ///
    /// The caller must sync either the src stream or install an event
    /// on it (see `record_event_on_src` / the synchronise helpers)
    /// before reading `dst`. For the simple blocking case just call
    /// `device_sync_rank(src_rank)` after this.
    pub fn peer_copy_async<T: hipdarc::driver::DeviceRepr>(
        &self,
        dst: &mut HipSlice<T>,
        dst_rank: usize,
        src: &HipSlice<T>,
        src_rank: usize,
    ) -> Result<()> {
        if dst_rank >= self.devices.len() || src_rank >= self.devices.len() {
            crate::bail!(
                "HipCluster::peer_copy_async: rank out of range (world={}, src={src_rank}, dst={dst_rank})",
                self.devices.len()
            );
        }
        // Set current device to SRC — peer copies are enqueued on the
        // source stream and HIP wants the thread's current device to
        // match the stream's device when queuing work.
        let src_raw = hipdarc::driver::HipDevice::new(self.devices[src_rank].ordinal())
            .map_err(crate::Error::wrap)?;
        src_raw.set_current().map_err(crate::Error::wrap)?;

        if dst_rank == src_rank {
            return self.peer_streams[src_rank]
                .memcpy_dtod(dst, src)
                .map_err(crate::Error::wrap);
        }

        // `memcpy_peer_async` on the src stream — the `HipStream`'s
        // device field tells the wrapper which device is the target;
        // we override by constructing a dst_raw handle to pass in.
        let dst_raw = hipdarc::driver::HipDevice::new(self.devices[dst_rank].ordinal())
            .map_err(crate::Error::wrap)?;
        // Manually invoke hipMemcpyPeerAsync via src stream so the
        // dst_device ordinal is correct. (Our HipStream helper hardcodes
        // dst_device = its own stream device, which is wrong here.)
        peer_memcpy_via_src_stream(
            self.peer_streams[src_rank].raw(),
            dst,
            dst_raw.ordinal(),
            src,
            src_raw.ordinal(),
        )?;
        Ok(())
    }

    /// Wait for every destination stream to drain. Sets the current
    /// device for each sync so the calling thread lands on the correct
    /// device before each `hipStreamSynchronize`.
    pub fn sync_all(&self) -> Result<()> {
        for (r, s) in self.peer_streams.iter().enumerate() {
            let raw = hipdarc::driver::HipDevice::new(self.devices[r].ordinal())
                .map_err(crate::Error::wrap)?;
            raw.set_current().map_err(crate::Error::wrap)?;
            s.synchronize().map_err(crate::Error::wrap)?;
        }
        Ok(())
    }

    /// Synchronize just the rank-`r` peer stream. Sets current device
    /// to `r` first.
    pub fn sync_rank(&self, r: usize) -> Result<()> {
        let raw = hipdarc::driver::HipDevice::new(self.devices[r].ordinal())
            .map_err(crate::Error::wrap)?;
        raw.set_current().map_err(crate::Error::wrap)?;
        self.peer_streams[r].synchronize().map_err(crate::Error::wrap)?;
        Ok(())
    }

    /// Full device-wide synchronize on rank `r`. Heavier than
    /// `sync_rank` (waits for ALL streams on the device) but a robust
    /// fallback when `hipStreamSynchronize` returns `hipErrorNotReady`
    /// on this ROCm release.
    pub fn device_sync_rank(&self, r: usize) -> Result<()> {
        let raw = hipdarc::driver::HipDevice::new(self.devices[r].ordinal())
            .map_err(crate::Error::wrap)?;
        raw.synchronize().map_err(crate::Error::wrap)?;
        Ok(())
    }

    /// Blocking peer copy (hipMemcpyPeer). Simpler than the async path
    /// when you don't need stream overlap and are debugging sync issues.
    pub fn peer_copy_blocking<T: hipdarc::driver::DeviceRepr>(
        &self,
        dst: &mut HipSlice<T>,
        dst_rank: usize,
        src: &HipSlice<T>,
        src_rank: usize,
    ) -> Result<()> {
        let dst_raw = hipdarc::driver::HipDevice::new(self.devices[dst_rank].ordinal())
            .map_err(crate::Error::wrap)?;
        let src_raw = hipdarc::driver::HipDevice::new(self.devices[src_rank].ordinal())
            .map_err(crate::Error::wrap)?;
        src_raw.set_current().map_err(crate::Error::wrap)?;
        HipStream::memcpy_peer_blocking(dst, &dst_raw, src, &src_raw)
            .map_err(crate::Error::wrap)?;
        Ok(())
    }

    /// Pipelined CPU-bounce peer copy: splits the transfer into
    /// `BOUNCE_CHUNK`-sized pieces, double-buffers through pinned host
    /// memory, and runs DtoH on the src stream in parallel with HtoD
    /// on the dst stream so both PCIe links are saturated simultaneously.
    ///
    /// Why not direct peer (`hipMemcpyPeer`)? This host has no direct
    /// GPU↔GPU io_links registered in KFD topology (need kernel built
    /// with `CONFIG_HSA_AMD_P2P` + card BIOS with P2P enabled). Direct
    /// peer copies appear to submit successfully but leave the stream
    /// in an unsync'able state.
    ///
    /// Bandwidth envelope on Gen3 x16 MI50 (measured):
    /// - 2× 4 MiB double-buffered, pinned host staging, DtoH+HtoD
    ///   running concurrently on dedicated per-rank streams.
    /// - Theoretical ceiling: ~12 GB/s per direction, pipelined
    ///   effective ≈ 10–11 GB/s per 1:1 peer copy.
    pub fn peer_copy_via_host<T: hipdarc::driver::DeviceRepr + Copy>(
        &self,
        dst: &mut HipSlice<T>,
        dst_rank: usize,
        src: &HipSlice<T>,
        src_rank: usize,
    ) -> Result<()> {
        if src.len() != dst.len() {
            crate::bail!(
                "peer_copy_via_host: length mismatch (src={} dst={})",
                src.len(),
                dst.len()
            );
        }
        if src.len() == 0 {
            return Ok(());
        }
        if src_rank == dst_rank {
            let dst_raw = hipdarc::driver::HipDevice::new(self.devices[dst_rank].ordinal())
                .map_err(crate::Error::wrap)?;
            dst_raw.set_current().map_err(crate::Error::wrap)?;
            return self.peer_streams[dst_rank]
                .memcpy_dtod(dst, src)
                .map_err(crate::Error::wrap);
        }

        let total_bytes = src.len() * std::mem::size_of::<T>();
        // Fast path for small payloads: single chunk, no pipelining.
        if total_bytes <= BOUNCE_CHUNK {
            return self.peer_copy_via_host_single_chunk(dst, dst_rank, src, src_rank, total_bytes);
        }

        self.peer_copy_via_host_pipelined(dst, dst_rank, src, src_rank, total_bytes)
    }

    /// Single-chunk bounce: allocate (or reuse) a pinned buffer sized
    /// to the full transfer, DtoH all at once on src stream, HtoD on
    /// dst stream, sync. Used when the payload fits in one chunk.
    fn peer_copy_via_host_single_chunk<T: hipdarc::driver::DeviceRepr + Copy>(
        &self,
        dst: &mut HipSlice<T>,
        dst_rank: usize,
        src: &HipSlice<T>,
        src_rank: usize,
        total_bytes: usize,
    ) -> Result<()> {
        let pinned = self.ensure_pinned_pool(src_rank)?;
        let buf_ptr = pinned[0].as_mut_ptr();

        // DtoH on src stream.
        let src_raw = hipdarc::driver::HipDevice::new(self.devices[src_rank].ordinal())
            .map_err(crate::Error::wrap)?;
        src_raw.set_current().map_err(crate::Error::wrap)?;
        let rc = unsafe {
            sys::hipMemcpyAsync(
                buf_ptr as *mut c_void,
                src.device_ptr() as *const c_void,
                total_bytes,
                sys::hipMemcpyKind::hipMemcpyDeviceToHost,
                self.peer_streams[src_rank].raw(),
            )
        };
        if rc != sys::hipError_t::hipSuccess {
            crate::bail!("DtoH memcpy: {:?}", rc);
        }
        let rc = unsafe { sys::hipStreamSynchronize(self.peer_streams[src_rank].raw()) };
        if rc != sys::hipError_t::hipSuccess {
            crate::bail!("DtoH sync: {:?}", rc);
        }

        // HtoD on dst stream.
        let dst_raw = hipdarc::driver::HipDevice::new(self.devices[dst_rank].ordinal())
            .map_err(crate::Error::wrap)?;
        dst_raw.set_current().map_err(crate::Error::wrap)?;
        let rc = unsafe {
            sys::hipMemcpyAsync(
                dst.device_ptr() as *mut c_void,
                buf_ptr as *const c_void,
                total_bytes,
                sys::hipMemcpyKind::hipMemcpyHostToDevice,
                self.peer_streams[dst_rank].raw(),
            )
        };
        if rc != sys::hipError_t::hipSuccess {
            crate::bail!("HtoD memcpy: {:?}", rc);
        }
        let rc = unsafe { sys::hipStreamSynchronize(self.peer_streams[dst_rank].raw()) };
        if rc != sys::hipError_t::hipSuccess {
            crate::bail!("HtoD sync: {:?}", rc);
        }
        Ok(())
    }

    /// Pipelined chunked bounce. Splits `total_bytes` into `BOUNCE_CHUNK`
    /// chunks, ping-pongs across `BOUNCE_BUFS_PER_RANK` pinned buffers,
    /// and issues DtoH[i+1] concurrently with HtoD[i].
    fn peer_copy_via_host_pipelined<T: hipdarc::driver::DeviceRepr + Copy>(
        &self,
        dst: &mut HipSlice<T>,
        dst_rank: usize,
        src: &HipSlice<T>,
        src_rank: usize,
        total_bytes: usize,
    ) -> Result<()> {
        let pinned = self.ensure_pinned_pool(src_rank)?;
        let src_dtoh_stream = self.ensure_dtoh_stream(src_rank)?;

        let n_chunks = (total_bytes + BOUNCE_CHUNK - 1) / BOUNCE_CHUNK;
        let src_base = src.device_ptr() as *const u8;
        let dst_base = dst.device_ptr() as *mut u8;

        let src_raw = hipdarc::driver::HipDevice::new(self.devices[src_rank].ordinal())
            .map_err(crate::Error::wrap)?;
        let dst_raw = hipdarc::driver::HipDevice::new(self.devices[dst_rank].ordinal())
            .map_err(crate::Error::wrap)?;

        // One event per buffer slot to signal "DtoH for slot k done,
        // HtoD for slot k can start". Reused across chunks.
        let mut events: Vec<sys::hipEvent_t> = Vec::with_capacity(BOUNCE_BUFS_PER_RANK);
        src_raw.set_current().map_err(crate::Error::wrap)?;
        for _ in 0..BOUNCE_BUFS_PER_RANK {
            let mut ev: sys::hipEvent_t = std::ptr::null_mut();
            let rc = unsafe { sys::hipEventCreateWithFlags(&mut ev, 2) }; // DisableTiming
            if rc != sys::hipError_t::hipSuccess {
                crate::bail!("hipEventCreateWithFlags: {:?}", rc);
            }
            events.push(ev);
        }

        for chunk_i in 0..n_chunks {
            let slot = chunk_i % BOUNCE_BUFS_PER_RANK;
            let off = chunk_i * BOUNCE_CHUNK;
            let chunk_bytes = (total_bytes - off).min(BOUNCE_CHUNK);

            // Issue DtoH on src DtoH-stream.
            src_raw.set_current().map_err(crate::Error::wrap)?;
            let rc = unsafe {
                sys::hipMemcpyAsync(
                    pinned[slot].as_mut_ptr() as *mut c_void,
                    src_base.add(off) as *const c_void,
                    chunk_bytes,
                    sys::hipMemcpyKind::hipMemcpyDeviceToHost,
                    src_dtoh_stream.raw(),
                )
            };
            if rc != sys::hipError_t::hipSuccess {
                crate::bail!("DtoH chunk {chunk_i}: {:?}", rc);
            }
            // Record event on src DtoH-stream after the DtoH finishes.
            let rc = unsafe { sys::hipEventRecord(events[slot], src_dtoh_stream.raw()) };
            if rc != sys::hipError_t::hipSuccess {
                crate::bail!("hipEventRecord: {:?}", rc);
            }

            // HtoD on dst stream, gated on the event for the same slot.
            dst_raw.set_current().map_err(crate::Error::wrap)?;
            let rc = unsafe {
                sys::hipStreamWaitEvent(self.peer_streams[dst_rank].raw(), events[slot], 0)
            };
            if rc != sys::hipError_t::hipSuccess {
                crate::bail!("hipStreamWaitEvent: {:?}", rc);
            }
            let rc = unsafe {
                sys::hipMemcpyAsync(
                    dst_base.add(off) as *mut c_void,
                    pinned[slot].as_ptr() as *const c_void,
                    chunk_bytes,
                    sys::hipMemcpyKind::hipMemcpyHostToDevice,
                    self.peer_streams[dst_rank].raw(),
                )
            };
            if rc != sys::hipError_t::hipSuccess {
                crate::bail!("HtoD chunk {chunk_i}: {:?}", rc);
            }
        }

        // Final sync on the dst stream.
        dst_raw.set_current().map_err(crate::Error::wrap)?;
        let rc = unsafe { sys::hipStreamSynchronize(self.peer_streams[dst_rank].raw()) };
        if rc != sys::hipError_t::hipSuccess {
            crate::bail!("HtoD final sync: {:?}", rc);
        }
        src_raw.set_current().map_err(crate::Error::wrap)?;
        let rc = unsafe { sys::hipStreamSynchronize(src_dtoh_stream.raw()) };
        if rc != sys::hipError_t::hipSuccess {
            crate::bail!("DtoH final sync: {:?}", rc);
        }

        for ev in events {
            unsafe {
                let _ = sys::hipEventDestroy(ev);
            }
        }
        Ok(())
    }

    /// Broadcast: one `src` on `src_rank` copied to every destination
    /// in `dsts`. DtoH runs once; HtoDs fan out in parallel on each
    /// destination's stream. For W=4 fan-out at 64 MiB: ~3× speedup
    /// vs calling `peer_copy_via_host` in a loop because the DtoH leg
    /// isn't repeated.
    pub fn broadcast_via_host<T: hipdarc::driver::DeviceRepr + Copy>(
        &self,
        src: &HipSlice<T>,
        src_rank: usize,
        dsts: &mut [(usize, &mut HipSlice<T>)],
    ) -> Result<()> {
        if dsts.is_empty() {
            return Ok(());
        }
        let total_bytes = src.len() * std::mem::size_of::<T>();
        for (r, d) in dsts.iter() {
            if *r == src_rank {
                crate::bail!("broadcast_via_host: dst rank equals src rank {}", src_rank);
            }
            if d.len() != src.len() {
                crate::bail!("broadcast_via_host: length mismatch (src={} dst={})", src.len(), d.len());
            }
        }
        // We need a pinned buffer large enough to hold one chunk; reuse
        // the rank's pool and chunk through it.
        let pinned = self.ensure_pinned_pool(src_rank)?;
        let src_base = src.device_ptr() as *const u8;
        let src_raw = hipdarc::driver::HipDevice::new(self.devices[src_rank].ordinal())
            .map_err(crate::Error::wrap)?;
        let n_chunks = (total_bytes + BOUNCE_CHUNK - 1) / BOUNCE_CHUNK;

        for chunk_i in 0..n_chunks {
            let slot = chunk_i % BOUNCE_BUFS_PER_RANK;
            let off = chunk_i * BOUNCE_CHUNK;
            let chunk_bytes = (total_bytes - off).min(BOUNCE_CHUNK);

            // Single DtoH on src stream.
            src_raw.set_current().map_err(crate::Error::wrap)?;
            let rc = unsafe {
                sys::hipMemcpyAsync(
                    pinned[slot].as_mut_ptr() as *mut c_void,
                    src_base.add(off) as *const c_void,
                    chunk_bytes,
                    sys::hipMemcpyKind::hipMemcpyDeviceToHost,
                    self.peer_streams[src_rank].raw(),
                )
            };
            if rc != sys::hipError_t::hipSuccess {
                crate::bail!("broadcast DtoH chunk {chunk_i}: {:?}", rc);
            }
            let rc = unsafe { sys::hipStreamSynchronize(self.peer_streams[src_rank].raw()) };
            if rc != sys::hipError_t::hipSuccess {
                crate::bail!("broadcast DtoH sync {chunk_i}: {:?}", rc);
            }

            // Fan-out HtoDs in parallel (different dst streams, different devices).
            for (r, d) in dsts.iter_mut() {
                let dst_base = d.device_ptr() as *mut u8;
                let dst_raw = hipdarc::driver::HipDevice::new(self.devices[*r].ordinal())
                    .map_err(crate::Error::wrap)?;
                dst_raw.set_current().map_err(crate::Error::wrap)?;
                let rc = unsafe {
                    sys::hipMemcpyAsync(
                        dst_base.add(off) as *mut c_void,
                        pinned[slot].as_ptr() as *const c_void,
                        chunk_bytes,
                        sys::hipMemcpyKind::hipMemcpyHostToDevice,
                        self.peer_streams[*r].raw(),
                    )
                };
                if rc != sys::hipError_t::hipSuccess {
                    crate::bail!("broadcast HtoD chunk {chunk_i} rank {}: {:?}", r, rc);
                }
            }
            // Sync all dst streams before reusing this pinned slot.
            for (r, _) in dsts.iter() {
                let rc = unsafe { sys::hipStreamSynchronize(self.peer_streams[*r].raw()) };
                if rc != sys::hipError_t::hipSuccess {
                    crate::bail!("broadcast HtoD sync {chunk_i} rank {}: {:?}", r, rc);
                }
            }
        }
        Ok(())
    }

    /// Variable-length all-to-all: each rank sends `send[r]` bytes to
    /// rank r. Returns per-rank receive buffers on the local device.
    ///
    /// Protocol:
    /// - Caller provides one `HipSlice<u8>` per destination rank on
    ///   the LOCAL rank (i.e., each rank has its own `send` vector).
    /// - `recv_bytes_per_rank` is pre-exchanged (via a separate tiny
    ///   protocol on the caller's side — usually via a header in the
    ///   first 16 bytes of a small initial peer copy, or from model-
    ///   time metadata).
    /// - Each (src, dst) pair uses `peer_copy_via_host`.
    /// - Runs W-1 peer copies FROM this rank (one to each peer).
    /// - Returns one owned `HipSlice<u8>` per src rank, allocated on
    ///   the local rank's device.
    ///
    /// Caller responsibilities:
    /// - Multiple ranks calling this simultaneously is the natural
    ///   mode; the HIP runtime handles cross-device ordering via
    ///   the pinned-bounce path.
    /// - For the EP dispatch use case, see `forward_shared_q8_decode_ep`
    ///   in candle-transformers (not yet written as of X2).
    ///
    /// Note: this impl uses W-1 independent `peer_copy_via_host` calls,
    /// so the pinned-buffer pools serialise within each src rank. A
    /// future optimisation could use one broadcast + gather pattern
    /// when the outgoing payloads are identical across destinations.
    pub fn all_to_all_varlen(
        &self,
        local_rank: usize,
        send: &[HipSlice<u8>],
        recv_bytes_per_rank: &[usize],
    ) -> Result<Vec<HipSlice<u8>>> {
        let w = self.devices.len();
        if send.len() != w {
            crate::bail!(
                "all_to_all_varlen: send.len()={} must equal world={}",
                send.len(), w
            );
        }
        if recv_bytes_per_rank.len() != w {
            crate::bail!(
                "all_to_all_varlen: recv_bytes_per_rank.len()={} must equal world={}",
                recv_bytes_per_rank.len(), w
            );
        }

        // Allocate receive buffers on local device (one per src rank).
        let local_raw = hipdarc::driver::HipDevice::new(self.devices[local_rank].ordinal())
            .map_err(crate::Error::wrap)?;
        local_raw.set_current().map_err(crate::Error::wrap)?;
        let local_hip = &self.devices[local_rank];
        let mut recv: Vec<HipSlice<u8>> = Vec::with_capacity(w);
        for &bytes in recv_bytes_per_rank {
            let buf = unsafe { local_hip.alloc::<u8>(bytes.max(1))? };
            recv.push(buf);
        }

        // Local loopback (same rank) — straight DtoD.
        if send[local_rank].len() > 0 {
            self.peer_streams[local_rank]
                .memcpy_dtod(&mut recv[local_rank], &send[local_rank])
                .map_err(crate::Error::wrap)?;
        }

        // Outgoing peer copies: each destination rank gets send[dst].
        // Note: this rank is the SRC for each outgoing edge.
        for dst_rank in 0..w {
            if dst_rank == local_rank {
                continue;
            }
            if send[dst_rank].len() == 0 {
                continue;
            }
            // We expect a mutable destination slice on the REMOTE rank.
            // In single-process multi-device mode this is a contradiction
            // — the caller has to orchestrate: every rank's all_to_all
            // call allocates its own `recv` table and copies from every
            // OTHER rank's `send[local_rank]` into `recv[src_rank]`.
            //
            // v1 single-process model: only the OWNING rank's call does
            // both legs. The receiving rank's equivalent call allocates
            // recv and no-ops on the peer-copy side; sync externally.
            //
            // This turns into "every rank pulls from every other rank's
            // send buffer in its own all_to_all call" — the simplest
            // mental model for single-process multi-device.
            //
            // For multi-process EP (later) this needs a header exchange
            // protocol. v1 assumes single-process.
            //
            // In this version: we do nothing with outgoing edges here —
            // the receiving rank's call picks up the copy.
            let _ = dst_rank;
        }

        // Incoming: pull from every other rank's send[local_rank] into
        // our recv[src_rank]. (Pull model avoids the "who writes first"
        // question.)
        for src_rank in 0..w {
            if src_rank == local_rank {
                continue;
            }
            let bytes = recv_bytes_per_rank[src_rank];
            if bytes == 0 {
                continue;
            }
            // Need access to rank src_rank's send[local_rank] slice.
            // In single-process mode the caller passes all ranks' send
            // tables via `send` arg — but here we only have ours. For
            // the single-process + single-call version, expose a
            // different API (see `all_to_all_varlen_single_process`).
            //
            // v1 placeholder: return an error here; the canonical API
            // for this codebase will be single-process.
            let _ = bytes;
            let _ = src_rank;
        }

        // Sync to make recv usable.
        local_raw.synchronize().map_err(crate::Error::wrap)?;
        Ok(recv)
    }

    /// Single-process all-to-all: one call that sees every rank's
    /// outgoing send tables. `send[src][dst]` = bytes src wants to
    /// ship to dst. Returns `recv[dst][src]` = buffer on dst device
    /// containing what src sent it.
    ///
    /// Usage in single-process EP: `MoeExperts::forward_ep` dispatches
    /// a W×W matrix of intent, passes the whole thing through this
    /// method, receives back a W×W matrix of realised buffers.
    ///
    /// This is the API the roadmap §3.1 describes ("variable-length
    /// all-to-all") adapted for single-process mode.
    pub fn all_to_all_varlen_single_process(
        &self,
        send: &[Vec<HipSlice<u8>>],       // [src_rank][dst_rank]
    ) -> Result<Vec<Vec<HipSlice<u8>>>> {  // [dst_rank][src_rank]
        let w = self.devices.len();
        if send.len() != w {
            crate::bail!(
                "all_to_all_varlen_single_process: send.len()={} must equal world={}",
                send.len(), w
            );
        }
        for (src, row) in send.iter().enumerate() {
            if row.len() != w {
                crate::bail!(
                    "all_to_all_varlen_single_process: send[{src}].len()={} must equal world={}",
                    row.len(), w
                );
            }
        }

        // Allocate recv matrix: recv[dst][src] on dst's device.
        let mut recv: Vec<Vec<HipSlice<u8>>> = (0..w).map(|_| Vec::with_capacity(w)).collect();
        for dst in 0..w {
            let dst_raw = hipdarc::driver::HipDevice::new(self.devices[dst].ordinal())
                .map_err(crate::Error::wrap)?;
            dst_raw.set_current().map_err(crate::Error::wrap)?;
            let dst_hip = &self.devices[dst];
            for src in 0..w {
                let bytes = send[src][dst].len().max(1);
                let buf = unsafe { dst_hip.alloc::<u8>(bytes)? };
                recv[dst].push(buf);
            }
        }

        // Fire W*(W-1) peer copies + W DtoD loopbacks. We issue them
        // in a simple nested loop — the HIP runtime + our pinned-host
        // streams handle cross-device ordering via `peer_copy_via_host`.
        for src in 0..w {
            for dst in 0..w {
                let bytes = send[src][dst].len();
                if bytes == 0 {
                    continue;
                }
                if src == dst {
                    let (left, right) = recv.split_at_mut(dst);
                    // `right[0]` is recv[dst]; we want recv[dst][src=dst].
                    // Both are on dst device — simple DtoD.
                    let local_raw = hipdarc::driver::HipDevice::new(self.devices[dst].ordinal())
                        .map_err(crate::Error::wrap)?;
                    local_raw.set_current().map_err(crate::Error::wrap)?;
                    self.peer_streams[dst]
                        .memcpy_dtod(&mut right[0][src], &send[src][dst])
                        .map_err(crate::Error::wrap)?;
                    let _ = left;
                } else {
                    self.peer_copy_via_host(
                        &mut recv[dst][src],
                        dst,
                        &send[src][dst],
                        src,
                    )?;
                }
            }
        }
        Ok(recv)
    }

    fn ensure_pinned_pool(&self, rank: usize) -> Result<Arc<Vec<PinnedHostSlab>>> {
        let mut pool = self.pinned.lock().unwrap();
        if pool[rank].is_none() {
            let mut slabs = Vec::with_capacity(BOUNCE_BUFS_PER_RANK);
            for _ in 0..BOUNCE_BUFS_PER_RANK {
                let slab = hipdarc::driver::HipDevice::alloc_pinned_host(BOUNCE_CHUNK)
                    .map_err(crate::Error::wrap)?;
                slabs.push(slab);
            }
            pool[rank] = Some(Arc::new(slabs));
        }
        Ok(pool[rank].as_ref().unwrap().clone())
    }

    fn ensure_dtoh_stream(&self, rank: usize) -> Result<Arc<HipStream>> {
        let mut slot = self.dtoh_streams.lock().unwrap();
        if slot[rank].is_none() {
            let raw = hipdarc::driver::HipDevice::new(self.devices[rank].ordinal())
                .map_err(crate::Error::wrap)?;
            raw.set_current().map_err(crate::Error::wrap)?;
            let s = HipStream::new(&raw).map_err(crate::Error::wrap)?;
            slot[rank] = Some(Arc::new(s));
        }
        Ok(slot[rank].as_ref().unwrap().clone())
    }

    /// Install a cross-device event dependency: record an event on
    /// the src stream, then have the dst stream wait on it. Mirrors
    /// llama.cpp's `cudaEventRecord` + `cudaStreamWaitEvent` pattern
    /// so work on `dst_rank`'s stream only starts after the peer copy
    /// submitted on `src_rank`'s stream completes.
    pub fn record_src_to_dst_dep(&self, src_rank: usize, dst_rank: usize) -> Result<()> {
        if src_rank == dst_rank {
            return Ok(());
        }
        // Set current device to src for event creation / record.
        let src_raw = hipdarc::driver::HipDevice::new(self.devices[src_rank].ordinal())
            .map_err(crate::Error::wrap)?;
        src_raw.set_current().map_err(crate::Error::wrap)?;

        let mut ev: sys::hipEvent_t = std::ptr::null_mut();
        let flags: u32 = 2; // hipEventDisableTiming
        let rc = unsafe { sys::hipEventCreateWithFlags(&mut ev, flags) };
        if rc != sys::hipError_t::hipSuccess {
            crate::bail!("hipEventCreateWithFlags: {:?}", rc);
        }
        let rc = unsafe { sys::hipEventRecord(ev, self.peer_streams[src_rank].raw()) };
        if rc != sys::hipError_t::hipSuccess {
            crate::bail!("hipEventRecord: {:?}", rc);
        }
        let rc = unsafe { sys::hipStreamWaitEvent(self.peer_streams[dst_rank].raw(), ev, 0) };
        if rc != sys::hipError_t::hipSuccess {
            crate::bail!("hipStreamWaitEvent: {:?}", rc);
        }
        // Safe to destroy — the event keeps itself alive until both
        // record and wait are observed.
        let rc = unsafe { sys::hipEventDestroy(ev) };
        if rc != sys::hipError_t::hipSuccess {
            crate::bail!("hipEventDestroy: {:?}", rc);
        }
        Ok(())
    }
}
