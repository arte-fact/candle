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
use hipdarc::driver::{HipSlice, HipStream};
use std::sync::Arc;

/// A cluster of HIP devices that can exchange data via peer-copies.
#[derive(Debug)]
pub struct HipCluster {
    /// Devices, indexed by rank.
    devices: Vec<HipDevice>,
    /// One receive stream per (dst_rank): all peer copies TO dst_rank
    /// issue on `peer_streams[dst_rank]`. Copies with different sources
    /// serialise per destination but run concurrently across destinations.
    peer_streams: Vec<Arc<HipStream>>,
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

        Ok(Self { devices, peer_streams })
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
    /// on `src_rank`. The copy is enqueued on `peer_streams[dst_rank]`
    /// — sync that stream (or call `sync_all`) before reading `dst`.
    ///
    /// Sets the current device to `dst_rank` as a side effect. HIP's
    /// stream APIs return `hipErrorNotReady` on sync when the calling
    /// thread's current device doesn't match the stream's device, so
    /// every entry point into this cluster that touches a stream must
    /// land the thread on the right device first.
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
        let dst_raw = hipdarc::driver::HipDevice::new(self.devices[dst_rank].ordinal())
            .map_err(crate::Error::wrap)?;
        dst_raw.set_current().map_err(crate::Error::wrap)?;

        if dst_rank == src_rank {
            return self.peer_streams[dst_rank]
                .memcpy_dtod(dst, src)
                .map_err(crate::Error::wrap);
        }

        let src_raw = hipdarc::driver::HipDevice::new(self.devices[src_rank].ordinal())
            .map_err(crate::Error::wrap)?;
        self.peer_streams[dst_rank]
            .memcpy_peer_async(dst, src, &src_raw)
            .map_err(crate::Error::wrap)?;
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
        HipStream::memcpy_peer_blocking(dst, &dst_raw, src, &src_raw)
            .map_err(crate::Error::wrap)?;
        Ok(())
    }
}
