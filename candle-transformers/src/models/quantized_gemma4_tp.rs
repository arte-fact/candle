//! X8.4 — Multi-threaded TP+EP dispatcher for gemma4.
//!
//! `W` long-lived worker threads each own a per-rank `ModelWeights`
//! (loaded via [`ModelWeights::from_gguf_tp_ep_rank`]) and wait on a
//! channel for forward requests. Every TP/EP sync point within the
//! per-rank forward fires via the blocks' attached `tp_ctx` /
//! `ep_partition`, so no forward-loop duplication is needed.
//!
//! Persistent workers (vs `std::thread::scope` per forward) amortise
//! thread startup cost across many tokens — at decode, per-call
//! dispatch is a channel send + receive (<10 µs) instead of thread
//! spawn + join (~hundreds of µs).
//!
//! **Correctness**: each rank sees identical full (reduced) outputs
//! after every all-reduce, so all ranks follow the same control flow
//! and produce the same logits. Rank 0's logits are returned.
//!
//! **VRAM**: per-rank weight footprint ≈ (dense_bytes + moe_bytes) / W,
//! matching vLLM TP+EP hybrid mode.

#![cfg(feature = "hip")]

use super::quantized_gemma4::ModelWeights;
use candle::hip_backend::cluster::HipCluster;
use candle::quantized::gguf_file;
use candle::{Device, HipDevice, Result, Tensor};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;

/// Commands sent to a persistent rank worker.
enum WorkerReq {
    /// Run `ModelWeights::forward(&tokens, index_pos)`. Only rank 0
    /// ships the result tensor back; other ranks send `Ok(None)`.
    Forward { tokens: Tensor, index_pos: usize },
    /// Invoke `ModelWeights::clear_kv_cache` on this rank.
    ClearKv,
    /// Leave the recv loop.
    Shutdown,
}

/// Response from a worker: either the forward's logits (rank 0 only)
/// or a simple ack.
type WorkerResp = Result<Option<Tensor>>;

struct RankWorker {
    tx: Sender<WorkerReq>,
    rx: Receiver<WorkerResp>,
    join: Option<JoinHandle<()>>,
}

pub struct ModelWeightsTp {
    cluster: Arc<HipCluster>,
    workers: Vec<RankWorker>,
    devices: Vec<Device>,
}

impl ModelWeightsTp {
    /// Load a TP+EP replica on each of `devices` and spawn a persistent
    /// worker thread per rank that owns the replica for the lifetime of
    /// this struct.
    pub fn from_gguf_tp_ep(
        path: impl AsRef<std::path::Path>,
        devices: &[Device],
    ) -> Result<Self> {
        if devices.is_empty() {
            candle::bail!("ModelWeightsTp: at least one device required");
        }
        let hip_devs: Vec<HipDevice> = devices
            .iter()
            .map(|d| match d {
                Device::Hip(h) => Ok(h.clone()),
                _ => Err(candle::Error::Msg(
                    "ModelWeightsTp: all devices must be HIP".into(),
                )),
            })
            .collect::<Result<_>>()?;
        let cluster = Arc::new(HipCluster::new(hip_devs)?);

        let w = devices.len();
        let path = path.as_ref().to_owned();

        // Spawn workers; each loads its rank's ModelWeights inside the
        // worker thread so the HIP context is pinned from the start.
        let mut workers = Vec::with_capacity(w);
        let mut load_errs: Vec<Option<candle::Error>> = (0..w).map(|_| None).collect();
        let (load_done_tx, load_done_rx) = channel::<(usize, Result<()>)>();
        let mut join_handles: Vec<JoinHandle<()>> = Vec::with_capacity(w);
        let mut tx_list: Vec<Sender<WorkerReq>> = Vec::with_capacity(w);
        let mut rx_list: Vec<Receiver<WorkerResp>> = Vec::with_capacity(w);

        for r in 0..w {
            let (tx_req, rx_req) = channel::<WorkerReq>();
            let (tx_resp, rx_resp) = channel::<WorkerResp>();
            let dev_r = devices[r].clone();
            let dev_h = match &dev_r {
                Device::Hip(h) => h.clone(),
                _ => unreachable!(),
            };
            let cluster_c = cluster.clone();
            let path_c = path.clone();
            let load_done = load_done_tx.clone();

            let join = std::thread::spawn(move || {
                // Pin HIP context for the life of the thread.
                if let Err(e) = candle::hip_backend::hipdarc::driver::HipDevice::new(
                    dev_h.ordinal(),
                )
                .and_then(|d| d.set_current())
                {
                    let _ = load_done.send((
                        r,
                        Err(candle::Error::Msg(format!("set_current: {e}"))),
                    ));
                    return;
                }

                // Load the rank's replica.
                let model = match gguf_file::Content::read_mmap(&path_c)
                    .and_then(|(ct, blob)| {
                        ModelWeights::from_gguf_tp_ep_rank(ct, blob, dev_r.clone(), cluster_c, r)
                    }) {
                    Ok(m) => m,
                    Err(e) => {
                        let _ = load_done.send((r, Err(e)));
                        return;
                    }
                };
                let _ = load_done.send((r, Ok(())));

                let mut model = model;
                loop {
                    let req = match rx_req.recv() {
                        Ok(q) => q,
                        Err(_) => break, // sender dropped
                    };
                    match req {
                        WorkerReq::Forward { tokens, index_pos } => {
                            let tokens = match tokens.to_device(&dev_r) {
                                Ok(t) => t,
                                Err(e) => {
                                    let _ = tx_resp.send(Err(e));
                                    continue;
                                }
                            };
                            let out = model.forward(&tokens, index_pos);
                            let reply = match (r, out) {
                                (0, Ok(t)) => Ok(Some(t)),
                                (_, Ok(_)) => Ok(None),
                                (_, Err(e)) => Err(e),
                            };
                            let _ = tx_resp.send(reply);
                        }
                        WorkerReq::ClearKv => {
                            model.clear_kv_cache();
                            let _ = tx_resp.send(Ok(None));
                        }
                        WorkerReq::Shutdown => break,
                    }
                }
            });
            join_handles.push(join);
            tx_list.push(tx_req);
            rx_list.push(rx_resp);
        }
        drop(load_done_tx);

        // Wait for all W ranks to finish loading (or report failure).
        for _ in 0..w {
            let (r, status) = load_done_rx
                .recv()
                .map_err(|_| candle::Error::Msg("worker load channel closed".into()))?;
            if let Err(e) = status {
                load_errs[r] = Some(e);
            }
        }
        if let Some(idx) = load_errs.iter().position(|e| e.is_some()) {
            // Send shutdown to any alive workers + return the first error.
            for tx in &tx_list {
                let _ = tx.send(WorkerReq::Shutdown);
            }
            for j in join_handles {
                let _ = j.join();
            }
            return Err(load_errs[idx].take().unwrap());
        }

        for (i, join) in join_handles.into_iter().enumerate() {
            workers.push(RankWorker {
                tx: tx_list[i].clone(),
                rx: std::mem::replace(&mut rx_list[i], channel().1),
                join: Some(join),
            });
        }
        // The clones in tx_list still hold senders; drop them so the
        // underlying channel's ref-count isn't inflated. The workers
        // hold their RankWorker's tx; that's the one that matters.
        drop(tx_list);

        Ok(Self {
            cluster,
            workers,
            devices: devices.to_vec(),
        })
    }

    pub fn world_size(&self) -> usize {
        self.workers.len()
    }

    /// Forward step. Broadcasts `tokens` + `index_pos` to each worker;
    /// each worker runs its rank's forward (with barrier all-reduces
    /// firing inside attention / MLP / MoE blocks). Returns rank 0's
    /// logits.
    pub fn forward(&self, tokens: &Tensor, index_pos: usize) -> Result<Tensor> {
        // Stage tokens on CPU once; each worker copies to its own device.
        let tokens_cpu = if matches!(tokens.device(), Device::Cpu) {
            tokens.clone()
        } else {
            tokens.to_device(&Device::Cpu)?
        };
        for w in &self.workers {
            w.tx
                .send(WorkerReq::Forward {
                    tokens: tokens_cpu.clone(),
                    index_pos,
                })
                .map_err(|_| candle::Error::Msg("worker tx closed".into()))?;
        }
        let mut rank0: Option<Tensor> = None;
        let mut first_err: Option<candle::Error> = None;
        for (r, w) in self.workers.iter().enumerate() {
            let resp = w
                .rx
                .recv()
                .map_err(|_| candle::Error::Msg(format!("worker {r} rx closed")))?;
            match resp {
                Ok(Some(t)) if r == 0 => rank0 = Some(t),
                Ok(_) => (),
                Err(e) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
            }
        }
        if let Some(e) = first_err {
            return Err(e);
        }
        rank0.ok_or_else(|| candle::Error::Msg("no rank 0 forward output".into()))
    }

    pub fn cluster(&self) -> &Arc<HipCluster> {
        &self.cluster
    }

    pub fn clear_kv_cache(&self) {
        for w in &self.workers {
            let _ = w.tx.send(WorkerReq::ClearKv);
        }
        for w in &self.workers {
            let _ = w.rx.recv();
        }
    }
}

impl Drop for ModelWeightsTp {
    fn drop(&mut self) {
        for w in &self.workers {
            let _ = w.tx.send(WorkerReq::Shutdown);
        }
        for w in &mut self.workers {
            if let Some(j) = w.join.take() {
                let _ = j.join();
            }
        }
    }
}
