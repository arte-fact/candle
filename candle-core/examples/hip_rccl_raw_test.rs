//! Raw RCCL AllReduce test using group API (single-threaded, no candle Tensors).

use anyhow::Result;
use hipdarc::driver::{HipDevice, HipStream};
use hipdarc::rccl::{self, Comm, DataType, NcclUniqueId, ReduceOp};
use std::sync::Arc;

fn main() -> Result<()> {
    let num_gpus = 4usize;
    println!("=== Raw RCCL AllReduce Test ({num_gpus} GPUs) ===");

    let id = NcclUniqueId::new().map_err(|e| anyhow::anyhow!("{e}"))?;

    // Init all comms concurrently via threads (CommInitRank is blocking collective)
    let handles: Vec<_> = (0..num_gpus)
        .map(|g| {
            let id = id;
            std::thread::spawn(move || -> Result<(Arc<HipStream>, Arc<Comm>)> {
                let dev = HipDevice::new(g).map_err(|e| anyhow::anyhow!("{e}"))?;
                let stream = Arc::new(HipStream::new(&dev).map_err(|e| anyhow::anyhow!("{e}"))?);
                let comm = Comm::new(num_gpus, id, g, &stream)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                Ok((stream, Arc::new(comm)))
            })
        })
        .collect();

    let mut streams = Vec::new();
    let mut comms = Vec::new();
    for (g, h) in handles.into_iter().enumerate() {
        let (s, c) = h.join().map_err(|_| anyhow::anyhow!("thread {g} panic"))??;
        streams.push(s);
        comms.push(c);
    }
    println!("All comms initialized");

    // Allocate src/dst on each GPU
    let mut srcs = Vec::new();
    let mut dsts = Vec::new();
    for g in 0..num_gpus {
        let val = (g + 1) as f32;
        streams[g].device().set_current().map_err(|e| anyhow::anyhow!("{e}"))?;
        let src = streams[g].clone_htod(&[val, val * 10.0, val * 100.0])
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let dst = streams[g].alloc_zeros::<f32>(3)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        streams[g].synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
        println!("GPU {g}: src=[{val}, {}, {}]", val * 10.0, val * 100.0);
        srcs.push(src);
        dsts.push(dst);
    }

    // AllReduce using group API (single thread, batched launch)
    println!("Launching AllReduce via group API...");
    rccl::group_start().map_err(|e| anyhow::anyhow!("{e}"))?;
    for g in 0..num_gpus {
        unsafe {
            comms[g].all_reduce(
                srcs[g].device_ptr() as *const _,
                dsts[g].device_ptr() as *mut _,
                3,
                DataType::Float32,
                ReduceOp::Sum,
                &streams[g],
            ).map_err(|e| anyhow::anyhow!("AllReduce GPU {g}: {e}"))?;
        }
    }
    rccl::group_end().map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("Group end OK");

    // Sync all streams
    for g in 0..num_gpus {
        streams[g].synchronize().map_err(|e| anyhow::anyhow!("sync GPU {g}: {e}"))?;
    }

    // Read results
    let expected: f32 = (1..=num_gpus).map(|r| r as f32).sum();
    for g in 0..num_gpus {
        let result = streams[g].clone_dtoh(&dsts[g]).map_err(|e| anyhow::anyhow!("{e}"))?;
        println!(
            "GPU {g}: result = {:?} (expected [{expected}, {}, {}])",
            result, expected * 10.0, expected * 100.0,
        );
        assert!((result[0] - expected).abs() < 0.01);
    }

    println!("\nRCCL AllReduce PASSED!");
    Ok(())
}
