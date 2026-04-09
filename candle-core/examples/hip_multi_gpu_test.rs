//! Test multi-GPU AllReduce via RCCL on HIP.
//! Spawns N processes (one per GPU), each creates a tensor,
//! and AllReduce sums them across all GPUs via a CustomOp1.
//!
//! Usage: cargo run --release --features hip -p candle-core --example hip_multi_gpu_test

use anyhow::Result;
use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, Device, HipStorage, Layout, Shape, Tensor};
use hipdarc::rccl::{Comm, DataType, NcclUniqueId, ReduceOp, NCCL_UNIQUE_ID_BYTES};
use std::io::Write;
use std::sync::Arc;

struct AllReduceSum {
    comm: Arc<Comm>,
}

impl CustomOp1 for AllReduceSum {
    fn name(&self) -> &'static str {
        "allreduce_sum"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("AllReduce is not implemented on CPU")
    }

    fn hip_fwd(&self, s: &HipStorage, l: &Layout) -> candle_core::Result<(HipStorage, Shape)> {
        let elem_count = l.shape().elem_count();
        let dev = s.device();
        let dst = dev.alloc_zeros::<f32>(elem_count)?;

        let (src_ptr, dst_ptr) = match (&s.slice, &dst) {
            (candle_core::hip_backend::HipStorageSlice::F32(src), dst_s) => {
                (src.device_ptr(), dst_s.device_ptr())
            }
            _ => candle_core::bail!("AllReduce only supports f32"),
        };

        unsafe {
            self.comm
                .all_reduce(
                    src_ptr as *const _,
                    dst_ptr as *mut _,
                    elem_count,
                    DataType::Float32,
                    ReduceOp::Sum,
                    dev.stream(),
                )
                .map_err(|e| candle_core::Error::Msg(format!("RCCL AllReduce: {e}")))?;
        }

        let out = HipStorage {
            slice: candle_core::hip_backend::HipStorageSlice::F32(dst),
            device: dev.clone(),
        };
        Ok((out, l.shape().clone()))
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let mut num_gpus: usize = 0;
    let mut rank: Option<usize> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--num-gpus" => { num_gpus = args[i + 1].parse()?; i += 2; }
            "--rank" => { rank = Some(args[i + 1].parse()?); i += 2; }
            _ => i += 1,
        }
    }

    if num_gpus == 0 {
        for g in 0..16 {
            match Device::new_hip(g) {
                Ok(_) => num_gpus = g + 1,
                Err(_) => break,
            }
        }
        println!("Detected {num_gpus} HIP GPUs");
    }

    if num_gpus < 2 {
        println!("Need at least 2 GPUs for multi-GPU test");
        return Ok(());
    }

    let comm_file = std::path::PathBuf::from("/tmp/candle_rccl_test.id");

    match rank {
        None => {
            println!("=== Multi-GPU AllReduce Test ({num_gpus} GPUs) ===");
            let _ = std::fs::remove_file(&comm_file);

            let exe = std::env::current_exe()?;
            let children: Vec<_> = (0..num_gpus)
                .map(|r| {
                    std::process::Command::new(&exe)
                        .args(["--num-gpus", &num_gpus.to_string(), "--rank", &r.to_string()])
                        .spawn()
                        .expect("failed to spawn child")
                })
                .collect();

            for mut child in children {
                let status = child.wait()?;
                if !status.success() {
                    anyhow::bail!("Child process failed: {status}");
                }
            }

            let _ = std::fs::remove_file(&comm_file);
            println!("\n=== All ranks completed successfully ===");
        }
        Some(rank) => run_rank(rank, num_gpus, &comm_file)?,
    }

    Ok(())
}

fn run_rank(rank: usize, num_gpus: usize, comm_file: &std::path::Path) -> Result<()> {
    let device = Device::new_hip(rank)?;
    let hip_dev = device.as_hip_device()?;

    // Exchange unique ID via file
    let id = if rank == 0 {
        let id = NcclUniqueId::new().map_err(|e| anyhow::anyhow!("RCCL: {e}"))?;
        let tmp = comm_file.with_extension("tmp");
        std::fs::File::create(&tmp)?.write_all(id.as_bytes())?;
        std::fs::rename(&tmp, comm_file)?;
        id
    } else {
        while !comm_file.exists() {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
        let data = std::fs::read(comm_file)?;
        let bytes: [u8; NCCL_UNIQUE_ID_BYTES] = data.try_into().map_err(|_| {
            anyhow::anyhow!("Invalid RCCL unique ID length")
        })?;
        NcclUniqueId::from_bytes(&bytes)
    };

    let comm = Arc::new(
        Comm::new(num_gpus, id, rank, hip_dev.stream())
            .map_err(|e| anyhow::anyhow!("RCCL comm init: {e}"))?
    );

    // Each rank creates a tensor with value = rank + 1
    let val = (rank + 1) as f32;
    let input = Tensor::new(&[val, val * 10.0, val * 100.0], &device)?;
    println!("Rank {rank}: input = {:?}", input.to_vec1::<f32>()?);

    // AllReduce via CustomOp1
    let all_reduce = AllReduceSum { comm };
    let output = input.apply_op1_no_bwd(&all_reduce)?;

    let result = output.to_vec1::<f32>()?;
    let expected: f32 = (1..=num_gpus as u32).map(|r| r as f32).sum();
    println!(
        "Rank {rank}: result = {:?} (expected [{expected}, {}, {}])",
        result,
        expected * 10.0,
        expected * 100.0,
    );

    assert!(
        (result[0] - expected).abs() < 0.01,
        "AllReduce mismatch on rank {rank}!"
    );

    println!("Rank {rank}: PASSED");
    Ok(())
}
