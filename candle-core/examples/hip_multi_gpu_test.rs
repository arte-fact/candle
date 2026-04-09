//! Test multi-GPU capabilities on HIP.
//! Tests both direct multi-device tensor ops and RCCL AllReduce.

use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    // Detect GPUs
    let mut num_gpus = 0;
    for g in 0..16 {
        match Device::new_hip(g) {
            Ok(_) => num_gpus = g + 1,
            Err(_) => break,
        }
    }
    println!("Detected {num_gpus} HIP GPUs");

    if num_gpus < 2 {
        println!("Need at least 2 GPUs");
        return Ok(());
    }

    // Test 1: Create tensors on each GPU independently
    println!("\n=== Test 1: Independent GPU tensors ===");
    for g in 0..num_gpus {
        let dev = Device::new_hip(g)?;
        let t = Tensor::new(&[g as f32 + 1.0, (g as f32 + 1.0) * 10.0], &dev)?;
        println!("  GPU {g}: {:?} on {:?}", t.to_vec1::<f32>()?, dev.location());
    }

    // Test 2: Matmul on each GPU
    println!("\n=== Test 2: Matmul on each GPU ===");
    for g in 0..num_gpus {
        let dev = Device::new_hip(g)?;
        let a = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &dev)?;
        let b = Tensor::new(&[[5.0f32, 6.0], [7.0, 8.0]], &dev)?;
        let c = a.matmul(&b)?;
        println!("  GPU {g}: matmul = {:?}", c.to_vec2::<f32>()?);
    }

    // Test 3: Move data between GPUs (GPU0 → CPU → GPU1)
    println!("\n=== Test 3: GPU-to-GPU via CPU ===");
    let dev0 = Device::new_hip(0)?;
    let dev1 = Device::new_hip(1)?;
    let t0 = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &dev0)?;
    println!("  Created on GPU 0: {:?}", t0.to_vec1::<f32>()?);
    // GPU→GPU transfer goes through CPU
    let t1 = t0.to_device(&Device::Cpu)?.to_device(&dev1)?;
    println!("  Moved to GPU 1: {:?} on {:?}", t1.to_vec1::<f32>()?, t1.device().location());
    let sum_cpu = (&t0.to_device(&Device::Cpu)? + &t1.to_device(&Device::Cpu)?)?;
    println!("  Sum (via CPU): {:?}", sum_cpu.to_vec1::<f32>()?);

    // Test 4: Parallel matmul on all GPUs with same data
    println!("\n=== Test 4: Parallel matmul across {num_gpus} GPUs ===");
    let data_a: Vec<f32> = (0..64*64).map(|i| (i % 7) as f32 / 7.0).collect();
    let data_b: Vec<f32> = (0..64*64).map(|i| (i % 11) as f32 / 11.0).collect();
    for g in 0..num_gpus {
        let dev = Device::new_hip(g)?;
        let a = Tensor::new(data_a.as_slice(), &dev)?.reshape((64, 64))?;
        let b = Tensor::new(data_b.as_slice(), &dev)?.reshape((64, 64))?;
        let c = a.matmul(&b)?;
        dev.as_hip_device()?.stream().synchronize()
            .map_err(|e| anyhow::anyhow!("sync GPU {g}: {e}"))?;
        let val = c.get(0)?.get(0)?.to_scalar::<f32>()?;
        println!("  GPU {g}: 64x64 matmul [0][0] = {val:.4}");
    }

    // Test 5: RCCL AllReduce (multi-process, requires iommu=pt)
    println!("\n=== Test 5: RCCL AllReduce ({num_gpus} GPUs) ===");
    {
        use hipdarc::rccl::{Comm, DataType, NcclUniqueId, ReduceOp};
        use std::sync::Arc;

        // Single-process multi-device: init all comms in one process
        let id = NcclUniqueId::new().map_err(|e| anyhow::anyhow!("RCCL: {e}"))?;

        // CommInitRank is a blocking collective — all ranks must call it
        // concurrently. Use threads to avoid deadlock.
        let handles: Vec<_> = (0..num_gpus)
            .map(|g| {
                let id = id;
                std::thread::spawn(move || -> Result<(Device, Arc<Comm>)> {
                    let dev = Device::new_hip(g)?;
                    let hip = dev.as_hip_device()?;
                    let comm = Comm::new(num_gpus, id, g, hip.stream())
                        .map_err(|e| anyhow::anyhow!("RCCL init GPU {g}: {e}"))?;
                    Ok((dev, Arc::new(comm)))
                })
            })
            .collect();

        let mut devs = Vec::new();
        let mut comms = Vec::new();
        for (g, h) in handles.into_iter().enumerate() {
            let (dev, comm) = h.join().map_err(|_| anyhow::anyhow!("Thread {g} panicked"))??;
            devs.push(dev);
            comms.push(comm);
        }
        println!("  All {num_gpus} communicators initialized");

        // Use CustomOp1 to access storage internals
        struct AllReduceTest(Arc<Comm>);
        impl candle_core::CustomOp1 for AllReduceTest {
            fn name(&self) -> &'static str { "allreduce_test" }
            fn cpu_fwd(&self, _: &candle_core::CpuStorage, _: &candle_core::Layout)
                -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
                candle_core::bail!("cpu not supported")
            }
            fn hip_fwd(&self, s: &candle_core::HipStorage, l: &candle_core::Layout)
                -> candle_core::Result<(candle_core::HipStorage, candle_core::Shape)> {
                use candle_core::backend::BackendStorage;
                let dev = s.device();
                let n = l.shape().elem_count();
                let src_slice = match &s.slice {
                    candle_core::hip_backend::HipStorageSlice::F32(sl) => sl,
                    _ => candle_core::bail!("f32 only"),
                };
                // Ensure correct GPU context in this thread
                dev.stream().device().set_current()
                    .map_err(|e| candle_core::Error::Msg(format!("set_current: {e}")))?;
                // Out-of-place AllReduce: separate src and dst buffers
                let dst_slice = dev.alloc_zeros::<f32>(n)?;
                dev.stream().synchronize().map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
                unsafe {
                    self.0.all_reduce(
                        src_slice.device_ptr() as *const _,
                        dst_slice.device_ptr() as *mut _,
                        n,
                        DataType::Float32, ReduceOp::Sum, dev.stream())
                        .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
                }
                dev.stream().synchronize().map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
                Ok((candle_core::HipStorage {
                    slice: candle_core::hip_backend::HipStorageSlice::F32(dst_slice),
                    device: dev.clone(),
                }, l.shape().clone()))
            }
        }

        // AllReduce is also a collective — all ranks must call concurrently.
        let expected: f32 = (1..=num_gpus).map(|r| r as f32).sum();

        // Create inputs on each GPU
        let mut inputs = Vec::new();
        for g in 0..num_gpus {
            let v = (g + 1) as f32;
            inputs.push(Tensor::new(&[v, v * 10.0, v * 100.0], &devs[g])?);
        }

        // Launch AllReduce on all GPUs concurrently via threads
        let ar_handles: Vec<_> = (0..num_gpus)
            .map(|g| {
                let input = inputs[g].clone();
                let comm = comms[g].clone();
                std::thread::spawn(move || -> Result<Vec<f32>> {
                    let op = AllReduceTest(comm);
                    let output = input.apply_op1_no_bwd(&op)?;
                    Ok(output.to_vec1::<f32>()?)
                })
            })
            .collect();

        for (g, h) in ar_handles.into_iter().enumerate() {
            let result = h.join().map_err(|_| anyhow::anyhow!("AR thread {g} panicked"))??;
            println!(
                "  GPU {g}: allreduce = {:?} (expected [{expected}, {}, {}])",
                result, expected * 10.0, expected * 100.0,
            );
            assert!((result[0] - expected).abs() < 0.01, "Mismatch on GPU {g}");
        }
        println!("  RCCL AllReduce verified!");
    }

    println!("\nAll multi-GPU tests passed!");
    Ok(())
}
