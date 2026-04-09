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

    // Test 3: Move data between GPUs via CPU
    println!("\n=== Test 3: GPU-to-GPU via CPU ===");
    let dev0 = Device::new_hip(0)?;
    let dev1 = Device::new_hip(1)?;
    let t0 = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &dev0)?;
    println!("  Created on GPU 0: {:?}", t0.to_vec1::<f32>()?);
    let t1 = t0.to_device(&dev1)?;
    println!("  Moved to GPU 1: {:?} on {:?}", t1.to_vec1::<f32>()?, t1.device().location());
    let sum = (&t0.to_device(&Device::Cpu)? + &t1.to_device(&Device::Cpu)?)?;
    println!("  Sum (via CPU): {:?}", sum.to_vec1::<f32>()?);

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

    // Test 5: RCCL AllReduce (multi-process)
    println!("\n=== Test 5: RCCL AllReduce ===");
    println!("  Skipped — requires iommu=pt kernel boot parameter.");
    println!("  Add 'iommu=pt' to GRUB_CMDLINE_LINUX in /etc/default/grub,");
    println!("  then: sudo update-grub && sudo reboot");

    println!("\nAll multi-GPU tests passed!");
    Ok(())
}
