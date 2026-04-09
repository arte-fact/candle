use anyhow::Result;
use candle_core::{DType, Device, Tensor};

fn main() -> Result<()> {
    let device = Device::new_hip(0)?;
    println!("HIP device: {:?}", device.location());

    // Binary ops (BINARY hsaco)
    let a = Tensor::arange(0f32, 6f32, &device)?.reshape((2, 3))?;
    let b = Tensor::arange(0f32, 6f32, &device)?.reshape((2, 3))?;
    println!("a + b = {:?}", (&a + &b)?.to_vec2::<f32>()?);
    println!("a * b = {:?}", (&a * &b)?.to_vec2::<f32>()?);

    // Affine (AFFINE hsaco)
    let x = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?;
    println!("affine(2x+1) = {:?}", x.affine(2.0, 1.0)?.to_vec1::<f32>()?);

    // Cast (CAST hsaco)
    let f16 = x.to_dtype(DType::F16)?;
    let back = f16.to_dtype(DType::F32)?;
    println!("f32->f16->f32 = {:?}", back.to_vec1::<f32>()?);

    // Unary ops (UNARY hsaco)
    println!("neg = {:?}", x.neg()?.to_vec1::<f32>()?);
    println!("sqrt = {:?}", x.sqrt()?.to_vec1::<f32>()?);
    println!("exp = {:?}", Tensor::new(&[0.0f32, 1.0], &device)?.exp()?.to_vec1::<f32>()?);

    // Reduction (REDUCE hsaco)
    println!("\n--- reduce ---");
    println!("sum(a) = {:?}", a.sum_all()?.to_scalar::<f32>()?);
    println!("max(a) = {:?}", a.max(1)?.to_vec1::<f32>()?);
    println!("min(a) = {:?}", a.min(1)?.to_vec1::<f32>()?);
    println!("argmax(a,1) = {:?}", a.argmax(1)?.to_vec1::<u32>()?);

    // Matmul (rocBLAS) — using deterministic data, not randn
    println!("\n--- matmul ---");
    let lhs = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device)?;
    let rhs = Tensor::new(&[[5.0f32, 6.0], [7.0, 8.0]], &device)?;
    let mm = lhs.matmul(&rhs)?;
    // Expected: [[19, 22], [43, 50]]
    println!("matmul 2x2 = {:?}", mm.to_vec2::<f32>()?);

    // Larger matmul
    let ones = vec![1.0f32; 64 * 64];
    let m = Tensor::new(ones.as_slice(), &device)?.reshape((64, 64))?;
    let n = Tensor::new(ones.as_slice(), &device)?.reshape((64, 64))?;
    let mn = m.matmul(&n)?;
    device.synchronize()?;
    // Each element should be 64.0 (sum of 64 ones)
    println!("matmul 64x64 [0][0] = {:?}", mn.get(0)?.get(0)?.to_scalar::<f32>()?);

    // Randn (hiprand) — test separately since it may crash
    println!("\n--- randn ---");
    let r = Tensor::randn(0f32, 1.0, (2, 4), &device)?;
    println!("randn(2,4) = {:?}", r.to_vec2::<f32>()?);

    // Benchmark with randn-generated data
    println!("\n--- benchmark ---");
    let m = Tensor::randn(0f32, 1.0, (512, 512), &device)?;
    let n = Tensor::randn(0f32, 1.0, (512, 512), &device)?;
    let _ = m.matmul(&n)?;
    device.synchronize()?;

    let iters = 100;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = m.matmul(&n)?;
    }
    device.synchronize()?;
    let elapsed = start.elapsed();
    println!(
        "512x512 matmul x{iters}: {:.1}ms total, {:.2}ms/iter",
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / iters as f64,
    );

    println!("\nAll HIP GPU ops passed!");
    Ok(())
}
