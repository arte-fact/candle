//! Compare CPU vs GPU results for individual ops to find the correctness bug.
use anyhow::Result;
use candle_core::{Device, Tensor};

fn check(name: &str, cpu: &Tensor, gpu: &Tensor) -> Result<()> {
    let cpu_vals = cpu.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    let gpu_vals = gpu.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    let max_diff = cpu_vals
        .iter()
        .zip(gpu_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let status = if max_diff < 0.01 { "OK" } else { "MISMATCH" };
    println!("  {name}: max_diff={max_diff:.6} [{status}]");
    if max_diff >= 0.01 {
        println!("    cpu[0..4]: {:?}", &cpu_vals[..4.min(cpu_vals.len())]);
        println!("    gpu[0..4]: {:?}", &gpu_vals[..4.min(gpu_vals.len())]);
    }
    Ok(())
}

fn main() -> Result<()> {
    let cpu = Device::Cpu;
    let gpu = Device::new_hip(0)?;

    let data: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) / 256.0).collect();

    // Test 1: Basic arithmetic
    println!("=== Basic ops ===");
    let cpu_t = Tensor::new(data.as_slice(), &cpu)?;
    let gpu_t = Tensor::new(data.as_slice(), &gpu)?;
    let cpu_r = (&cpu_t * &cpu_t)?;
    let gpu_r = (&gpu_t * &gpu_t)?;
    check("mul", &cpu_r, &gpu_r)?;

    // Test 2: Matmul
    println!("\n=== Matmul ===");
    let m_data: Vec<f32> = (0..64 * 64).map(|i| ((i % 17) as f32 - 8.0) / 10.0).collect();
    let cpu_m = Tensor::new(m_data.as_slice(), &cpu)?.reshape((64, 64))?;
    let gpu_m = Tensor::new(m_data.as_slice(), &gpu)?.reshape((64, 64))?;
    let cpu_mm = cpu_m.matmul(&cpu_m)?;
    let gpu_mm = gpu_m.matmul(&gpu_m)?;
    check("matmul_64x64", &cpu_mm.flatten_all()?, &gpu_mm.flatten_all()?)?;

    // Test 3: Unary ops
    println!("\n=== Unary ops ===");
    let cpu_sqrt = cpu_t.abs()?.sqrt()?;
    let gpu_sqrt = gpu_t.abs()?.sqrt()?;
    check("sqrt", &cpu_sqrt, &gpu_sqrt)?;

    let cpu_neg = cpu_t.neg()?;
    let gpu_neg = gpu_t.neg()?;
    check("neg", &cpu_neg, &gpu_neg)?;

    // Test 4: Sum/Reduce
    println!("\n=== Reduce ===");
    let cpu_sum = cpu_t.reshape((8, 64))?.sum(1)?;
    let gpu_sum = gpu_t.reshape((8, 64))?.sum(1)?;
    check("sum_dim1", &cpu_sum, &gpu_sum)?;

    // Test 5: Quantized matmul
    println!("\n=== Quantized matmul ===");
    {
        use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
        use candle_core::Module;

        let w_data: Vec<f32> = (0..256 * 64)
            .map(|i| ((i % 19) as f32 - 9.0) / 50.0)
            .collect();
        let w_cpu = Tensor::new(w_data.as_slice(), &cpu)?.reshape((256, 64))?;
        let qw_cpu = QTensor::quantize(&w_cpu, GgmlDType::Q4_0)?;
        let qm_cpu = QMatMul::from_qtensor(qw_cpu)?;

        let w_gpu = Tensor::new(w_data.as_slice(), &gpu)?.reshape((256, 64))?;
        let qw_gpu = QTensor::quantize(&w_gpu, GgmlDType::Q4_0)?;
        let qm_gpu = QMatMul::from_qtensor(qw_gpu)?;

        let inp_data: Vec<f32> = (0..2 * 64).map(|i| (i as f32) / 64.0).collect();
        let cpu_inp = Tensor::new(inp_data.as_slice(), &cpu)?.reshape((2, 64))?;
        let gpu_inp = Tensor::new(inp_data.as_slice(), &gpu)?.reshape((2, 64))?;

        let cpu_out = qm_cpu.forward(&cpu_inp)?;
        let gpu_out = qm_gpu.forward(&gpu_inp)?;
        check("qmatmul_q4_0", &cpu_out.flatten_all()?, &gpu_out.flatten_all()?)?;
    }

    println!("\nDone!");
    Ok(())
}
