//! Test quantized tensor operations on HIP GPU.
use anyhow::Result;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Tensor};

fn main() -> Result<()> {
    let device = Device::new_hip(0)?;
    println!("HIP device: {:?}", device.location());

    // Create a float tensor on GPU, quantize to Q4_0, dequantize back
    println!("\n--- Q4_0 round-trip on GPU ---");
    let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
    let tensor = Tensor::new(data.as_slice(), &device)?;
    println!("Source tensor on {:?}", tensor.device().location());

    let qtensor = QTensor::quantize(&tensor, GgmlDType::Q4_0)?;
    println!("Quantized: {} bytes, dtype {:?}", qtensor.storage_size_in_bytes(), qtensor.dtype());

    let dequant = qtensor.dequantize(&device)?;
    let dequant_vals = dequant.to_vec1::<f32>()?;
    println!("Dequant[0..8]: {:?}", &dequant_vals[..8]);
    println!("Original[0..8]: {:?}", &data[..8]);

    // Quantized matmul via QMatMul
    println!("\n--- QMatMul on GPU ---");
    let weight_data: Vec<f32> = (0..512 * 256)
        .map(|i| ((i % 17) as f32 - 8.0) / 100.0)
        .collect();
    let weight = Tensor::new(weight_data.as_slice(), &device)?.reshape((512, 256))?;
    let qweight = QTensor::quantize(&weight, GgmlDType::Q4_0)?;
    let qmatmul = QMatMul::from_qtensor(qweight)?;

    let input_data: Vec<f32> = (0..4 * 256)
        .map(|i| ((i % 13) as f32 - 6.0) / 10.0)
        .collect();
    let input = Tensor::new(input_data.as_slice(), &device)?.reshape((4, 256))?;
    let output = qmatmul.forward(&input)?;
    println!("Output shape: {:?}", output.shape());

    let out_vals = output.to_vec2::<f32>()?;
    println!("Output[0][0..4]: {:?}", &out_vals[0][..4]);

    let sum = output.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert!(sum > 0.0, "Output should be non-zero");
    println!("abs_sum: {sum:.2}");

    println!("\nAll quantized GPU ops passed!");
    Ok(())
}
