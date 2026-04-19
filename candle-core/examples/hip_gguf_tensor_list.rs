//! Dump tensor names × dtype × shape for a GGUF. Useful for auditing what
//! matmul paths fire on which tensors.
//!
//! Usage: cargo run --release --features hip -p candle-core --example hip_gguf_tensor_list -- PATH [DTYPE_FILTER]
use anyhow::Result;
use candle_core::quantized::gguf_file::Content;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).unwrap_or(
        "/artefact/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf",
    );
    let filter = args.get(2).cloned();

    let mut f = std::fs::File::open(path)?;
    let c = Content::read(&mut f)?;
    let mut out: Vec<(String, String, Vec<usize>)> = c
        .tensor_infos
        .iter()
        .map(|(n, i)| (format!("{:?}", i.ggml_dtype), n.clone(), i.shape.dims().to_vec()))
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    for (dt, name, shape) in &out {
        if let Some(ref f) = filter {
            if !dt.contains(f) { continue; }
        }
        println!("{:8} {:40} {:?}", dt, name, shape);
    }
    Ok(())
}
