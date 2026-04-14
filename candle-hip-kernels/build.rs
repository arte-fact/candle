use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn which(name: &str) -> Option<PathBuf> {
    env::var_os("PATH").and_then(|paths| {
        env::split_paths(&paths)
            .map(|p| p.join(name))
            .find(|p| p.exists())
    })
}

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    // Rerun when files are added/removed from src/ (e.g. a new `.cu`).
    // Individual files already get their own rerun-if-changed below.
    println!("cargo::rerun-if-changed=src");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/hip_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");
    println!("cargo::rerun-if-changed=src/gfx906_primitives.cuh");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let rocm_path = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());

    // Find hipcc: try multiple locations.
    let candidates = [
        PathBuf::from(&rocm_path).join("bin/hipcc"),
        PathBuf::from("/opt/rocm/bin/hipcc"),
        PathBuf::from("/usr/bin/hipcc"),
        PathBuf::from("/usr/local/bin/hipcc"),
    ];
    let hipcc = candidates
        .iter()
        .find(|p| p.exists())
        .cloned()
        .or_else(|| which("hipcc"))
        .expect("hipcc not found. Set ROCM_PATH or ensure hipcc is in PATH.");

    let gfx_targets =
        env::var("HIP_OFFLOAD_ARCH").unwrap_or_else(|_| "gfx906".to_string());

    let src_dir = Path::new("src");
    let cu_files: Vec<_> = fs::read_dir(src_dir)
        .expect("cannot read src/")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "cu"))
        .collect();

    let mut hsaco_entries = Vec::new();

    for cu_path in &cu_files {
        let stem = cu_path.file_stem().unwrap().to_str().unwrap();
        let hsaco_path = out_dir.join(format!("{stem}.hsaco"));
        println!("cargo::rerun-if-changed={}", cu_path.display());

        // Compile to a raw GPU code object (ELF/HSACO) for hipModuleLoadData.
        // --no-gpu-bundle-output produces plain ELF instead of a Clang offload bundle.
        let output = Command::new(&hipcc)
            .args([
                "--cuda-device-only",
                "-c",
                "--no-gpu-bundle-output",
                &format!("--offload-arch={gfx_targets}"),
                "-std=c++17",
                "-O3",
                "-I",
                src_dir.to_str().unwrap(),
                "-D__HIP_PLATFORM_AMD__",
                "-DWARP_SIZE=64",
                "-o",
                hsaco_path.to_str().unwrap(),
                cu_path.to_str().unwrap(),
            ])
            .output()
            .unwrap_or_else(|e| panic!("failed to run hipcc at {}: {e}", hipcc.display()));

        // Always print hipcc output for diagnostics during development.
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stdout.is_empty() {
            println!("cargo:warning=hipcc stdout for {}: {}", stem, stdout.replace('\n', " | "));
        }
        if !stderr.is_empty() {
            println!("cargo:warning=hipcc stderr for {}: {}", stem, stderr.replace('\n', " | "));
        }
        if !output.status.success() {
            panic!(
                "hipcc failed for {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
                cu_path.display(),
                stdout,
                stderr
            );
        }

        hsaco_entries.push((stem.to_string(), hsaco_path));
    }

    // Generate hsaco.rs with include_bytes! for each compiled kernel.
    let mut hsaco_rs = String::new();
    for (stem, path) in &hsaco_entries {
        let const_name = stem.to_uppercase();
        hsaco_rs.push_str(&format!(
            "pub const {const_name}: &[u8] = include_bytes!(r\"{}\");\n",
            path.display()
        ));
    }
    fs::write(out_dir.join("hsaco.rs"), hsaco_rs).expect("failed to write hsaco.rs");
}
