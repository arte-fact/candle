use std::env;

fn main() {
    let rocm_path = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());

    println!("cargo:rustc-link-search=native={rocm_path}/lib");

    // Also search alternate ROCm installations for libraries like rocblas
    // that may be installed from a different ROCm version
    let rocm_alt = env::var("ROCM_ALT_PATH").ok();
    if let Some(ref alt) = rocm_alt {
        println!("cargo:rustc-link-search=native={alt}/lib");
    }
    // Also search /opt/rocm-7.1.1 as a common alternate location
    if std::path::Path::new("/opt/rocm-7.1.1/lib").exists() {
        println!("cargo:rustc-link-search=native=/opt/rocm-7.1.1/lib");
    }

    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=rocblas");
    println!("cargo:rustc-link-lib=dylib=rccl");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_ALT_PATH");
}
