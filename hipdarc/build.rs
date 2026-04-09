use std::env;

fn main() {
    let rocm_path = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());

    println!("cargo:rustc-link-search=native={rocm_path}/lib");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=rocblas");
    // hiprand/rocrand omitted: segfaults on some ROCm installations.
    // RNG is done on CPU with rand crate, then uploaded to device.
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
}
