//! Raw FFI bindings for HIP runtime, rocBLAS, and hiprand.
//! Enum variant names match the C API for clarity.
#![allow(non_camel_case_types, dead_code, clippy::enum_variant_names)]

use libc::{c_char, c_int, c_uint, c_void, size_t};

// ---------------------------------------------------------------------------
// HIP types
// ---------------------------------------------------------------------------
pub type hipDevice_t = c_int;
pub type hipStream_t = *mut c_void;
pub type hipModule_t = *mut c_void;
pub type hipFunction_t = *mut c_void;
pub type hipDeviceptr_t = *mut c_void;
pub type hipEvent_t = *mut c_void;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum hipError_t {
    hipSuccess = 0,
    hipErrorInvalidValue = 1,
    hipErrorOutOfMemory = 2,
    // hipErrorMemoryAllocation is an alias for hipErrorOutOfMemory (both = 2)
    hipErrorNotInitialized = 3,
    hipErrorDeinitialized = 4,
    hipErrorInvalidDevice = 100,
    hipErrorInvalidImage = 200,
    hipErrorInvalidContext = 201,
    hipErrorInvalidKernelFile = 218,
    hipErrorInvalidHandle = 400,
    hipErrorNotFound = 500,
    hipErrorNotReady = 600,
    hipErrorLaunchFailure = 719,
    hipErrorUnknown = 999,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct hipDeviceProp_t {
    pub name: [c_char; 256],
    pub total_global_mem: size_t,
    pub shared_mem_per_block: size_t,
    pub regs_per_block: c_int,
    pub warp_size: c_int,
    pub max_threads_per_block: c_int,
    pub max_threads_dim: [c_int; 3],
    pub max_grid_size: [c_int; 3],
    pub clock_rate: c_int,
    pub memory_clock_rate: c_int,
    pub memory_bus_width: c_int,
    pub total_const_mem: size_t,
    pub major: c_int,
    pub minor: c_int,
    pub multi_processor_count: c_int,
    pub l2_cache_size: c_int,
    pub max_threads_per_multi_processor: c_int,
    pub compute_mode: c_int,
    pub gc_n_arch_name: [c_char; 256],
    // Padding for forward compatibility — the real struct is larger but we
    // only need the fields above. 4096 bytes covers ROCm 5.x–7.x layouts.
    _pad: [u8; 2048],
}

// ---------------------------------------------------------------------------
// HIP runtime API
// ---------------------------------------------------------------------------
extern "C" {
    pub fn hipInit(flags: c_uint) -> hipError_t;
    pub fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
    pub fn hipSetDevice(device_id: c_int) -> hipError_t;
    pub fn hipGetDevice(device_id: *mut c_int) -> hipError_t;
    pub fn hipGetDeviceProperties(prop: *mut hipDeviceProp_t, device_id: c_int) -> hipError_t;
    pub fn hipDeviceSynchronize() -> hipError_t;

    // Memory
    pub fn hipMalloc(ptr: *mut hipDeviceptr_t, size: size_t) -> hipError_t;
    pub fn hipFree(ptr: hipDeviceptr_t) -> hipError_t;
    /// Stream-ordered async allocator. Backed by an internal memory pool
    /// in the HIP runtime — orders of magnitude cheaper than the
    /// synchronous `hipMalloc` for the per-op temporary buffers that
    /// candle's HIP backend creates by the hundred per decoded token.
    /// Available since ROCm 5.1.
    pub fn hipMallocAsync(
        ptr: *mut hipDeviceptr_t,
        size: size_t,
        stream: hipStream_t,
    ) -> hipError_t;
    pub fn hipFreeAsync(ptr: hipDeviceptr_t, stream: hipStream_t) -> hipError_t;
    pub fn hipMemset(dst: hipDeviceptr_t, value: c_int, size_bytes: size_t) -> hipError_t;
    pub fn hipMemsetAsync(
        dst: hipDeviceptr_t,
        value: c_int,
        size_bytes: size_t,
        stream: hipStream_t,
    ) -> hipError_t;
    pub fn hipMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        size_bytes: size_t,
        kind: hipMemcpyKind,
    ) -> hipError_t;
    pub fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        size_bytes: size_t,
        kind: hipMemcpyKind,
        stream: hipStream_t,
    ) -> hipError_t;

    // Streams
    pub fn hipStreamCreate(stream: *mut hipStream_t) -> hipError_t;
    pub fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t;
    pub fn hipStreamDestroy(stream: hipStream_t) -> hipError_t;

    // Modules (load compiled GPU code objects)
    pub fn hipModuleLoadData(module: *mut hipModule_t, image: *const c_void) -> hipError_t;
    pub fn hipModuleUnload(module: hipModule_t) -> hipError_t;
    pub fn hipModuleGetFunction(
        function: *mut hipFunction_t,
        module: hipModule_t,
        name: *const c_char,
    ) -> hipError_t;
    pub fn hipModuleLaunchKernel(
        f: hipFunction_t,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: hipStream_t,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> hipError_t;
}

// ---------------------------------------------------------------------------
// rocBLAS types and API
// ---------------------------------------------------------------------------
pub type rocblas_handle = *mut c_void;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum rocblas_status {
    rocblas_status_success = 0,
    rocblas_status_invalid_handle = 1,
    rocblas_status_not_implemented = 2,
    rocblas_status_invalid_pointer = 3,
    rocblas_status_invalid_size = 4,
    rocblas_status_memory_error = 5,
    rocblas_status_internal_error = 6,
    rocblas_status_perf_degraded = 7,
    rocblas_status_size_query_mismatch = 8,
    rocblas_status_size_increased = 9,
    rocblas_status_size_unchanged = 10,
    rocblas_status_invalid_value = 11,
    rocblas_status_continue = 12,
    rocblas_status_check_numerics_fail = 13,
    rocblas_status_excluded_from_build = 14,
    rocblas_status_arch_mismatch = 15,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum rocblas_operation {
    rocblas_operation_none = 111,
    rocblas_operation_transpose = 112,
    rocblas_operation_conjugate_transpose = 113,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum rocblas_datatype {
    rocblas_datatype_f16_r = 150,
    rocblas_datatype_f32_r = 151,
    rocblas_datatype_f64_r = 152,
    rocblas_datatype_f16_c = 153,
    rocblas_datatype_f32_c = 154,
    rocblas_datatype_f64_c = 155,
    rocblas_datatype_bf16_r = 168,
    rocblas_datatype_i8_r = 160,
    rocblas_datatype_i32_r = 162,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum rocblas_gemm_algo {
    rocblas_gemm_algo_standard = 0,
}

extern "C" {
    pub fn rocblas_create_handle(handle: *mut rocblas_handle) -> rocblas_status;
    pub fn rocblas_destroy_handle(handle: rocblas_handle) -> rocblas_status;
    pub fn rocblas_set_stream(handle: rocblas_handle, stream: hipStream_t) -> rocblas_status;

    pub fn rocblas_gemm_strided_batched_ex(
        handle: rocblas_handle,
        trans_a: rocblas_operation,
        trans_b: rocblas_operation,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        a_type: rocblas_datatype,
        lda: c_int,
        stride_a: i64,
        b: *const c_void,
        b_type: rocblas_datatype,
        ldb: c_int,
        stride_b: i64,
        beta: *const c_void,
        c: *mut c_void,
        c_type: rocblas_datatype,
        ldc: c_int,
        stride_c: i64,
        d: *mut c_void,
        d_type: rocblas_datatype,
        ldd: c_int,
        stride_d: i64,
        batch_count: c_int,
        compute_type: rocblas_datatype,
        algo: rocblas_gemm_algo,
        solution_index: i32,
        flags: u32,
    ) -> rocblas_status;
}

// ---------------------------------------------------------------------------
// hiprand types and API
// ---------------------------------------------------------------------------
pub type hiprandGenerator_t = *mut c_void;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum hiprandStatus_t {
    HIPRAND_STATUS_SUCCESS = 0,
    HIPRAND_STATUS_VERSION_MISMATCH = 100,
    HIPRAND_STATUS_NOT_INITIALIZED = 101,
    HIPRAND_STATUS_ALLOCATION_FAILED = 102,
    HIPRAND_STATUS_TYPE_ERROR = 103,
    HIPRAND_STATUS_OUT_OF_RANGE = 104,
    HIPRAND_STATUS_LENGTH_NOT_MULTIPLE = 105,
    HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106,
    HIPRAND_STATUS_LAUNCH_FAILURE = 201,
    HIPRAND_STATUS_INTERNAL_ERROR = 999,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum hiprandRngType_t {
    HIPRAND_RNG_PSEUDO_DEFAULT = 400,
    HIPRAND_RNG_PSEUDO_XORWOW = 401,
    HIPRAND_RNG_PSEUDO_MRG32K3A = 402,
    HIPRAND_RNG_PSEUDO_MTGP32 = 403,
    HIPRAND_RNG_PSEUDO_PHILOX4_32_10 = 404,
    HIPRAND_RNG_QUASI_DEFAULT = 500,
    HIPRAND_RNG_QUASI_SOBOL32 = 501,
}

extern "C" {
    pub fn hiprandCreateGenerator(
        generator: *mut hiprandGenerator_t,
        rng_type: hiprandRngType_t,
    ) -> hiprandStatus_t;
    pub fn hiprandDestroyGenerator(generator: hiprandGenerator_t) -> hiprandStatus_t;
    pub fn hiprandSetStream(
        generator: hiprandGenerator_t,
        stream: hipStream_t,
    ) -> hiprandStatus_t;
    pub fn hiprandSetPseudoRandomGeneratorSeed(
        generator: hiprandGenerator_t,
        seed: u64,
    ) -> hiprandStatus_t;
    pub fn hiprandGenerateUniform(
        generator: hiprandGenerator_t,
        output_data: *mut f32,
        n: size_t,
    ) -> hiprandStatus_t;
    pub fn hiprandGenerateUniformDouble(
        generator: hiprandGenerator_t,
        output_data: *mut f64,
        n: size_t,
    ) -> hiprandStatus_t;
    pub fn hiprandGenerateNormal(
        generator: hiprandGenerator_t,
        output_data: *mut f32,
        n: size_t,
        mean: f32,
        stddev: f32,
    ) -> hiprandStatus_t;
    pub fn hiprandGenerateNormalDouble(
        generator: hiprandGenerator_t,
        output_data: *mut f64,
        n: size_t,
        mean: f64,
        stddev: f64,
    ) -> hiprandStatus_t;
}
