use crate::backend::{BackendDevice, BackendStorage};
use crate::{CpuStorage, DType, Result, Shape};
pub use candle_hip_kernels as kernels;
use half::{bf16, f16};
use hipdarc::driver::{HipDevice as RawHipDevice, HipFunc, HipModule, HipStream};
use hipdarc::rocblas::RocBlas;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use super::{HipError, HipStorage, HipStorageSlice, WrapErr};

/// Unique identifier for HIP devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

pub struct ModuleStore {
    mdls: [Option<Arc<HipModule>>; kernels::ALL_IDS.len()],
}

#[derive(Clone)]
pub struct HipDevice {
    id: DeviceId,
    ordinal: usize,
    stream: Arc<HipStream>,
    modules: Arc<RwLock<ModuleStore>>,
    custom_modules: Arc<RwLock<HashMap<String, Arc<HipModule>>>>,
    pub(crate) blas: Arc<std::sync::OnceLock<RocBlas>>,
    blas_stream: Arc<HipStream>,
    // CPU-based RNG: generate on host, upload to device. hiprand/rocrand
    // segfaults on some ROCm installations so we avoid it entirely.
    cpu_rng: Arc<Mutex<rand::rngs::StdRng>>,
    seed_value: Arc<RwLock<u64>>,
}

impl std::fmt::Debug for HipDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HipDevice({:?})", self.id)
    }
}

// -- Low-level helpers (mirror CudaDevice) -----------------------------------

impl HipDevice {
    pub fn stream(&self) -> &Arc<HipStream> {
        &self.stream
    }

    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Allocate uninitialized device memory.
    ///
    /// # Safety
    /// The returned memory is uninitialized.
    pub unsafe fn alloc<T: hipdarc::driver::DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<hipdarc::driver::HipSlice<T>> {
        self.stream.alloc::<T>(len).w()
    }

    pub fn alloc_zeros<T: hipdarc::driver::ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<hipdarc::driver::HipSlice<T>> {
        // O3 diagnostic: when CANDLE_TRACE_ZEROS=1, dump a stack trace and
        // element count so we can localise which call site emits the
        // `__amd_rocclr_fillBufferAligned` dispatches that the rocprofv3
        // trace shows as 18-35 % of decode GPU time.  Only enable for
        // short runs — the backtrace capture is very expensive.
        if std::env::var_os("CANDLE_TRACE_ZEROS").is_some() {
            static COUNT: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            let n = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Skip the initial setup allocs (KV cache init). Two possible
            // filters: `CANDLE_TRACE_ZEROS_START=N` sets an offset, and
            // `CANDLE_TRACE_ZEROS_MAXB=N` caps by byte size.  Defaults keep
            // only small allocs after the first 200 calls.
            let start = std::env::var("CANDLE_TRACE_ZEROS_START")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(200u64);
            let maxb = std::env::var("CANDLE_TRACE_ZEROS_MAXB")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(1_000_000usize);
            if n >= start && len * std::mem::size_of::<T>() <= maxb {
                let bt = std::backtrace::Backtrace::force_capture();
                let bt_str = format!("{bt}");
                // Keep the first 8 frames that mention candle_transformers
                // or candle_core model code.
                let mut frames = Vec::<String>::new();
                for line in bt_str.lines() {
                    if frames.len() >= 8 { break; }
                    let l = line.trim();
                    if l.contains("candle_transformers::")
                        || l.contains("candle_core::kv_cache")
                        || l.contains("candle_nn::kv_cache")
                        || l.contains("quantized_blocks")
                        || l.contains("quantized_gemma4")
                        || l.contains("quantized_qwen")
                        || (l.starts_with("at ") && (l.contains(".rs:") || l.contains("/candle-")))
                    {
                        frames.push(l.to_string());
                    }
                }
                eprintln!(
                    "[alloc_zeros #{n}] bytes={}\n  {}",
                    len * std::mem::size_of::<T>(),
                    frames.join("\n  ")
                );
            }
        }
        self.stream.alloc_zeros::<T>(len).w()
    }

    pub fn clone_htod<T: hipdarc::driver::DeviceRepr>(
        &self,
        src: &[T],
    ) -> Result<hipdarc::driver::HipSlice<T>> {
        self.stream.clone_htod(src).w()
    }

    pub fn clone_dtoh<T: hipdarc::driver::DeviceRepr>(
        &self,
        src: &hipdarc::driver::HipSlice<T>,
    ) -> Result<Vec<T>> {
        self.stream.clone_dtoh(src).w()
    }

    pub fn clone_dtod<T: hipdarc::driver::DeviceRepr>(
        &self,
        src: &hipdarc::driver::HipSlice<T>,
    ) -> Result<hipdarc::driver::HipSlice<T>> {
        self.stream.clone_dtod(src).w()
    }
}

// -- Kernel loading ----------------------------------------------------------

impl HipDevice {
    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn rocblas_handle(&self) -> Result<&RocBlas> {
        if let Some(blas) = self.blas.get() {
            return Ok(blas);
        }
        match RocBlas::new(&self.blas_stream) {
            Ok(blas) => {
                let _ = self.blas.set(blas);
                Ok(self.blas.get().unwrap())
            }
            Err(e) => Err(HipError::InternalError(
                Box::leak(format!("rocBLAS init failed: {e:?}. For gfx906: install patched rocBLAS with Tensile gfx906 kernels.").into_boxed_str())
            ).into())
        }
    }

    /// Load a kernel function from a precompiled HSACO module, caching the
    /// module for future lookups.
    pub fn get_or_load_func(&self, fn_name: &str, mdl: &kernels::Module) -> Result<HipFunc> {
        // Fast path: module already loaded.
        {
            let ms = self.modules.read().map_err(|_| HipError::InternalError("module store lock poisoned"))?;
            if let Some(loaded) = ms.mdls[mdl.index()].as_ref() {
                let func = loaded.load_function(fn_name).map_err(|e| HipError::Load {
                    driver: e,
                    module_name: fn_name.to_string(),
                })?;
                return Ok(HipFunc::new(func, self.stream.clone()));
            }
        }
        // Slow path: load HSACO and cache.
        let mut ms = self.modules.write().map_err(|_| HipError::InternalError("module store lock poisoned"))?;
        // Double-check after acquiring write lock.
        if let Some(loaded) = ms.mdls[mdl.index()].as_ref() {
            let func = loaded.load_function(fn_name).map_err(|e| HipError::Load {
                driver: e,
                module_name: fn_name.to_string(),
            })?;
            return Ok(HipFunc::new(func, self.stream.clone()));
        }
        let hip_module = HipModule::load_data(mdl.hsaco()).map_err(|e| HipError::Load {
            driver: e,
            module_name: fn_name.to_string(),
        })?;
        let hip_module = Arc::new(hip_module);
        ms.mdls[mdl.index()] = Some(hip_module.clone());
        let func = hip_module
            .load_function(fn_name)
            .map_err(|e| HipError::Load {
                driver: e,
                module_name: fn_name.to_string(),
            })?;
        Ok(HipFunc::new(func, self.stream.clone()))
    }

    /// Load a custom kernel from raw HSACO bytes (user-provided).
    pub fn get_or_load_custom_func(
        &self,
        fn_name: &str,
        module_name: &str,
        hsaco: &[u8],
    ) -> Result<HipFunc> {
        {
            let ms = self.custom_modules.read().map_err(|_| HipError::InternalError("custom module lock poisoned"))?;
            if let Some(mdl) = ms.get(module_name) {
                let func = mdl.load_function(fn_name).w()?;
                return Ok(HipFunc::new(func, self.stream.clone()));
            }
        }
        let mut ms = self.custom_modules.write().map_err(|_| HipError::InternalError("custom module lock poisoned"))?;
        let hip_module = Arc::new(HipModule::load_data(hsaco).w()?);
        ms.insert(module_name.to_string(), hip_module.clone());
        let func = hip_module.load_function(fn_name).w()?;
        Ok(HipFunc::new(func, self.stream.clone()))
    }
}

// -- BackendDevice implementation --------------------------------------------

impl BackendDevice for HipDevice {
    type Storage = HipStorage;

    fn new(ordinal: usize) -> Result<Self> {
        use rand::SeedableRng;
        let raw = RawHipDevice::new(ordinal).w()?;
        let stream = Arc::new(HipStream::new(&raw).w()?);
        let blas = Arc::new(std::sync::OnceLock::new());
        let blas_stream = stream.clone();
        let module_store = ModuleStore {
            mdls: [const { None }; kernels::ALL_IDS.len()],
        };
        let seed: u64 = 299792458;
        Ok(Self {
            id: DeviceId::new(),
            ordinal,
            stream,
            blas,
            blas_stream,
            cpu_rng: Arc::new(Mutex::new(rand::rngs::StdRng::seed_from_u64(seed))),
            modules: Arc::new(RwLock::new(module_store)),
            custom_modules: Arc::new(RwLock::new(HashMap::new())),
            seed_value: Arc::new(RwLock::new(seed)),
        })
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        use rand::SeedableRng;
        let mut rng = self.cpu_rng.lock().map_err(|_| {
            crate::Error::Msg("rng mutex poisoned".to_string())
        })?;
        *rng = rand::rngs::StdRng::seed_from_u64(seed);
        *self.seed_value.write().map_err(|_| {
            crate::Error::Msg("seed_value rwlock poisoned".to_string())
        })? = seed;
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        let seed = self.seed_value.read().map_err(|_| {
            crate::Error::Msg("seed_value rwlock poisoned".to_string())
        })?;
        Ok(*seed)
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Hip {
            gpu_id: self.ordinal,
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<HipStorage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => HipStorageSlice::U8(self.alloc_zeros::<u8>(elem_count)?),
            DType::U32 => HipStorageSlice::U32(self.alloc_zeros::<u32>(elem_count)?),
            DType::I16 => HipStorageSlice::I16(self.alloc_zeros::<i16>(elem_count)?),
            DType::I32 => HipStorageSlice::I32(self.alloc_zeros::<i32>(elem_count)?),
            DType::I64 => HipStorageSlice::I64(self.alloc_zeros::<i64>(elem_count)?),
            DType::BF16 => HipStorageSlice::BF16(self.alloc_zeros::<bf16>(elem_count)?),
            DType::F16 => HipStorageSlice::F16(self.alloc_zeros::<f16>(elem_count)?),
            DType::F32 => HipStorageSlice::F32(self.alloc_zeros::<f32>(elem_count)?),
            DType::F64 => HipStorageSlice::F64(self.alloc_zeros::<f64>(elem_count)?),
            DType::F8E4M3 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(
                    HipError::UnsupportedDtype { dtype, op: "zeros" }.into()
                )
            }
        };
        Ok(HipStorage { slice, device: self.clone() })
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<HipStorage> {
        use rand::distr::Distribution;
        let elem_count = shape.elem_count();
        let mut rng = self.cpu_rng.lock().map_err(|_| crate::Error::Msg("rng mutex poisoned".into()))?;
        let dist = rand::distr::Uniform::new(lo, up).map_err(|e| crate::Error::Msg(e.to_string()))?;
        let slice = match dtype {
            DType::F32 => {
                let data: Vec<f32> = (0..elem_count).map(|_| dist.sample(&mut *rng) as f32).collect();
                HipStorageSlice::F32(self.clone_htod(&data)?)
            }
            DType::F64 => {
                let data: Vec<f64> = (0..elem_count).map(|_| dist.sample(&mut *rng)).collect();
                HipStorageSlice::F64(self.clone_htod(&data)?)
            }
            _ => {
                return Err(HipError::UnsupportedDtype { dtype, op: "rand_uniform" }.into())
            }
        };
        Ok(HipStorage { slice, device: self.clone() })
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<HipStorage> {
        use rand::distr::Distribution;
        let elem_count = shape.elem_count();
        let mut rng = self.cpu_rng.lock().map_err(|_| crate::Error::Msg("rng mutex poisoned".into()))?;
        let dist = rand_distr::Normal::new(mean, std).map_err(|e| crate::Error::Msg(e.to_string()))?;
        let slice = match dtype {
            DType::F32 => {
                let data: Vec<f32> = (0..elem_count).map(|_| dist.sample(&mut *rng) as f32).collect();
                HipStorageSlice::F32(self.clone_htod(&data)?)
            }
            DType::F64 => {
                let data: Vec<f64> = (0..elem_count).map(|_| dist.sample(&mut *rng)).collect();
                HipStorageSlice::F64(self.clone_htod(&data)?)
            }
            _ => {
                return Err(HipError::UnsupportedDtype { dtype, op: "rand_normal" }.into())
            }
        };
        Ok(HipStorage { slice, device: self.clone() })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<HipStorage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => HipStorageSlice::U8(self.alloc::<u8>(elem_count)?),
            DType::U32 => HipStorageSlice::U32(self.alloc::<u32>(elem_count)?),
            DType::I16 => HipStorageSlice::I16(self.alloc::<i16>(elem_count)?),
            DType::I32 => HipStorageSlice::I32(self.alloc::<i32>(elem_count)?),
            DType::I64 => HipStorageSlice::I64(self.alloc::<i64>(elem_count)?),
            DType::BF16 => HipStorageSlice::BF16(self.alloc::<bf16>(elem_count)?),
            DType::F16 => HipStorageSlice::F16(self.alloc::<f16>(elem_count)?),
            DType::F32 => HipStorageSlice::F32(self.alloc::<f32>(elem_count)?),
            DType::F64 => HipStorageSlice::F64(self.alloc::<f64>(elem_count)?),
            DType::F8E4M3 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(
                    HipError::UnsupportedDtype { dtype, op: "alloc_uninit" }.into()
                )
            }
        };
        Ok(HipStorage { slice, device: self.clone() })
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> Result<HipStorage> {
        let cpu_storage = T::to_cpu_storage(data);
        BackendDevice::storage_from_cpu_storage(self, &cpu_storage)
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<HipStorage> {
        let slice = match storage {
            CpuStorage::U8(data) => HipStorageSlice::U8(self.clone_htod(data)?),
            CpuStorage::U32(data) => HipStorageSlice::U32(self.clone_htod(data)?),
            CpuStorage::I16(data) => HipStorageSlice::I16(self.clone_htod(data)?),
            CpuStorage::I32(data) => HipStorageSlice::I32(self.clone_htod(data)?),
            CpuStorage::I64(data) => HipStorageSlice::I64(self.clone_htod(data)?),
            CpuStorage::BF16(data) => HipStorageSlice::BF16(self.clone_htod(data)?),
            CpuStorage::F16(data) => HipStorageSlice::F16(self.clone_htod(data)?),
            CpuStorage::F32(data) => HipStorageSlice::F32(self.clone_htod(data)?),
            CpuStorage::F64(data) => HipStorageSlice::F64(self.clone_htod(data)?),
            CpuStorage::F8E4M3(_)
            | CpuStorage::F6E2M3(_)
            | CpuStorage::F6E3M2(_)
            | CpuStorage::F4(_)
            | CpuStorage::F8E8M0(_) => {
                return Err(HipError::UnsupportedDtype {
                    dtype: storage.dtype(),
                    op: "storage_from_cpu_storage",
                }
                .into())
            }
        };
        Ok(HipStorage { slice, device: self.clone() })
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<HipStorage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().w()
    }
}
