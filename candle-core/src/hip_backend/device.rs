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
    pub(crate) blas: Arc<RocBlas>,
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

    pub fn rocblas_handle(&self) -> Arc<RocBlas> {
        self.blas.clone()
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
        let blas = Arc::new(RocBlas::new(&stream).w()?);
        let module_store = ModuleStore {
            mdls: [const { None }; kernels::ALL_IDS.len()],
        };
        let seed: u64 = 299792458;
        Ok(Self {
            id: DeviceId::new(),
            ordinal,
            stream,
            blas,
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
