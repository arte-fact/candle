//! Safe wrapper around hiprand for GPU random number generation.

use crate::driver::{HipSlice, HipStream};
use crate::error::{check_hiprand, HiprandError};
use crate::sys;
use std::sync::Arc;

/// A hiprand pseudo-random number generator bound to a stream.
pub struct HipRng {
    generator: sys::hiprandGenerator_t,
    // Keep stream alive to prevent use-after-free in hiprand operations.
    _stream: Arc<HipStream>,
}

unsafe impl Send for HipRng {}

impl HipRng {
    /// Create a new XORWOW PRNG on the given stream with the given seed.
    pub fn new(seed: u64, stream: &Arc<HipStream>) -> Result<Self, HiprandError> {
        stream.device().set_current().map_err(|_| {
            HiprandError::Status(sys::hiprandStatus_t::HIPRAND_STATUS_INTERNAL_ERROR)
        })?;
        let mut generator = std::ptr::null_mut();
        unsafe {
            check_hiprand(sys::hiprandCreateGenerator(
                &mut generator,
                sys::hiprandRngType_t::HIPRAND_RNG_PSEUDO_XORWOW,
            ))?;
            check_hiprand(sys::hiprandSetStream(generator, stream.raw()))?;
            check_hiprand(sys::hiprandSetPseudoRandomGeneratorSeed(generator, seed))?;
        }
        Ok(Self {
            generator,
            _stream: stream.clone(),
        })
    }

    /// Fill a device buffer with uniform random f32 values in [0, 1).
    pub fn fill_with_uniform(&mut self, data: &mut HipSlice<f32>) -> Result<(), HiprandError> {
        if data.is_empty() {
            return Ok(());
        }
        unsafe {
            check_hiprand(sys::hiprandGenerateUniform(
                self.generator,
                data.ptr as *mut f32,
                data.len,
            ))
        }
    }

    /// Fill a device buffer with uniform random f64 values in [0, 1).
    pub fn fill_with_uniform_f64(&mut self, data: &mut HipSlice<f64>) -> Result<(), HiprandError> {
        if data.is_empty() {
            return Ok(());
        }
        unsafe {
            check_hiprand(sys::hiprandGenerateUniformDouble(
                self.generator,
                data.ptr as *mut f64,
                data.len,
            ))
        }
    }

    /// Fill a device buffer with normally distributed f32 values.
    ///
    /// Note: hiprand requires `data.len()` to be even for normal distributions.
    pub fn fill_with_normal(
        &mut self,
        data: &mut HipSlice<f32>,
        mean: f32,
        stddev: f32,
    ) -> Result<(), HiprandError> {
        if data.is_empty() {
            return Ok(());
        }
        assert!(data.len().is_multiple_of(2), "hiprandGenerateNormal requires even element count");
        unsafe {
            check_hiprand(sys::hiprandGenerateNormal(
                self.generator,
                data.ptr as *mut f32,
                data.len,
                mean,
                stddev,
            ))
        }
    }

    /// Fill a device buffer with normally distributed f64 values.
    pub fn fill_with_normal_f64(
        &mut self,
        data: &mut HipSlice<f64>,
        mean: f64,
        stddev: f64,
    ) -> Result<(), HiprandError> {
        if data.is_empty() {
            return Ok(());
        }
        unsafe {
            check_hiprand(sys::hiprandGenerateNormalDouble(
                self.generator,
                data.ptr as *mut f64,
                data.len,
                mean,
                stddev,
            ))
        }
    }
}

impl Drop for HipRng {
    fn drop(&mut self) {
        if !self.generator.is_null() {
            unsafe {
                let _ = sys::hiprandDestroyGenerator(self.generator);
            }
        }
    }
}

impl std::fmt::Debug for HipRng {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipRng").finish()
    }
}
