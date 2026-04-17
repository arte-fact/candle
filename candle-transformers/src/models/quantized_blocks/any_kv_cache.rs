//! AnyKvCache — enum wrapping F32 `KvCache` and Q8 `QKvCache` so model
//! code can keep a uniform cache type while the env flag picks the
//! underlying storage. `quantized_gemma4` used to store
//! `Vec<Option<KvCache>>` directly; it now uses `Vec<Option<AnyKvCache>>`.
//!
//! Q8 is only available with the `hip` feature. On non-HIP builds the
//! `Q8` arm simply isn't constructible — no ifdef needed in callers.

use candle::{Result, Tensor};
use candle_nn::kv_cache::KvCache;

#[cfg(feature = "hip")]
use candle::hip_backend::q8_kv_cache::QKvCache;
#[cfg(feature = "hip")]
use candle::hip_backend::turbo_kv_cache::{QTurboKvCache, TbVariant};

#[derive(Debug)]
pub enum AnyKvCache {
    F32(KvCache),
    #[cfg(feature = "hip")]
    Q8(QKvCache),
    #[cfg(feature = "hip")]
    Tb(QTurboKvCache),
}

impl AnyKvCache {
    pub fn is_q8(&self) -> bool {
        #[cfg(feature = "hip")]
        {
            matches!(self, AnyKvCache::Q8(_))
        }
        #[cfg(not(feature = "hip"))]
        {
            false
        }
    }

    pub fn is_tb(&self) -> bool {
        #[cfg(feature = "hip")]
        {
            matches!(self, AnyKvCache::Tb(_))
        }
        #[cfg(not(feature = "hip"))]
        {
            false
        }
    }

    /// True when the cache stores compressed (Q8 or Tb) KV — i.e. the
    /// padded-slab fast path that expects raw F32/F16 can't run.
    pub fn is_compressed(&self) -> bool {
        self.is_q8() || self.is_tb()
    }

    #[cfg(feature = "hip")]
    pub fn as_tb(&self) -> Option<&QTurboKvCache> {
        match self {
            AnyKvCache::Tb(c) => Some(c),
            _ => None,
        }
    }

    #[cfg(feature = "hip")]
    pub fn as_tb_mut(&mut self) -> Option<&mut QTurboKvCache> {
        match self {
            AnyKvCache::Tb(c) => Some(c),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<&KvCache> {
        match self {
            AnyKvCache::F32(c) => Some(c),
            #[cfg(feature = "hip")]
            AnyKvCache::Q8(_) => None,
            #[cfg(feature = "hip")]
            AnyKvCache::Tb(_) => None,
        }
    }

    pub fn as_f32_mut(&mut self) -> Option<&mut KvCache> {
        match self {
            AnyKvCache::F32(c) => Some(c),
            #[cfg(feature = "hip")]
            AnyKvCache::Q8(_) => None,
            #[cfg(feature = "hip")]
            AnyKvCache::Tb(_) => None,
        }
    }

    #[cfg(feature = "hip")]
    pub fn as_q8(&self) -> Option<&QKvCache> {
        match self {
            AnyKvCache::Q8(c) => Some(c),
            _ => None,
        }
    }

    #[cfg(feature = "hip")]
    pub fn as_q8_mut(&mut self) -> Option<&mut QKvCache> {
        match self {
            AnyKvCache::Q8(c) => Some(c),
            _ => None,
        }
    }

    pub fn reset(&mut self) {
        match self {
            AnyKvCache::F32(c) => {
                c.k_cache_mut().reset();
                c.v_cache_mut().reset();
            }
            #[cfg(feature = "hip")]
            AnyKvCache::Q8(c) => c.reset(),
            #[cfg(feature = "hip")]
            AnyKvCache::Tb(c) => c.reset(),
        }
    }

    pub fn current_seq_len(&self) -> usize {
        match self {
            AnyKvCache::F32(c) => c.k_cache().current_seq_len(),
            #[cfg(feature = "hip")]
            AnyKvCache::Q8(c) => c.current_seq_len(),
            #[cfg(feature = "hip")]
            AnyKvCache::Tb(c) => c.current_seq_len(),
        }
    }

    pub fn max_seq_len(&self) -> usize {
        match self {
            AnyKvCache::F32(c) => c.k_cache().max_seq_len(),
            #[cfg(feature = "hip")]
            AnyKvCache::Q8(c) => c.dims().2,
            #[cfg(feature = "hip")]
            AnyKvCache::Tb(c) => c.dims().2,
        }
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        match self {
            AnyKvCache::F32(c) => {
                let _ = c.append(k, v)?;
                Ok(())
            }
            #[cfg(feature = "hip")]
            AnyKvCache::Q8(c) => c.append(k, v),
            #[cfg(feature = "hip")]
            AnyKvCache::Tb(c) => c.append(k, v),
        }
    }

    /// Full-range F32 view of the K slab in canonical `(B, H_kv, T_cur, D)`
    /// layout. F32 arm returns `k_cache().all_data()` narrow'd to `t_cur`
    /// (or `None` if the cache is empty). Q8 arm dequantises on demand
    /// into a fresh F32 tensor — caller is expected to drop it right
    /// after the attention call.
    ///
    /// Only valid when `k_is_transposed == false` on the F32 arm.
    pub fn canonical_k_f32(&self) -> Result<Option<Tensor>> {
        match self {
            AnyKvCache::F32(c) => {
                if c.k_is_transposed() {
                    candle::bail!("AnyKvCache::canonical_k_f32: F32 cache is K-transposed, cannot return canonical view")
                }
                c.k()
            }
            #[cfg(feature = "hip")]
            AnyKvCache::Q8(c) => {
                if c.current_seq_len() == 0 {
                    Ok(None)
                } else {
                    Ok(Some(c.dequantize_k()?))
                }
            }
            #[cfg(feature = "hip")]
            AnyKvCache::Tb(c) => {
                if c.current_seq_len() == 0 {
                    Ok(None)
                } else {
                    Ok(Some(c.dequantize_k()?))
                }
            }
        }
    }

    pub fn canonical_v_f32(&self) -> Result<Option<Tensor>> {
        match self {
            AnyKvCache::F32(c) => c.v(),
            #[cfg(feature = "hip")]
            AnyKvCache::Q8(c) => {
                if c.current_seq_len() == 0 {
                    Ok(None)
                } else {
                    Ok(Some(c.dequantize_v()?))
                }
            }
            #[cfg(feature = "hip")]
            AnyKvCache::Tb(c) => {
                if c.current_seq_len() == 0 {
                    Ok(None)
                } else {
                    Ok(Some(c.dequantize_v()?))
                }
            }
        }
    }
}
