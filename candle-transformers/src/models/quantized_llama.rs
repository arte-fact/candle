//! Quantized llama model implementation.
//!
//! This provides a quantized implementation of the llama language model architecture.
//! The model implements parameter efficient quantization for reduced memory usage
//! while maintaining model quality.
//!
//! Key characteristics:
//! - Transformer decoder architecture
//! - Support for 2/3/4/8-bit quantization
//! - Optimized memory usage through quantization
//! - Configurable model sizes and parameter counts
//!
//! - 💻 [GH Link](https://github.com/facebookresearch/llama)
//! - 📝 [Paper](https://arxiv.org/abs/2302.13971)
//!
//! ![](https://raw.githubusercontent.com/huggingface/candle/main/candle-examples/examples/quantized/assets/aoc.gif)
//!

use std::collections::HashMap;

use crate::quantized_nn::RmsNorm;
use candle::quantized::QTensor;
use candle::quantized::{ggml_file, gguf_file};
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};

pub const MAX_SEQ_LEN: usize = 4096;

// QMatMul wrapper adding some tracing.
#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle::quantized::QMatMul::from_qtensor(qtensor)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        self.feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

/// Fused Q4_0 FFN decode path. Replaces the rmsnorm + 3 MMVQs + silu_mul +
/// quantize + residual chain (7 launches) with a 5-launch pipeline:
///   1. rmsnorm_q8_fused(x, norm_w) → (x_q8_buf, x_f32)
///   2. matmul_gate_up_preq8(w_gate, w_up, x_q8) → (gate, up) [1 kernel]
///   3. silu(gate) * up → hidden
///   4. quantize_q8_1(hidden) → hidden_q8
///   5. matmul_down_residual_preq8(w_down, hidden_q8, residual=x) → out [1 kernel]
///
/// Precondition: `x` is (B=1, 1, hidden) or (1, hidden) f32 HIP, all three
/// weights are Q4_0. Returns (B=1, 1, hidden) f32 tensor.
#[cfg(feature = "hip")]
fn fused_q4_0_ffn_decode(
    x: &Tensor,
    ffn_norm: &RmsNorm,
    w_gate: &std::sync::Arc<candle::quantized::QTensor>,
    w_up: &std::sync::Arc<candle::quantized::QTensor>,
    w_down: &std::sync::Arc<candle::quantized::QTensor>,
) -> Result<Tensor> {
    use candle::quantized::hip as qhip;
    use candle::hip_backend;
    let dev = match x.device() {
        Device::Hip(d) => d.clone(),
        _ => candle::bail!("fused_q4_0_ffn_decode: x must be HIP"),
    };
    let x = if x.is_contiguous() { x.clone() } else { x.contiguous()? };
    let (_intermediate_dim, hidden_dim) = w_gate.shape().dims2()?;
    // Step 1: rmsnorm + quantize_q8_1(x_norm) → (x_q8_buf, _).
    let (x_q8_buf, _) = hip_backend::rmsnorm_q8_fused(
        &x, ffn_norm.weight(), ffn_norm.eps_f32() as f64, false,
    )?;
    let x_q8_view = x_q8_buf.slice(0..x_q8_buf.len());

    // Step 2: fused gate + up MMVQ.
    let rhs_shape: Vec<usize> = vec![1, 1, hidden_dim];
    let ((gate_storage, gate_shape), (up_storage, up_shape)) =
        w_gate.matmul_gate_up_preq8(w_up, &x_q8_view, &rhs_shape)?;
    let gate = Tensor::from_storage(
        candle::Storage::Hip(gate_storage),
        gate_shape,
        candle::op::BackpropOp::none(),
        false,
    );
    let up = Tensor::from_storage(
        candle::Storage::Hip(up_storage),
        up_shape,
        candle::op::BackpropOp::none(),
        false,
    );
    // Step 3: silu(gate) * up
    let hidden = (candle_nn::ops::silu(&gate)? * up)?;
    // Step 4: quantize hidden → hidden_q8
    let hidden_c = if hidden.is_contiguous() { hidden.clone() } else { hidden.contiguous()? };
    let (h_st, h_l) = hidden_c.storage_and_layout();
    let h_hip = match &*h_st {
        candle::Storage::Hip(s) => s,
        _ => candle::bail!("fused_q4_0_ffn_decode: hidden not HIP"),
    };
    let h_slice = h_hip.as_hip_slice::<f32>()?;
    let h_view = {
        let (lo, hi) = h_l.contiguous_offsets()
            .ok_or_else(|| candle::Error::Msg("non-contig hidden".into()))?;
        h_slice.slice(lo..hi)
    };
    let k = hidden_dim; // columns of W_down
    let ky = 1; // b*m
    let kx_padded = qhip::pad(k, qhip::MATRIX_ROW_PADDING);
    let num_blocks = (kx_padded + qhip::CUDA_QUANTIZE_BLOCK_SIZE - 1) / qhip::CUDA_QUANTIZE_BLOCK_SIZE;
    let q8_bytes = ky * (kx_padded / 32) * 36;
    let mut hidden_q8_buf = unsafe { dev.alloc::<u8>(q8_bytes)? };
    qhip::quantize_q8_1(&h_view, &mut hidden_q8_buf, k, ky, &dev)?;
    drop(h_st);
    let hidden_q8_view = hidden_q8_buf.slice(0..hidden_q8_buf.len());

    // Step 5: fused down + residual add (residual = original x, flat).
    let (x_st, x_l) = x.storage_and_layout();
    let x_hip = match &*x_st {
        candle::Storage::Hip(s) => s,
        _ => candle::bail!("fused_q4_0_ffn_decode: x not HIP"),
    };
    let x_slice = x_hip.as_hip_slice::<f32>()?;
    let x_view = {
        let (lo, hi) = x_l.contiguous_offsets()
            .ok_or_else(|| candle::Error::Msg("non-contig x".into()))?;
        x_slice.slice(lo..hi)
    };
    let (out_storage, _out_shape) =
        w_down.matmul_down_residual_preq8(&hidden_q8_view, &x_view, &rhs_shape)?;
    drop(x_st);
    Ok(Tensor::from_storage(
        candle::Storage::Hip(out_storage),
        candle::Shape::from_dims(&[1, 1, hidden_dim]),
        candle::op::BackpropOp::none(),
        false,
    ))
}

#[cfg(feature = "hip")]
impl Mlp {
    /// Extract the three Q4_0 weight tensors if ALL three MMVQs are Q4_0.
    /// Returns `(gate, up, down)`. Shapes: gate/up = (intermediate, hidden),
    /// down = (hidden, intermediate).
    fn as_q4_0_trio(&self) -> Option<(
        std::sync::Arc<candle::quantized::QTensor>,
        std::sync::Arc<candle::quantized::QTensor>,
        std::sync::Arc<candle::quantized::QTensor>,
    )> {
        use candle::quantized::{GgmlDType, QMatMul};
        let w1 = match &self.feed_forward_w1.inner { QMatMul::QTensor(t) => t.clone(), _ => return None };
        let w2 = match &self.feed_forward_w2.inner { QMatMul::QTensor(t) => t.clone(), _ => return None };
        let w3 = match &self.feed_forward_w3.inner { QMatMul::QTensor(t) => t.clone(), _ => return None };
        if w1.dtype() != GgmlDType::Q4_0 || w2.dtype() != GgmlDType::Q4_0 || w3.dtype() != GgmlDType::Q4_0 {
            return None;
        }
        Some((w1, w3, w2))
    }
}

#[derive(Debug, Clone)]
enum MlpOrMoe {
    Mlp(Mlp),
    MoE {
        n_expert_used: usize,
        feed_forward_gate_inp: QMatMul,
        experts: Vec<Mlp>,
    },
}

impl Module for MlpOrMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::MoE {
                feed_forward_gate_inp,
                experts,
                n_expert_used,
            } => {
                let (b_size, seq_len, hidden_dim) = xs.dims3()?;
                let xs = xs.reshape(((), hidden_dim))?;
                let router_logits = feed_forward_gate_inp.forward(&xs)?;
                let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

                // In order to extract topk, we extract the data from the tensor and manipulate it
                // directly. Maybe we will want to use some custom ops instead at some point.
                let routing_weights = routing_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;

                // routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
                // top_x contains the row indexes to evaluate for each expert.
                let mut top_x = vec![vec![]; experts.len()];
                let mut selected_rws = vec![vec![]; experts.len()];
                for (row_idx, rw) in routing_weights.iter().enumerate() {
                    let mut dst = (0..rw.len() as u32).collect::<Vec<u32>>();
                    dst.sort_by(|&i, &j| rw[j as usize].total_cmp(&rw[i as usize]));
                    let mut sum_routing_weights = 0f32;
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        sum_routing_weights += routing_weight;
                        top_x[expert_idx].push(row_idx as u32);
                    }
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        selected_rws[expert_idx].push(routing_weight / sum_routing_weights)
                    }
                }

                // routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
                // expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

                let mut ys = xs.zeros_like()?;
                for (expert_idx, expert_layer) in experts.iter().enumerate() {
                    let top_x = &top_x[expert_idx];
                    if top_x.is_empty() {
                        continue;
                    }
                    let top_x = Tensor::new(top_x.as_slice(), xs.device())?;
                    let selected_rws =
                        Tensor::new(selected_rws[expert_idx].as_slice(), xs.device())?
                            .reshape(((), 1))?;
                    // Index the correct hidden states and compute the expert hidden state for
                    // the current expert. We need to make sure to multiply the output hidden
                    // states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                    let current_state = xs.index_select(&top_x, 0)?.reshape(((), hidden_dim))?;
                    // current_hidden_states = expert_layer(current_state, routing_weights[top_x_list, idx_list, None])
                    let current_hidden_states = expert_layer.forward(&current_state)?;
                    let current_hidden_states =
                        current_hidden_states.broadcast_mul(&selected_rws)?;
                    ys = ys.index_add(&top_x, &current_hidden_states, 0)?;
                }

                let ys = ys.reshape((b_size, seq_len, hidden_dim))?;
                Ok(ys)
            }
            Self::Mlp(mlp) => mlp.forward(xs),
        }
    }
}

/// G1: Lightweight KV cache with pre-allocated buffers and O(1) append.
/// Avoids Tensor::cat (alloc + copy every step) and KvCache::slice_set (expensive copy2d).
/// Uses `Tensor::slice_set` only in the simple contiguous-on-last-dim case,
/// which degenerates to a single linear memcpy.
#[derive(Debug, Clone)]
struct KvCacheDirect {
    /// K stored as (B, n_kv_h, D, max_T) — seq on last dim.
    k_buf: Tensor,
    /// V stored as (B, n_kv_h, D, max_T) — seq on last dim (transposed storage).
    v_buf: Tensor,
    current_t: usize,
    max_t: usize,
}

impl KvCacheDirect {
    fn new(k_shape: &[usize], v_shape: &[usize], max_t: usize, dtype: DType, device: &Device) -> Result<Self> {
        // Both K and V: (B, n_kv_h, D, max_T) — seq on last dim.
        let mut ks = k_shape.to_vec();
        ks[3] = max_t;
        let k_buf = Tensor::zeros(ks, dtype, device)?;
        let mut vs = v_shape.to_vec();
        vs[3] = max_t;
        let v_buf = Tensor::zeros(vs, dtype, device)?;
        Ok(Self { k_buf, v_buf, current_t: 0, max_t })
    }

    fn reset(&mut self) {
        self.current_t = 0;
        // Don't drop buffers — reuse them. Just reset the position.
    }

    fn append(&mut self, k_t_new: &Tensor, v_new: &Tensor) -> Result<()> {
        let seq_new = k_t_new.dim(3)?; // K is (B, H, D, seq_new)

        // Grow if needed
        if self.current_t + seq_new > self.max_t {
            let grow = std::cmp::max(self.max_t, seq_new);
            let mut ks = self.k_buf.dims().to_vec();
            ks[3] = grow;
            let k_ext = Tensor::zeros(ks, self.k_buf.dtype(), self.k_buf.device())?;
            self.k_buf = Tensor::cat(&[&self.k_buf, &k_ext], 3)?;
            let mut vs = self.v_buf.dims().to_vec();
            vs[3] = grow;
            let v_ext = Tensor::zeros(vs, self.v_buf.dtype(), self.v_buf.device())?;
            self.v_buf = Tensor::cat(&[&self.v_buf, &v_ext], 3)?;
            self.max_t += grow;
        }

        // K: slice_set on dim=3 (last dim) — this IS contiguous.
        self.k_buf.slice_set(k_t_new, 3, self.current_t)?;
        // V: slice_set on dim=3 (last dim) — contiguous write.
        self.v_buf.slice_set(v_new, 3, self.current_t)?;
        self.current_t += seq_new;
        Ok(())
    }

    fn k(&self) -> Result<Tensor> {
        // narrow on last dim = contiguous
        self.k_buf.narrow(3, 0, self.current_t)
    }

    fn v(&self) -> Result<Tensor> {
        // narrow on last dim = contiguous ✓
        self.v_buf.narrow(3, 0, self.current_t)
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    mlp_or_moe: MlpOrMoe,
    ffn_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    /// G1: KvCache with K pre-transposed. K stored as (B,H,D,T) with dim=3,
    /// V stored as (B,H,T,D) with dim=2 (natural). V access during decode
    /// attention is coalesced across D (stride[3]=1, stride[2]=D).
    kv_cache: Option<candle_nn::kv_cache::KvCache>,
    device: Device,
    span_attn: tracing::Span,
    span_rot: tracing::Span,
    span_mlp: tracing::Span,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

impl LayerWeights {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        // The call to contiguous below is only necessary when processing the prompt.
        // When the seq_len is 1 in the inference loop, this is a no-op.
        candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, _n_embd) = x.dims3()?;
        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            // This call to contiguous ensures that the fast kernel can be called below. It's
            // actually a no-op except when processing the initial prompt so has no significant
            // impact on performance.
            .contiguous()?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        // G1: KvCache with K pre-transposed. V kept in natural (B,H,T,D)
        // layout (dim_v=2), giving coalesced decode access across D.
        let kv_cache = self.kv_cache.get_or_insert_with(|| {
            candle_nn::kv_cache::KvCache::new_k_transposed(2, 4096)
        });
        if index_pos == 0 {
            kv_cache.k_cache_mut().reset();
            kv_cache.v_cache_mut().reset();
        }
        let (_k_ignore, _v_ignore) = kv_cache.append(&k, &v)?;
        // llamacpp-turbo trick: pad the attention sequence length to a
        // multiple of 256 so kernel args stay identical for 256 consecutive
        // decode tokens. Unwritten cache positions are zero (Cache::append
        // zero-initialises the buffer), and we pass a mask that marks
        // positions [T_cur, L_k_padded) as -inf so softmax drops them.
        let t_cur = kv_cache.k_cache().current_seq_len();
        let max_t = kv_cache.k_cache().max_seq_len();
        // Pad attention sequence length to a multiple of this so kernel
        // args stay constant for many consecutive decode tokens, enabling
        // G2 replay to skip Rust-side dispatch overhead entirely. Matches
        // llamacpp-turbo's GGML_KQ_MASK_PAD. Only on when G2 replay is on
        // — padding without replay is net-negative (extra attention work
        // without the CPU-overhead compensation).
        let pad_default: usize = if std::env::var("CANDLE_G2_REPLAY").is_ok() { 256 } else { 1 };
        let pad_t = std::env::var("CANDLE_NKV_PAD")
            .ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(pad_default);
        let l_k_padded = if pad_t > 1 {
            (((t_cur + pad_t - 1) / pad_t) * pad_t).min(max_t)
        } else {
            t_cur
        };
        let (k_t, v_attn, l_k) = if l_k_padded > t_cur && seq_len == 1 {
            // Decode path with padding: read full-padded slab from the
            // cache buffer (positions beyond T_cur are zero).
            let k_full = kv_cache.k_cache().all_data().as_ref()
                .ok_or_else(|| candle::Error::Msg("k cache empty".into()))?;
            let v_full = kv_cache.v_cache().all_data().as_ref()
                .ok_or_else(|| candle::Error::Msg("v cache empty".into()))?;
            // K is stored (B, H, D, max_T); narrow dim=3 to L_k_padded.
            let k_t = k_full.narrow(3, 0, l_k_padded)?;
            // V is stored (B, H, max_T, D); narrow dim=2 to L_k_padded.
            let v_attn = v_full.narrow(2, 0, l_k_padded)?;
            (k_t, v_attn, l_k_padded)
        } else {
            let k_t = kv_cache.k()?.ok_or_else(|| candle::Error::Msg("kv cache empty".into()))?;
            let v = kv_cache.v()?.ok_or_else(|| candle::Error::Msg("kv cache empty".into()))?;
            (k_t, v, t_cur)
        };
        let v = v_attn;
        let _ = l_k; // reserved for later use by G2 plan rebuild logic

        // Attack C: HIP fast path using `gqa_attention_k_transposed`
        // with the pre-transposed K from our kv_cache. K is already in
        // `(B, n_kv_h, D, T)` layout so the inner matmul is a direct
        // `(B, n_kv_h, n_rep*L, D) × (B, n_kv_h, D, T)` with no
        // transpose materialisation per call. Internally still uses
        // Q0c's fused `masked_softmax_scale`.
        //
        // Mask conversion is the same as Q1: u8 causal mask →
        // additive f32 → reshaped to (1, 1, seq_len, kv_len).
        #[cfg(feature = "hip")]
        let hip_fast_out: Option<Tensor> = if matches!(q.device(), Device::Hip(_))
            && q.dtype() == DType::F32
            && q.is_contiguous()
            // K/V from KvCache are non-contiguous narrow views on last dim.
            // gqa_attention_k_transposed handles non-contiguous k_t internally
            // (strided flash-attn kernel or internal contiguous() fallback).
        {
            let mask_additive: Option<Tensor> = match mask {
                None => {
                    // Decode path: if we padded L_k beyond T_cur, build a
                    // mask that kills padded positions. Shape (1, 1, 1, L_k).
                    if seq_len == 1 && l_k_padded > t_cur {
                        // Build via CPU u8 vector, move to GPU, apply neg_inf.
                        let mut m = vec![0u8; l_k_padded];
                        for j in t_cur..l_k_padded { m[j] = 1; }
                        let m_u8 = Tensor::from_vec(m, (1, 1, 1, l_k_padded), q.device())?;
                        let zero = Tensor::zeros((1, 1, 1, l_k_padded), DType::F32, q.device())?;
                        let neg = self.neg_inf.broadcast_as((1, 1, 1, l_k_padded))?;
                        Some(m_u8.where_cond(&neg, &zero)?.contiguous()?)
                    } else {
                        None
                    }
                }
                Some(m) => {
                    // Prefill path: k_t is unpadded (we only pad decode).
                    // Keep the original mask conversion.
                    let mdims = m.dims();
                    let (a, b) = (mdims[0], mdims[1]);
                    let zero = Tensor::zeros(mdims, DType::F32, q.device())?;
                    let neg = self.neg_inf.broadcast_as(mdims)?;
                    let add = m.where_cond(&neg, &zero)?;
                    Some(add.reshape((1, 1, a, b))?.contiguous()?)
                }
            };
            let scale = 1.0 / (self.head_dim as f64).sqrt();
            let out = crate::models::quantized_blocks::attention::gqa_attention_k_transposed(
                &q,
                &k_t,
                &v,
                mask_additive.as_ref(),
                scale,
            )?;
            Some(out)
        } else {
            None
        };
        #[cfg(not(feature = "hip"))]
        let hip_fast_out: Option<Tensor> = None;

        let y = if let Some(out) = hip_fast_out {
            out
        } else if q.device().is_metal() && seq_len == 1 {
            // SDPA will do MQA for us. Metal/CPU path still uses the
            // normal (B, n_kv_h, T, D) layout; reconstruct from k_t.
            let k_normal = k_t.transpose(2, 3)?.contiguous()?;
            candle_nn::ops::sdpa(
                &q,
                &k_normal,
                &v,
                None,
                false,
                1. / (self.head_dim as f32).sqrt(),
                1.,
            )?
        } else {
            // Non-HIP fallback: reconstruct normal-layout K, then use
            // the original repeat_kv + rocBLAS chain.
            let k_normal = k_t.transpose(2, 3)?.contiguous()?;
            let k = crate::utils::repeat_kv(k_normal, self.n_head / self.n_kv_head)?;
            let v = crate::utils::repeat_kv(v, self.n_head / self.n_kv_head)?;

            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = match mask {
                None => att,
                Some(mask) => {
                    let mask = mask.broadcast_as(att.shape())?;
                    masked_fill(&att, &mask, &self.neg_inf)?
                }
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.matmul(&v.contiguous()?)?
        };

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, self.n_head * self.head_dim])?;
        let y = self.attention_wo.forward(&y)?;
        Ok(y)
    }
}

#[derive(Debug)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    /// Mask cache keyed by (seq_len, kv_len).
    /// kv_len = index_pos + seq_len, so the mask is rectangular when prefix
    /// KV cache entries exist (index_pos > 0).
    masks: HashMap<(usize, usize), Tensor>,
    span: tracing::Span,
    span_output: tracing::Span,
    /// G2: Decode op cache state.
    #[cfg(feature = "hip")]
    decode_state: DecodeState,
}

/// G2: State machine for decode op cache.
///
/// Tokens 1-2: warm up (pool + KvCache init).
/// Token 3: record ops + decode-alloc record (capture buffer addresses).
/// Token 4: record ops + decode-alloc replay (stable addresses) → build plan.
/// Token 5: first replay (optionally capture HIP graph for G3).
/// Token 6+: HIP graph replay (single API call for all kernels).
#[cfg(feature = "hip")]
enum DecodeState {
    /// Token 1: cold init.
    Init,
    /// Token 2: warm-up (pool stabilizing).
    WarmUp,
    /// Token 3: first recording captured (decode-alloc was in record mode).
    /// Also stores the input tensor device pointer so we can mark matching
    /// args as External in the plan (token-id tensor is freshly allocated
    /// in the normal pool between forwards → its ptr changes every call).
    Recorded1 {
        ops: Vec<candle::hip_backend::decode_cache::RecordedOp>,
        input_ptr: usize,
    },
    /// Plan built from two recordings, ready for direct replay.
    Replay(candle::hip_backend::decode_cache::DecodePlan),
    /// HIP graph captured — single-call execution per decode token.
    /// `DecodeGraph` owns the captured graph + per-op node handles + the
    /// stable kernelParams pointer arrays. On each decode token we patch
    /// the ~130 dynamic kernel nodes with fresh counter-advanced args, then
    /// issue a single `hipGraphLaunch`. Valid only while the live n_kv-
    /// padded L_k matches `captured_l_k`.
    Graph {
        plan: candle::hip_backend::decode_cache::DecodePlan,
        graph: candle::hip_backend::decode_cache::DecodeGraph,
        captured_l_k: usize,
    },
}

#[cfg(feature = "hip")]
impl std::fmt::Debug for DecodeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Init => write!(f, "Init"),
            Self::WarmUp => write!(f, "WarmUp"),
            Self::Recorded1 { ops, .. } => write!(f, "Recorded1({} ops)", ops.len()),
            Self::Replay(plan) => write!(f, "Replay({} ops)", plan.len()),
            Self::Graph { plan, .. } => write!(f, "Graph({} ops)", plan.len()),
        }
    }
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

impl ModelWeights {
    pub fn from_ggml(mut ct: ggml_file::Content, gqa: usize) -> Result<Self> {
        let head_dim = (ct.hparams.n_embd / ct.hparams.n_head) as usize;
        let (cos, sin) = precomput_freqs_cis(head_dim, 10000., &ct.device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &ct.device)?;
        let tok_embeddings = ct.remove("tok_embeddings.weight")?;
        let tok_embeddings = tok_embeddings.dequantize(&ct.device)?;
        let norm = RmsNorm::from_qtensor(ct.remove("norm.weight")?, 1e-5)?;
        let output = ct.remove("output.weight")?;
        let mut layers = Vec::with_capacity(ct.hparams.n_layer as usize);
        for layer_idx in 0..ct.hparams.n_layer {
            let prefix = format!("layers.{layer_idx}");
            let attention_wq = ct.remove(&format!("{prefix}.attention.wq.weight"))?;
            let attention_wk = ct.remove(&format!("{prefix}.attention.wk.weight"))?;
            let attention_wv = ct.remove(&format!("{prefix}.attention.wv.weight"))?;
            let attention_wo = ct.remove(&format!("{prefix}.attention.wo.weight"))?;
            let mlp_or_moe = {
                let feed_forward_w1 = ct.remove(&format!("{prefix}.feed_forward.w1.weight"))?;
                let feed_forward_w2 = ct.remove(&format!("{prefix}.feed_forward.w2.weight"))?;
                let feed_forward_w3 = ct.remove(&format!("{prefix}.feed_forward.w3.weight"))?;
                MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                })
            };
            let attention_norm = ct.remove(&format!("{prefix}.attention_norm.weight"))?;
            let ffn_norm = ct.remove(&format!("{prefix}.ffn_norm.weight"))?;
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, 1e-5)?,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, 1e-5)?,
                n_head: ct.hparams.n_head as usize,
                n_kv_head: ct.hparams.n_head as usize / gqa,
                head_dim: (ct.hparams.n_embd / ct.hparams.n_head) as usize,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                kv_cache: None,
                device: Device::Cpu,
                span_attn,
                span_rot,
                span_mlp,
            })
        }
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, ct.hparams.n_embd as usize),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            masks: HashMap::new(),
            span,
            span_output,
            #[cfg(feature = "hip")]
            decode_state: DecodeState::Init,
        })
    }

    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        // Auto-detect architecture prefix from GGUF metadata.
        // Supports llama, mistral, devstral, and other llama-compatible architectures.
        let arch = match md_get("general.architecture").and_then(|v| v.to_string()) {
            Ok(a) => a.to_string(),
            Err(_) => "llama".to_string(),
        };

        // Parameter extraction from metadata.
        let n_expert = md_get(&format!("{arch}.expert_count"))
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let n_expert_used = md_get(&format!("{arch}.expert_used_count"))
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let head_count = md_get(&format!("{arch}.attention.head_count"))?.to_u32()? as usize;
        let head_count_kv = md_get(&format!("{arch}.attention.head_count_kv"))?.to_u32()? as usize;
        let block_count = md_get(&format!("{arch}.block_count"))?.to_u32()? as usize;
        let embedding_length = md_get(&format!("{arch}.embedding_length"))?.to_u32()? as usize;
        let rope_dim = md_get(&format!("{arch}.rope.dimension_count"))?.to_u32()? as usize;
        // Strangely this value is generally 1e-6 in GGUF file but used to be 1e-5 by default.
        let rms_norm_eps = md_get(&format!("{arch}.attention.layer_norm_rms_epsilon"))?.to_f32()? as f64;

        let head_dim = md_get(&format!("{arch}.attention.key_length"))
            .and_then(|v| v.to_u32())
            .unwrap_or((embedding_length / head_count) as u32) as usize;
        let rope_freq_base = md_get(&format!("{arch}.rope.freq_base"))
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);
        let (cos, sin) = precomput_freqs_cis(rope_dim, rope_freq_base, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let tok_embeddings_q = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => tok_embeddings_q,
        };
        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;
            let mlp_or_moe = if n_expert <= 1 {
                let feed_forward_w1 =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
                let feed_forward_w2 =
                    ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
                let feed_forward_w3 =
                    ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
                MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                })
            } else {
                let feed_forward_gate_inp =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate_inp.weight"), device)?;
                let mut experts = Vec::with_capacity(n_expert);
                for i in 0..n_expert {
                    let feed_forward_w1 =
                        ct.tensor(reader, &format!("{prefix}.ffn_gate.{i}.weight"), device)?;
                    let feed_forward_w2 =
                        ct.tensor(reader, &format!("{prefix}.ffn_down.{i}.weight"), device)?;
                    let feed_forward_w3 =
                        ct.tensor(reader, &format!("{prefix}.ffn_up.{i}.weight"), device)?;
                    experts.push(Mlp {
                        feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                        feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                        feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                    })
                }
                MlpOrMoe::MoE {
                    n_expert_used,
                    feed_forward_gate_inp: QMatMul::from_qtensor(feed_forward_gate_inp)?,
                    experts,
                }
            };
            let attention_norm =
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                kv_cache: None,
                device: device.clone(),
                span_attn,
                span_rot,
                span_mlp,
            })
        }
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            masks: HashMap::new(),
            span,
            span_output,
            #[cfg(feature = "hip")]
            decode_state: DecodeState::Init,
        })
    }

    /// Load a GGUF model with layer-split across multiple devices.
    /// Layers are distributed evenly across the provided devices.
    /// Embeddings and output go on the first device.
    pub fn from_gguf_sharded<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        devices: &[Device],
    ) -> Result<Self> {
        if devices.is_empty() {
            candle::bail!("at least one device required");
        }
        // Even with 1 GPU, use the sharded path for consistent behavior.

        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let arch = match md_get("general.architecture").and_then(|v| v.to_string()) {
            Ok(a) => a.to_string(),
            Err(_) => "llama".to_string(),
        };

        let n_expert = md_get(&format!("{arch}.expert_count"))
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let n_expert_used = md_get(&format!("{arch}.expert_used_count"))
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let head_count = md_get(&format!("{arch}.attention.head_count"))?.to_u32()? as usize;
        let head_count_kv = md_get(&format!("{arch}.attention.head_count_kv"))?.to_u32()? as usize;
        let block_count = md_get(&format!("{arch}.block_count"))?.to_u32()? as usize;
        let embedding_length = md_get(&format!("{arch}.embedding_length"))?.to_u32()? as usize;
        let rope_dim = md_get(&format!("{arch}.rope.dimension_count"))?.to_u32()? as usize;
        let rms_norm_eps =
            md_get(&format!("{arch}.attention.layer_norm_rms_epsilon"))?.to_f32()? as f64;
        let head_dim = md_get(&format!("{arch}.attention.key_length"))
            .and_then(|v| v.to_u32())
            .unwrap_or((embedding_length / head_count) as u32) as usize;
        let rope_freq_base = md_get(&format!("{arch}.rope.freq_base"))
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);

        // Embeddings and output on the first GPU (same device as token tensors).
        let dev0 = &devices[0];

        let (cos, sin) = precomput_freqs_cis(rope_dim, rope_freq_base, dev0)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, dev0)?;

        let tok_embeddings_q = ct.tensor(reader, "token_embd.weight", dev0)?;
        let tok_embeddings = tok_embeddings_q.dequantize(dev0)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", dev0)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", dev0) {
            Ok(tensor) => tensor,
            Err(_) => tok_embeddings_q,
        };

        // Distribute layers across devices
        let n_devices = devices.len();
        println!(
            "Layer-split: {block_count} layers across {n_devices} devices (embed={embedding_length})"
        );

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let dev_idx = layer_idx * n_devices / block_count;
            let layer_device = &devices[dev_idx];
            let prefix = format!("blk.{layer_idx}");

            if layer_idx == 0 || dev_idx != (layer_idx.wrapping_sub(1)) * n_devices / block_count {
                println!("  layers {layer_idx}+ → {:?}", layer_device.location());
            }

            // Transfer cos/sin/neg_inf from CPU to this layer's device
            let layer_cos = cos.to_device(layer_device)?;
            let layer_sin = sin.to_device(layer_device)?;
            let layer_neg_inf = neg_inf.to_device(layer_device)?;

            let attention_wq =
                ct.tensor(reader, &format!("{prefix}.attn_q.weight"), layer_device)?;
            let attention_wk =
                ct.tensor(reader, &format!("{prefix}.attn_k.weight"), layer_device)?;
            let attention_wv =
                ct.tensor(reader, &format!("{prefix}.attn_v.weight"), layer_device)?;
            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), layer_device)?;
            let mlp_or_moe = if n_expert <= 1 {
                let feed_forward_w1 =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), layer_device)?;
                let feed_forward_w2 =
                    ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), layer_device)?;
                let feed_forward_w3 =
                    ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), layer_device)?;
                MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                })
            } else {
                let feed_forward_gate_inp =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate_inp.weight"), layer_device)?;
                let mut experts = Vec::with_capacity(n_expert);
                for i in 0..n_expert {
                    let feed_forward_w1 = ct.tensor(
                        reader,
                        &format!("{prefix}.ffn_gate.{i}.weight"),
                        layer_device,
                    )?;
                    let feed_forward_w2 = ct.tensor(
                        reader,
                        &format!("{prefix}.ffn_down.{i}.weight"),
                        layer_device,
                    )?;
                    let feed_forward_w3 = ct.tensor(
                        reader,
                        &format!("{prefix}.ffn_up.{i}.weight"),
                        layer_device,
                    )?;
                    experts.push(Mlp {
                        feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                        feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                        feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                    })
                }
                MlpOrMoe::MoE {
                    n_expert_used,
                    feed_forward_gate_inp: QMatMul::from_qtensor(feed_forward_gate_inp)?,
                    experts,
                }
            };
            let attention_norm =
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), layer_device)?;
            let ffn_norm =
                ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), layer_device)?;
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                cos: layer_cos,
                sin: layer_sin,
                neg_inf: layer_neg_inf,
                kv_cache: None,
                device: layer_device.clone(),
                span_attn,
                span_rot,
                span_mlp,
            })
        }
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            masks: HashMap::new(),
            span,
            span_output,
            #[cfg(feature = "hip")]
            decode_state: DecodeState::Init,
        })
    }

    /// Build a causal attention mask of shape `(seq_len, kv_len)` where
    /// `kv_len = index_pos + seq_len`.
    ///
    /// When `index_pos == 0` the mask is square `(seq_len, seq_len)` — the
    /// classic case with an empty KV cache.
    ///
    /// When `index_pos > 0` the KV cache already holds `index_pos` entries from
    /// a previously fed prefix.  The mask becomes rectangular: the first
    /// `index_pos` columns are all 0 (every query attends to every prefix key)
    /// and the remaining `seq_len` columns form the standard causal triangle
    /// (query at global position `index_pos + i` cannot attend to keys at global
    /// positions `> index_pos + i`).
    ///
    /// # Shape example  (index_pos=65, seq_len=4)
    /// ```text
    ///              kv 0..64 (prefix)   kv 65  kv 66  kv 67  kv 68
    /// query 65:       0  0 … 0           0      1      1      1
    /// query 66:       0  0 … 0           0      0      1      1
    /// query 67:       0  0 … 0           0      0      0      1
    /// query 68:       0  0 … 0           0      0      0      0
    /// ```
    fn mask(&mut self, seq_len: usize, index_pos: usize, device: &Device) -> Result<Tensor> {
        let kv_len = index_pos + seq_len;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            Ok(mask.clone())
        } else {
            let mask = crate::utils::build_causal_mask(seq_len, index_pos, device)?;
            self.masks.insert((seq_len, kv_len), mask.clone());
            Ok(mask)
        }
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        // Keep an unshadowed ref to the original input tensor parameter so
        // the G2 state machine (post-forward) can read its device pointer.
        #[cfg(feature = "hip")]
        let input_x_param: &Tensor = x;

        #[cfg(feature = "hip")]
        let is_decode = seq_len == 1 && index_pos > 0 && std::env::var("CANDLE_G2_DISABLE").is_err();

        #[cfg(feature = "hip")]
        if is_decode && std::env::var("CANDLE_G2_XPTR_DEBUG").is_ok() {
            let p = Self::hip_device_ptr(x).unwrap_or(0);
            eprintln!("[G2] forward(idx={}) x_ptr=0x{:x}", index_pos, p);
        }

        // ── G2/G3: Decode replay fast path ──────────────────────────────
        //
        // Once the plan is built (token 5+), skip the entire Rust forward
        // pass and replay recorded kernels directly.
        #[cfg(feature = "hip")]
        if is_decode {
            match &mut self.decode_state {
                // G3: HIP graph — single API call for all ops, with
                // kernel-node parameter patching for the ~130 dynamic args
                // each decode token.
                DecodeState::Graph { plan, graph, captured_l_k } => {
                    use candle::hip_backend::hipdarc;
                    // Check pad boundary: if the live n_kv-padded L_k no
                    // longer matches what was captured, the graph is stale.
                    let current_t = index_pos + seq_len;
                    let pad_default: usize = if std::env::var("CANDLE_G2_REPLAY").is_ok() { 256 } else { 1 };
                    let pad_t = std::env::var("CANDLE_NKV_PAD")
                        .ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(pad_default);
                    let live_l_k = if pad_t > 1 {
                        ((current_t + pad_t - 1) / pad_t) * pad_t
                    } else {
                        current_t
                    };
                    if live_l_k != *captured_l_k {
                        eprintln!("[G3] pad boundary crossed (captured={} live={}), rebuilding",
                            captured_l_k, live_l_k);
                        self.decode_state = DecodeState::Init;
                    } else {
                        hipdarc::driver::decode_alloc_resume();
                        let x_ptr_now = Self::hip_device_ptr(x)?;
                        let anchor = plan.input_anchor_ptr;
                        let nbytes = plan.input_anchor_bytes;
                        let dev = Self::hip_device_from_tensor(x)?;
                        if anchor != 0 && nbytes > 0 && x_ptr_now != anchor {
                            let stream = dev.stream();
                            unsafe {
                                let rc = hipdarc::sys::hipMemcpyAsync(
                                    anchor as hipdarc::sys::hipDeviceptr_t,
                                    x_ptr_now as *const std::ffi::c_void,
                                    nbytes,
                                    hipdarc::sys::hipMemcpyKind::hipMemcpyDeviceToDevice,
                                    stream.raw(),
                                );
                                if rc != hipdarc::sys::hipError_t::hipSuccess {
                                    candle::bail!("G3 input memcpy failed: {:?}", rc);
                                }
                            }
                        }
                        // Advance counters so dynamic args reflect the
                        // current token, then patch + launch.
                        plan.patch_all_externals(anchor);
                        if std::env::var("CANDLE_G2_NO_ADVANCE").is_err() {
                            plan.advance_counters();
                        }
                        unsafe { graph.patch_and_launch(plan, &dev)?; }
                        hipdarc::driver::decode_alloc_pause();
                        hipdarc::driver::decode_alloc_reset();
                        return plan.output_tensor(&dev);
                    }
                }
                // G2: Direct replay — infrastructure ready, attention uses
                // v2 (Tensile-free) so recorded kernels stay valid, BUT the
                // input tensor pointer changes per token (caller allocates
                // `input` in the normal pool between forwards). Replay uses
                // the stale token-4 input pointer → embedding kernel reads
                // wrong indices → garbage after ~8 replays.
                // TODO: promote input to a stable decode-alloc slot and
                // copy new token ids into it before each replay. Opt-in
                // via CANDLE_G2_REPLAY=1 for debugging.
                DecodeState::Replay(plan) => {
                    // Cap replay count. Beyond ~4 replays the counter-delta
                    // approach diverges. We fall through to normal forward
                    // (not a new recording — too expensive, see bench).
                    let max_replays = std::env::var("CANDLE_G2_REPLAY_MAX")
                        .ok()
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(4);
                    if std::env::var("CANDLE_G2_REPLAY").is_ok()
                        && plan.replay_count() < max_replays
                    {
                        use candle::hip_backend::hipdarc;
                        hipdarc::driver::decode_alloc_resume();
                        let x_ptr_now = Self::hip_device_ptr(x)?;
                        let anchor = plan.input_anchor_ptr;
                        let nbytes = plan.input_anchor_bytes;
                        let dev = Self::hip_device_from_tensor(x)?;
                        if anchor != 0 && nbytes > 0 && x_ptr_now != anchor {
                            let stream = dev.stream();
                            unsafe {
                                let rc = hipdarc::sys::hipMemcpyAsync(
                                    anchor as hipdarc::sys::hipDeviceptr_t,
                                    x_ptr_now as *const std::ffi::c_void,
                                    nbytes,
                                    hipdarc::sys::hipMemcpyKind::hipMemcpyDeviceToDevice,
                                    stream.raw(),
                                );
                                if rc != hipdarc::sys::hipError_t::hipSuccess {
                                    candle::bail!("G2 replay input memcpy failed: {:?}", rc);
                                }
                            }
                        }
                        plan.patch_all_externals(anchor);
                        if std::env::var("CANDLE_G2_NO_ADVANCE").is_err() {
                            plan.advance_counters();
                        }
                        unsafe { plan.replay(&dev)?; }
                        // G3: after a few successful replays (plan is
                        // warm, decode_alloc pool stable), capture the
                        // whole sequence as a HIP graph for single-launch
                        // execution. Opt-in via CANDLE_G3_GRAPH=1.
                        let g3_enabled = std::env::var("CANDLE_G3_GRAPH").is_ok();
                        let g3_after = std::env::var("CANDLE_G3_AFTER")
                            .ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(2);
                        let should_capture = g3_enabled
                            && plan.replay_count() == g3_after;
                        if should_capture {
                            let current_t = index_pos + seq_len;
                            let pad_t_env: usize = std::env::var("CANDLE_NKV_PAD")
                                .ok().and_then(|s| s.parse::<usize>().ok())
                                .unwrap_or(if g3_enabled || std::env::var("CANDLE_G2_REPLAY").is_ok() { 256 } else { 1 });
                            let live_l_k = if pad_t_env > 1 {
                                ((current_t + pad_t_env - 1) / pad_t_env) * pad_t_env
                            } else {
                                current_t
                            };
                            eprintln!("[G3] capturing graph after {} replays (L_k_padded={})",
                                g3_after, live_l_k);
                            // We need to transition state. Use mem::replace.
                            hipdarc::driver::decode_alloc_pause();
                            hipdarc::driver::decode_alloc_reset();
                            let output = plan.output_tensor(&dev)?;
                            let old = std::mem::replace(&mut self.decode_state, DecodeState::Init);
                            if let DecodeState::Replay(plan) = old {
                                match unsafe { candle::hip_backend::decode_cache::DecodeGraph::capture(&plan, &dev) } {
                                    Ok(graph) => {
                                        eprintln!("[G3] graph captured successfully");
                                        self.decode_state = DecodeState::Graph {
                                            plan, graph, captured_l_k: live_l_k,
                                        };
                                    }
                                    Err(e) => {
                                        eprintln!("[G3] graph capture failed: {:?} — staying in Replay", e);
                                        self.decode_state = DecodeState::Replay(plan);
                                    }
                                }
                            }
                            return Ok(output);
                        }
                        if std::env::var("CANDLE_G2_REPLAY_TRACE").is_ok() {
                            let _ = dev.stream().synchronize();
                            // Sample a few output values via the output_tensor.
                            let t = plan.output_tensor(&dev)?;
                            let sample = t.narrow(
                                t.rank() - 1,
                                0,
                                std::cmp::min(4, t.dim(t.rank()-1).unwrap_or(1)),
                            ).ok().and_then(|s| s.to_vec1::<f32>().ok());
                            eprintln!("[G2] replay#{} idx={} anchor=0x{:x} output head: {:?}",
                                plan.replay_count(), index_pos, anchor, sample);
                        }
                        hipdarc::driver::decode_alloc_pause();
                        hipdarc::driver::decode_alloc_reset();
                        return plan.output_tensor(&dev);
                    }
                    // Default: fall through to normal forward.
                }
                _ => {} // fall through to normal forward + recording
            }
        }

        // ── Normal forward pass ─────────────────────────────────────────
        // (used for prefill, first few decode tokens, and recording)

        // Pre-compute masks per device to avoid borrow conflict in the layer loop
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, index_pos, x.device())?)
        };
        let _enter = self.span.enter();
        let mut layer_in = self.tok_embeddings.forward(x)?;
        for layer in self.layers.iter_mut() {
            // Transfer activations to this layer's device if needed
            if !layer_in.device().same_device(&layer.device) {
                layer_in = layer_in.to_device(&layer.device)?;
            }
            let layer_mask = match &mask {
                Some(m) if !m.device().same_device(&layer.device) => {
                    Some(m.to_device(&layer.device)?)
                }
                other => other.clone(),
            };
            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(&x, layer_mask.as_ref(), index_pos)?;
            let x = (attn + residual)?;

            // MLP
            let _enter = layer.span_mlp.enter();
            // Fused Q4_0 decode FFN path: uses two fused kernels
            //   gate_up_fused:  gate MMVQ + up MMVQ in one launch (shared x_q8)
            //   down_residual:  down MMVQ + residual add in one launch
            // Saves 2 kernel launches per layer × 22 layers = 44/token.
            // Opt-out via CANDLE_FFN_FUSED_OFF=1.
            #[cfg(feature = "hip")]
            let ffn_fused: Option<Tensor> = if seq_len == 1
                && matches!(x.device(), Device::Hip(_))
                && std::env::var("CANDLE_FFN_FUSED_OFF").is_err()
            {
                if let MlpOrMoe::Mlp(mlp) = &layer.mlp_or_moe {
                    if let Some((w_gate, w_up, w_down)) = mlp.as_q4_0_trio() {
                        match fused_q4_0_ffn_decode(
                            &x, &layer.ffn_norm, &w_gate, &w_up, &w_down,
                        ) {
                            Ok(out) => Some(out),
                            Err(e) => {
                                eprintln!("[FFN] fused Q4_0 path failed: {:?} — falling back", e);
                                None
                            }
                        }
                    } else { None }
                } else { None }
            } else { None };
            #[cfg(not(feature = "hip"))]
            let ffn_fused: Option<Tensor> = None;

            let x = if let Some(fused_out) = ffn_fused {
                fused_out
            } else {
                let residual = &x;
                let x = layer.ffn_norm.forward(&x)?;
                let x = layer.mlp_or_moe.forward(&x)?;
                (x + residual)?
            };
            layer_in = x
        }
        // Norm/output are on the embedding device (first GPU).
        let embed_device = self.tok_embeddings.embeddings().device();
        if !layer_in.device().same_device(embed_device) {
            layer_in = layer_in.to_device(embed_device)?;
        }
        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        let _enter = self.span_output.enter();
        let result = self.output.forward(&x)?;

        // ── G2: Post-forward recording state machine ────────────────────
        #[cfg(feature = "hip")]
        if is_decode {
            use candle::hip_backend::decode_cache;
            use candle::hip_backend::hipdarc::driver as hdrv;

            if decode_cache::is_recording() {
                if let Some(recording) = decode_cache::stop_recording() {
                    // Capture input tensor pointer so we can mark matching
                    // args as External patches (the token-id tensor is
                    // freshly allocated in the normal pool between forwards).
                    // `x` at this point is shadowed by intermediate tensors;
                    // `input_x_param` below holds the original parameter.
                    let x_ptr_now = Self::hip_device_ptr(input_x_param)?;
                    match std::mem::replace(&mut self.decode_state, DecodeState::Init) {
                        DecodeState::WarmUp => {
                            hdrv::decode_alloc_start_replay();
                            decode_cache::start_recording();
                            eprintln!(
                                "[G2] token 3 recorded: {} ops, {} alloc entries, x_ptr=0x{:x}",
                                recording.len(),
                                hdrv::decode_alloc_entry_count(),
                                x_ptr_now,
                            );
                            self.decode_state = DecodeState::Recorded1 {
                                ops: recording,
                                input_ptr: x_ptr_now,
                            };
                        }
                        DecodeState::Recorded1 { ops: first, input_ptr: x_ptr_3 } => {
                            let output_ptr = Self::hip_device_ptr(&result)?;
                            let output_f32_count = result.elem_count();
                            let output_shape = result.dims().to_vec();
                            let externals = [x_ptr_3, x_ptr_now];
                            // Byte size of the input-id tensor for anchor memcpy.
                            let input_bytes = input_x_param.elem_count()
                                * input_x_param.dtype().size_in_bytes();
                            if let Some(mut plan) = decode_cache::DecodePlan::from_two_recordings_with_externals(
                                &first,
                                &recording,
                                output_ptr,
                                output_f32_count,
                                output_shape,
                                &externals,
                            ) {
                                plan.set_input_anchor_bytes(input_bytes);
                                eprintln!(
                                    "[G2] input anchor: ptr=0x{:x} bytes={}",
                                    plan.input_anchor_ptr, plan.input_anchor_bytes
                                );
                                eprintln!(
                                    "[G2] plan ready: {} ops, {} dynamic/{} fixed, {} external patches",
                                    plan.len(),
                                    plan.dynamic_arg_count(),
                                    plan.fixed_arg_count(),
                                    plan.external_patch_count(),
                                );
                                // Pause + reset so sampling allocs after this forward
                                // use normal pool, and next call's input tensor gets cursor 0.
                                hdrv::decode_alloc_pause();
                                hdrv::decode_alloc_reset();
                                self.decode_state = DecodeState::Replay(plan);
                            } else {
                                eprintln!("[G2] plan build failed — recordings diverged");
                                hdrv::decode_alloc_stop();
                                self.decode_state = DecodeState::Init;
                            }
                        }
                        _ => {
                            hdrv::decode_alloc_stop();
                            self.decode_state = DecodeState::Init;
                        }
                    }
                }
            } else {
                match &self.decode_state {
                    DecodeState::Init => {
                        self.decode_state = DecodeState::WarmUp;
                    }
                    DecodeState::WarmUp => {
                        // Token 2 done. Start decode-alloc recording + op recording.
                        hdrv::decode_alloc_start_record();
                        decode_cache::start_recording();
                    }
                    _ => {}
                }
            }
        }

        Ok(result)
    }

    /// Extract the raw HIP device pointer from a tensor's storage.
    #[cfg(feature = "hip")]
    fn hip_device_ptr(t: &Tensor) -> Result<usize> {
        let (storage, layout) = t.storage_and_layout();
        match &*storage {
            candle::Storage::Hip(hip) => {
                let base = hip.slice.device_ptr() as usize;
                let offset_bytes = layout.start_offset() * t.dtype().size_in_bytes();
                Ok(base + offset_bytes)
            }
            _ => candle::bail!("expected HIP storage for decode replay"),
        }
    }

    /// Extract the HipDevice from a tensor.
    #[cfg(feature = "hip")]
    fn hip_device_from_tensor(t: &Tensor) -> Result<candle::hip_backend::HipDevice> {
        match t.device() {
            candle::Device::Hip(dev) => Ok(dev.clone()),
            _ => candle::bail!("expected HIP device"),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::build_causal_mask;
    use candle::{Device, Result};

    // ── Mask shape tests ──────────────────────────────────────────────────────

    /// Classic square mask: index_pos=0 produces (seq_len, seq_len).
    #[test]
    fn causal_mask_square_shape() -> Result<()> {
        let mask = build_causal_mask(4, 0, &Device::Cpu)?;
        assert_eq!(mask.dims(), [4, 4]);
        Ok(())
    }

    /// Rectangular mask: index_pos=N produces (seq_len, N + seq_len).
    #[test]
    fn causal_mask_rectangular_shape() -> Result<()> {
        let mask = build_causal_mask(4, 65, &Device::Cpu)?;
        assert_eq!(mask.dims(), [4, 69]);
        Ok(())
    }

    // ── Mask value tests ──────────────────────────────────────────────────────

    /// Square mask values: standard lower-triangular pattern (0=attend, 1=block).
    ///
    /// For seq_len=3, index_pos=0:
    ///   row 0 (global pos 0): attend to pos 0             → [0, 1, 1]
    ///   row 1 (global pos 1): attend to pos 0..1           → [0, 0, 1]
    ///   row 2 (global pos 2): attend to pos 0..2           → [0, 0, 0]
    #[test]
    fn causal_mask_square_values() -> Result<()> {
        let mask = build_causal_mask(3, 0, &Device::Cpu)?;
        let data: Vec<u8> = mask.flatten_all()?.to_vec1()?;
        assert_eq!(data, [0, 1, 1, 0, 0, 1, 0, 0, 0]);
        Ok(())
    }

    /// Rectangular mask values: prefix columns are all-zero, user columns
    /// form the causal triangle.
    ///
    /// For seq_len=3, index_pos=2 → kv_len=5:
    ///   row 0 (global pos 2): attend to kv 0..2  → [0,0, 0,1,1]
    ///   row 1 (global pos 3): attend to kv 0..3  → [0,0, 0,0,1]
    ///   row 2 (global pos 4): attend to kv 0..4  → [0,0, 0,0,0]
    #[test]
    fn causal_mask_rectangular_values() -> Result<()> {
        let mask = build_causal_mask(3, 2, &Device::Cpu)?;
        let data: Vec<u8> = mask.flatten_all()?.to_vec1()?;
        #[rustfmt::skip]
        assert_eq!(data, [
            0, 0,  0, 1, 1,
            0, 0,  0, 0, 1,
            0, 0,  0, 0, 0,
        ]);
        Ok(())
    }

    /// A single-token query (seq_len=1) with prefix produces a single row
    /// of all zeros — it can attend to every key including itself.
    #[test]
    fn causal_mask_single_query_with_prefix() -> Result<()> {
        let mask = build_causal_mask(1, 10, &Device::Cpu)?;
        assert_eq!(mask.dims(), [1, 11]);
        let data: Vec<u8> = mask.flatten_all()?.to_vec1()?;
        assert!(
            data.iter().all(|&v| v == 0),
            "single-query mask should be all-zero"
        );
        Ok(())
    }

    // ── Mask broadcast compatibility test ─────────────────────────────────────

    /// Verify the mask can be broadcast to (batch, heads, seq_len, kv_len) —
    /// the exact shape produced by `Q @ K^T` in forward_attn.
    /// This is the broadcast that previously panicked when index_pos > 0.
    #[test]
    fn causal_mask_broadcasts_to_attention_shape() -> Result<()> {
        let batch = 1usize;
        let heads = 8usize;
        let seq_len = 4usize;
        let index_pos = 10usize;

        let mask = build_causal_mask(seq_len, index_pos, &Device::Cpu)?;
        // Simulate the attention score shape Q @ K^T → (batch, heads, seq_len, kv_len)
        let kv_len = index_pos + seq_len;
        let att_shape = &[batch, heads, seq_len, kv_len];
        let broadcasted = mask.broadcast_as(att_shape.as_slice())?;
        assert_eq!(broadcasted.dims(), att_shape);
        Ok(())
    }
}
