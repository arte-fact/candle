#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{Device, Result, Tensor};

#[test]
fn kv_cache_k_transposed_shape() -> Result<()> {
    // Attack C: KvCache storing K in (..., D, T) layout. Incoming K is
    // canonical (B, n_kv, T_new, D); `append` auto-transposes.
    let b = 1;
    let n_kv = 2;
    let d = 4;
    let mut cache = candle_nn::kv_cache::KvCache::new_k_transposed(/*dim_v*/ 2, 16);
    assert!(cache.k_is_transposed());

    // First append: 3 new tokens. K and V in (B, n_kv, 3, D).
    let k_new: Vec<f32> = (0..(b * n_kv * 3 * d)).map(|i| i as f32).collect();
    let v_new: Vec<f32> = k_new.iter().map(|x| *x + 100.0).collect();
    let k_t = Tensor::from_slice(&k_new, (b, n_kv, 3, d), &Device::Cpu)?;
    let v_t = Tensor::from_slice(&v_new, (b, n_kv, 3, d), &Device::Cpu)?;
    let (k_out, v_out) = cache.append(&k_t, &v_t)?;
    // K returned from cache is in storage layout: (B, n_kv, D, T_current=3).
    assert_eq!(k_out.dims(), &[b, n_kv, d, 3]);
    // V stays canonical.
    assert_eq!(v_out.dims(), &[b, n_kv, 3, d]);

    // Verify K content: the original k[b, h, t, d] element should now
    // live at k_out[b, h, d, t].
    let k_orig: Vec<f32> = k_t.flatten_all()?.to_vec1()?;
    let k_got: Vec<f32> = k_out.flatten_all()?.to_vec1()?;
    for h in 0..n_kv {
        for t in 0..3 {
            for di in 0..d {
                let orig_idx = ((h * 3) + t) * d + di;
                let got_idx = ((h * d) + di) * 3 + t;
                assert_eq!(
                    k_orig[orig_idx], k_got[got_idx],
                    "h={h} t={t} d={di}"
                );
            }
        }
    }

    // Second append: 1 more token, simulating decode.
    let k_next: Vec<f32> = (0..(b * n_kv * 1 * d)).map(|i| 1000.0 + i as f32).collect();
    let v_next: Vec<f32> = k_next.iter().map(|x| *x + 100.0).collect();
    let k_t2 = Tensor::from_slice(&k_next, (b, n_kv, 1, d), &Device::Cpu)?;
    let v_t2 = Tensor::from_slice(&v_next, (b, n_kv, 1, d), &Device::Cpu)?;
    let (k_out2, v_out2) = cache.append(&k_t2, &v_t2)?;
    assert_eq!(k_out2.dims(), &[b, n_kv, d, 4]);
    assert_eq!(v_out2.dims(), &[b, n_kv, 4, d]);

    // Verify the new token lands in the last column of K.
    let k_got2: Vec<f32> = k_out2.flatten_all()?.to_vec1()?;
    for h in 0..n_kv {
        for di in 0..d {
            let orig_idx = h * d + di;
            let got_idx = ((h * d) + di) * 4 + 3;
            assert_eq!(
                k_next[orig_idx], k_got2[got_idx],
                "tail token h={h} d={di}"
            );
        }
    }
    Ok(())
}

#[test]
fn kv_cache() -> Result<()> {
    let mut cache = candle_nn::kv_cache::Cache::new(0, 16);
    for _ in [0, 1] {
        assert_eq!(cache.current_seq_len(), 0);
        let data = cache.current_data()?;
        assert!(data.is_none());
        let t = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
        cache.append(&t)?;
        let data = cache.current_data()?.unwrap();
        assert_eq!(data.to_vec1::<f32>()?, [1., 2., 3.]);
        let t = Tensor::new(&[4f32], &Device::Cpu)?;
        cache.append(&t)?;
        let data = cache.current_data()?.unwrap();
        assert_eq!(data.to_vec1::<f32>()?, [1., 2., 3., 4.]);
        let t = Tensor::new(&[0f32, 5., 6., 7.], &Device::Cpu)?;
        cache.append(&t)?;
        let data = cache.current_data()?.unwrap();
        assert_eq!(data.to_vec1::<f32>()?, [1., 2., 3., 4., 0., 5., 6., 7.]);
        assert_eq!(cache.current_seq_len(), 8);
        cache.reset();
    }
    Ok(())
}

#[test]
fn rotating_kv_cache() -> Result<()> {
    let mut cache = candle_nn::kv_cache::RotatingCache::new(0, 6);
    for _ in [0, 1] {
        assert_eq!(cache.offset(), 0);
        assert_eq!(cache.current_seq_len(), 0);
        let data = cache.current_data()?;
        assert!(data.is_none());
        assert_eq!(cache.positions(1), &[0]);
        assert_eq!(cache.positions(2), &[0, 1]);
        let t = Tensor::new(&[1., 2., 3.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [1., 2., 3.]);
        assert_eq!(cache.positions(0), &[0, 1, 2]);
        assert_eq!(cache.positions(1), &[0, 1, 2, 3]);
        assert_eq!(cache.positions(2), &[0, 1, 2, 3, 4]);
        assert_eq!(cache.positions(3), &[0, 1, 2, 3, 4, 5]);
        assert_eq!(cache.positions(4), &[6, 1, 2, 3, 4, 5]);
        let t = Tensor::new(&[4.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [1., 2., 3., 4.]);
        let t = Tensor::new(&[0., 5., 6., 7.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [6., 7., 3., 4., 0., 5.]);
        assert_eq!(cache.current_seq_len(), 8);
        assert_eq!(cache.offset(), 2);

        let t = Tensor::new(&[8.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [6., 7., 8., 4., 0., 5.]);
        assert_eq!(cache.current_seq_len(), 9);
        assert_eq!(cache.offset(), 3);

        let t = Tensor::new(&[9., 10., 11.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [6., 7., 8., 9., 10., 11.]);
        assert_eq!(cache.current_seq_len(), 12);
        assert_eq!(cache.offset(), 0);

        let t = Tensor::new(&[12.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [12., 7., 8., 9., 10., 11.]);
        assert_eq!(cache.current_seq_len(), 13);
        assert_eq!(cache.offset(), 1);

        let mask = cache.attn_mask(2, &Device::Cpu)?.unwrap();
        assert_eq!(
            mask.to_vec2::<u8>()?,
            &[[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        );
        let mask = cache.attn_mask(3, &Device::Cpu)?.unwrap();
        assert_eq!(
            mask.to_vec2::<u8>()?,
            &[[0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0]],
        );
        assert_eq!(cache.positions(0), &[12, 7, 8, 9, 10, 11]);
        assert_eq!(cache.positions(2), &[12, 13, 14, 9, 10, 11]);
        assert_eq!(cache.positions(3), &[12, 13, 14, 15, 10, 11]);
        assert_eq!(cache.positions(8), &[13, 14, 15, 16, 17, 18, 19, 20]);
        let t = Tensor::new(&[0., 1., 2., 3., 4., 5., 6., 7., 8.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [0., 1., 2., 3., 4., 5., 6., 7., 8.]);
        assert_eq!(cache.current_seq_len(), 22);
        assert_eq!(cache.offset(), 0);
        assert_eq!(cache.positions(0), &[16, 17, 18, 19, 20, 21]);
        assert_eq!(cache.positions(1), &[22, 17, 18, 19, 20, 21]);

        let mask = cache.attn_mask(1, &Device::Cpu)?;
        assert!(mask.is_none());
        let mask = cache.attn_mask(2, &Device::Cpu)?.unwrap();
        assert_eq!(
            mask.to_vec2::<u8>()?,
            &[[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        );
        let mask = cache.attn_mask(3, &Device::Cpu)?.unwrap();
        assert_eq!(
            mask.to_vec2::<u8>()?,
            &[[0, 1, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        );
        let t = Tensor::new(&[42.], &Device::Cpu)?;

        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1::<f64>()?, [42., 4., 5., 6., 7., 8.]);
        assert_eq!(cache.current_seq_len(), 23);
        assert_eq!(cache.offset(), 1);

        cache.reset();
    }
    Ok(())
}
