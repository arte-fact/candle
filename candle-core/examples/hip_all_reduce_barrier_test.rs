//! X8.4.1 smoke test: thread-safe barrier all-reduce on HipCluster.
//!
//! W concurrent OS threads each call
//! `HipCluster::all_reduce_sum_f32_barrier(rank, input, &mut output)`.
//! Each rank's input pattern is `i + r * 1000.0`; after the collective,
//! every rank's output must equal `W * i + 1000 * Σ_r r`.

use anyhow::Result;
use candle_core::hip_backend::cluster::HipCluster;
use candle_core::{Device, HipDevice};

fn main() -> Result<()> {
    let mut devs = Vec::<HipDevice>::new();
    for ord in 0..16 {
        if let Ok(Device::Hip(d)) = Device::new_hip(ord) {
            devs.push(d);
        } else {
            break;
        }
    }
    let world = devs.len();
    if world < 2 {
        println!("Need at least 2 HIP devices; found {world}");
        return Ok(());
    }
    println!("World size = {world} devices");

    let cluster = std::sync::Arc::new(HipCluster::new(devs.clone())?);
    let devs_arc: std::sync::Arc<Vec<HipDevice>> = std::sync::Arc::new(devs);

    for n in [32usize, 2048, 65_536] {
        println!("\n== n = {n} floats ({} KB / rank) ==", n * 4 / 1024);

        // Pre-allocate per-rank input + output buffers in the main thread
        // so we can hand each worker thread a mutable reference.
        let mut inputs_main: Vec<_> = Vec::with_capacity(world);
        let mut outputs_main: Vec<_> = Vec::with_capacity(world);
        for r in 0..world {
            hipdarc::driver::HipDevice::new(devs_arc[r].ordinal())
                .unwrap()
                .set_current()
                .unwrap();
            let payload: Vec<f32> = (0..n)
                .map(|i| i as f32 + (r as f32) * 1000.0)
                .collect();
            let in_slab = devs_arc[r].stream().clone_htod(&payload).unwrap();
            let out_slab: hipdarc::driver::HipSlice<f32> =
                unsafe { devs_arc[r].alloc::<f32>(n)? };
            inputs_main.push(in_slab);
            outputs_main.push(out_slab);
        }
        for d in devs_arc.iter() {
            d.stream().synchronize().unwrap();
        }

        // Zip each rank's (input, output) so we can send exactly one
        // owned pair to each worker thread.
        let pairs: Vec<(
            hipdarc::driver::HipSlice<f32>,
            hipdarc::driver::HipSlice<f32>,
        )> = inputs_main.into_iter().zip(outputs_main.into_iter()).collect();

        let cluster_c = cluster.clone();
        let devs_c = devs_arc.clone();

        let t0 = std::time::Instant::now();
        let results: Vec<(
            usize,
            Vec<f32>,
            Result<(), candle_core::Error>,
        )> = std::thread::scope(|s| {
            let handles: Vec<_> = pairs
                .into_iter()
                .enumerate()
                .map(|(rank, (input, mut output))| {
                    let cluster = cluster_c.clone();
                    let devs = devs_c.clone();
                    s.spawn(move || {
                        // Each thread pins its HIP context first.
                        hipdarc::driver::HipDevice::new(devs[rank].ordinal())
                            .unwrap()
                            .set_current()
                            .unwrap();
                        let res = cluster.all_reduce_sum_f32_barrier(
                            rank, &input, &mut output,
                        );
                        // DtoH the rank's output for verification.
                        let got = devs[rank].stream().clone_dtoh(&output)
                            .map_err(|e| candle_core::Error::Msg(format!("dtoh: {e}")))
                            .unwrap_or_default();
                        (rank, got, res)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let elapsed = t0.elapsed();

        // Check results.
        let w = world as f32;
        let rank_sum: f32 = (0..world).map(|r| r as f32 * 1000.0).sum();
        let mut fails = 0usize;
        for (rank, got, res) in &results {
            res.as_ref().expect("all_reduce_sum_f32_barrier");
            for i in 0..n {
                let want = w * (i as f32) + rank_sum;
                if (got[i] - want).abs() > 1e-3 {
                    if fails < 4 {
                        println!(
                            "  rank {rank}, i {i}: got {} want {}",
                            got[i], want
                        );
                    }
                    fails += 1;
                }
            }
        }
        println!(
            "  barrier all-reduce: {} ranks × {} B, {:?}, fails={fails}",
            world,
            n * 4,
            elapsed
        );
        assert_eq!(fails, 0, "barrier all-reduce correctness");
    }

    println!("\nOK");
    Ok(())
}
