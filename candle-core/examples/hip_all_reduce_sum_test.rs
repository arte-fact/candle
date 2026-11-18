//! X4c.1 smoke test: HipCluster::all_reduce_sum_f32_via_host.
//!
//! Each rank r fills a device slice with a distinct pattern
//! (`buf[i] = i * 1.0 + r * 1000.0`). After the all-reduce, every
//! rank's output must equal the element-wise sum across all ranks.

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

    let cluster = HipCluster::new(devs.clone())?;

    // Sizes that span the MoE decode range and a stress case.
    for n in [32usize, 2048, 65_536, 524_288] {
        println!("\n== n = {n} floats ({} KB / rank) ==", n * 4 / 1024);

        // Build distinct patterns per rank.
        let mut inputs_owned: Vec<_> = Vec::with_capacity(world);
        let mut outputs_owned: Vec<_> = Vec::with_capacity(world);
        for r in 0..world {
            hipdarc::driver::HipDevice::new(devs[r].ordinal())
                .unwrap()
                .set_current()
                .unwrap();
            let payload: Vec<f32> = (0..n)
                .map(|i| i as f32 + (r as f32) * 1000.0)
                .collect();
            let in_slab = devs[r].stream().clone_htod(&payload).unwrap();
            let out_slab: hipdarc::driver::HipSlice<f32> =
                unsafe { devs[r].alloc::<f32>(n)? };
            inputs_owned.push(in_slab);
            outputs_owned.push(out_slab);
        }
        for d in &devs {
            d.stream().synchronize().unwrap();
        }

        let input_refs: Vec<&hipdarc::driver::HipSlice<f32>> =
            inputs_owned.iter().collect();
        let mut output_refs: Vec<&mut hipdarc::driver::HipSlice<f32>> =
            outputs_owned.iter_mut().collect();

        let t0 = std::time::Instant::now();
        cluster.all_reduce_sum_f32_via_host(&input_refs, &mut output_refs)?;
        let elapsed = t0.elapsed();

        // Expected: sum_r (i + r*1000) = W*i + 1000 * sum_r(r)
        let w = world as f32;
        let rank_sum: f32 = (0..world).map(|r| r as f32 * 1000.0).sum();
        let mut fails = 0usize;
        for r in 0..world {
            hipdarc::driver::HipDevice::new(devs[r].ordinal())
                .unwrap()
                .set_current()
                .unwrap();
            let got = devs[r].stream().clone_dtoh(&outputs_owned[r]).unwrap();
            for i in 0..n {
                let want = w * (i as f32) + rank_sum;
                if (got[i] - want).abs() > 1e-3 {
                    if fails < 4 {
                        println!(
                            "  rank {r}, i {i}: got {} want {}",
                            got[i], want
                        );
                    }
                    fails += 1;
                }
            }
        }
        println!(
            "  all-reduce: {} ranks × {} B, {:?}, fails={fails}",
            world,
            n * 4,
            elapsed
        );
        assert_eq!(fails, 0, "all_reduce_sum_f32_via_host correctness");
    }

    println!("\nOK");
    Ok(())
}
