//! X2 exit criterion: variable-length all-to-all on 4× MI50.
//!
//! Each rank sends a uniquely-tagged payload (bytes [rank, dst, seq, ...])
//! to every other rank. We verify every recv buffer contains the expected
//! (src, dst) tag and bench both small-payload (25 KB/pair, EP-dispatch
//! shape) and medium-payload (4 MiB/pair) throughput.

use anyhow::Result;
use candle_core::hip_backend::cluster::HipCluster;
use candle_core::{Device, HipDevice};
use hipdarc::driver::HipSlice;

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

    // --- Correctness sweep: 256 B per (src,dst) pair, tagged.
    const CORR_BYTES: usize = 256;
    println!("\n== correctness ({} B per src-dst pair) ==", CORR_BYTES);

    let mut send_tables: Vec<Vec<HipSlice<u8>>> = Vec::with_capacity(world);
    for src in 0..world {
        hipdarc::driver::HipDevice::new(devs[src].ordinal()).unwrap().set_current().unwrap();
        let mut row = Vec::with_capacity(world);
        for dst in 0..world {
            let mut payload = vec![0u8; CORR_BYTES];
            payload[0] = src as u8;
            payload[1] = dst as u8;
            for k in 2..CORR_BYTES {
                payload[k] = ((src * 37 + dst * 13 + k) % 251) as u8;
            }
            let slab = devs[src].stream().clone_htod(&payload).unwrap();
            row.push(slab);
        }
        send_tables.push(row);
    }
    for d in &devs {
        d.stream().synchronize().unwrap();
    }

    let recv = cluster.all_to_all_varlen_single_process(&send_tables)?;

    let mut fails = 0usize;
    for dst in 0..world {
        hipdarc::driver::HipDevice::new(devs[dst].ordinal()).unwrap().set_current().unwrap();
        for src in 0..world {
            let got = devs[dst].stream().clone_dtoh(&recv[dst][src]).unwrap();
            if got.len() < 2 {
                eprintln!("  [{src}->{dst}] FAIL: got.len()={}", got.len());
                fails += 1;
                continue;
            }
            if got[0] != src as u8 || got[1] != dst as u8 {
                eprintln!(
                    "  [{src}->{dst}] FAIL: tag=({}, {}), want=({src}, {dst})",
                    got[0], got[1]
                );
                fails += 1;
                continue;
            }
            for k in 2..CORR_BYTES {
                let expected = ((src * 37 + dst * 13 + k) % 251) as u8;
                if got[k] != expected {
                    eprintln!(
                        "  [{src}->{dst}] FAIL at byte {k}: got={}, want={expected}",
                        got[k]
                    );
                    fails += 1;
                    break;
                }
            }
        }
    }
    if fails == 0 {
        println!(
            "  all {}×{} = {} src-dst pairs OK",
            world, world, world * world
        );
    } else {
        anyhow::bail!("{fails} all-to-all correctness failures");
    }

    // --- Bench 1: EP dispatch-size (25 KB per pair).
    const EP_BYTES: usize = 25 * 1024;
    const EP_ITERS: usize = 50;
    println!(
        "\n== bench EP-dispatch size: {} KiB per pair, {} iters ==",
        EP_BYTES / 1024,
        EP_ITERS
    );
    bench_shape(&cluster, &devs, EP_BYTES, EP_ITERS)?;

    // --- Bench 2: medium payload (4 MiB per pair).
    const MED_BYTES: usize = 4 * 1024 * 1024;
    const MED_ITERS: usize = 5;
    println!(
        "\n== bench medium payload: {} MiB per pair, {} iters ==",
        MED_BYTES >> 20,
        MED_ITERS
    );
    bench_shape(&cluster, &devs, MED_BYTES, MED_ITERS)?;

    Ok(())
}

fn bench_shape(
    cluster: &HipCluster,
    devs: &[HipDevice],
    bytes: usize,
    iters: usize,
) -> Result<()> {
    let world = devs.len();
    // Allocate send matrix once.
    let mut send_tables: Vec<Vec<HipSlice<u8>>> = Vec::with_capacity(world);
    for src in 0..world {
        hipdarc::driver::HipDevice::new(devs[src].ordinal()).unwrap().set_current().unwrap();
        let mut row = Vec::with_capacity(world);
        for _dst in 0..world {
            let slab = unsafe { devs[src].alloc::<u8>(bytes).unwrap() };
            row.push(slab);
        }
        send_tables.push(row);
    }

    // Warm up.
    let _ = cluster.all_to_all_varlen_single_process(&send_tables)?;

    let mut times = Vec::new();
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        let _ = cluster.all_to_all_varlen_single_process(&send_tables)?;
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[iters / 2];
    // Total bytes transferred: W*(W-1) off-diagonal pairs × bytes.
    let off_diag = (world * (world - 1)) as f64;
    let agg_gbps = (off_diag * bytes as f64 / med) / 1e9;
    println!(
        "  median {:.3} ms   aggregate {:.2} GB/s across {} peer links",
        med * 1e3,
        agg_gbps,
        world * (world - 1),
    );
    Ok(())
}
