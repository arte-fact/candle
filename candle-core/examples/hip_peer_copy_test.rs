//! X1 exit criterion: 4-way peer-copy bandwidth on MI50 Gen3 x16.
//!
//! Allocates a 1 MiB buffer on each GPU, writes a known byte pattern,
//! peer-copies it to every other GPU, verifies contents, then benches
//! aggregate bandwidth with a larger payload. Target: ~12 GB/s aggregate
//! (close to Gen3 x16 ceiling of 16 GB/s usable per link per direction).

use anyhow::Result;
use candle_core::backend::BackendDevice;
use candle_core::hip_backend::cluster::HipCluster;
use candle_core::{Device, HipDevice};

const VERIFY_BYTES: usize = 1 << 20;      // 1 MiB correctness sweep
const BENCH_BYTES:  usize = 64 << 20;     // 64 MiB bandwidth bench
const BENCH_ITERS:  usize = 10;

fn main() -> Result<()> {
    // Build HipDevice handles for every detected GPU.
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
        println!("Need at least 2 HIP devices for this test; found {world}");
        return Ok(());
    }
    println!("World size = {world} devices");

    let cluster = HipCluster::new(devs.clone())?;

    // --- Correctness sweep: byte pattern round-trip across every ordered
    //     (src, dst) pair with src != dst.
    println!("\n== correctness sweep ({} B per copy) ==", VERIFY_BYTES);
    let pattern: Vec<u8> = (0..VERIFY_BYTES).map(|i| (i as u8).wrapping_mul(31)).collect();
    let mut fails = 0usize;

    // Seed src[r] on rank r with a per-rank-tagged copy of `pattern`.
    let src_bufs: Vec<_> = devs
        .iter()
        .enumerate()
        .map(|(r, d)| {
            let mut p = pattern.clone();
            p[0] = r as u8;
            d.stream().clone_htod(&p).unwrap()
        })
        .collect();
    let mut dst_bufs: Vec<_> = devs
        .iter()
        .map(|d| unsafe { d.alloc::<u8>(VERIFY_BYTES).unwrap() })
        .collect();
    // `clone_htod` is async — drain every default stream so the
    // pattern is actually on the device before peer copies start.
    for d in &devs {
        d.stream().synchronize().unwrap();
    }

    for src_rank in 0..world {
        for dst_rank in 0..world {
            if src_rank == dst_rank { continue; }
            // Use blocking peer copy for correctness test — simpler,
            // no stream ordering to worry about.
            cluster.peer_copy_blocking(
                &mut dst_bufs[dst_rank],
                dst_rank,
                &src_bufs[src_rank],
                src_rank,
            )?;
            // Set current device to dst_rank and fully sync before read.
            hipdarc::driver::HipDevice::new(devs[dst_rank].ordinal())
                .unwrap()
                .synchronize()
                .unwrap();
            let got = devs[dst_rank].stream().clone_dtoh(&dst_bufs[dst_rank]).unwrap();
            let tag = got[0];
            if tag != src_rank as u8 {
                eprintln!("  [{src_rank}->{dst_rank}] FAIL: tag={tag} != {src_rank}");
                fails += 1;
                continue;
            }
            // Spot-check a few positions beyond the tag.
            for i in [1usize, 4096, 1 << 15, VERIFY_BYTES - 1] {
                let expected = (i as u8).wrapping_mul(31);
                if got[i] != expected {
                    eprintln!(
                        "  [{src_rank}->{dst_rank}] FAIL at byte {i}: got={} != {expected}",
                        got[i]
                    );
                    fails += 1;
                    break;
                }
            }
        }
    }
    if fails == 0 {
        println!("  all {}×{} (src,dst) pairs OK", world, world - 1);
    } else {
        anyhow::bail!("{fails} peer-copy correctness failures");
    }

    // --- Bandwidth bench: each iter fans-out source rank 0's buffer to
    //     every other rank simultaneously. Aggregate GB/s = total bytes
    //     transferred / wall time.
    println!("\n== bandwidth bench ({} MiB per copy, {} iters) ==", BENCH_BYTES >> 20, BENCH_ITERS);
    let src_big = unsafe { devs[0].alloc::<u8>(BENCH_BYTES).unwrap() };
    let mut dst_bigs: Vec<_> = devs
        .iter()
        .skip(1)
        .map(|d| unsafe { d.alloc::<u8>(BENCH_BYTES).unwrap() })
        .collect();

    // Warm up the streams once.
    for (j, dst) in dst_bigs.iter_mut().enumerate() {
        let dst_rank = j + 1;
        cluster.peer_copy_async(dst, dst_rank, &src_big, 0)?;
    }
    cluster.sync_all()?;

    let mut times = Vec::new();
    for _ in 0..BENCH_ITERS {
        let t0 = std::time::Instant::now();
        for (j, dst) in dst_bigs.iter_mut().enumerate() {
            let dst_rank = j + 1;
            cluster.peer_copy_async(dst, dst_rank, &src_big, 0)?;
        }
        cluster.sync_all()?;
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[BENCH_ITERS / 2];
    let per_link_gbps = (BENCH_BYTES as f64 / med) / 1e9;
    let agg_gbps = per_link_gbps * (world - 1) as f64;
    println!(
        "  median {:.2} ms per fan-out of {} peers × {} MiB",
        med * 1e3,
        world - 1,
        BENCH_BYTES >> 20,
    );
    println!(
        "  per-link bandwidth: {:.2} GB/s; aggregate across {} links: {:.2} GB/s",
        per_link_gbps,
        world - 1,
        agg_gbps
    );
    // Gen3 x16 theoretical peak per direction = ~15.75 GB/s; practical ~12 GB/s.
    // We expect per-link close to that when the PCIe root complex is clear.
    println!(
        "  (reference: Gen3 x16 effective ≈ 12 GB/s per link direction on MI50)"
    );

    Ok(())
}
