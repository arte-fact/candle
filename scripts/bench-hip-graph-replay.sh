#!/usr/bin/env bash
# Benchmark HIP graph replay vs back-to-back fresh kernel launches.
#
# Validates the central HIP-graphs hypothesis: that for short, launch-bound
# sequences on gfx906/ROCm 7.1.1, capturing the sequence into a graph and
# replaying it amortises the per-launch driver overhead enough to
# materially close the decode gap with llamacpp-turbo.
#
# The candle-side estimate going in is ~960 launches × ~30 µs ≈ ~29 ms of
# pure launch cost per qwen35-9B decode token, out of ~32 ms total. If
# replay is even ~5× faster than fresh launches on the same chain, the
# multi-day model-side integration is worth the investment.
#
# Usage (run on the ROCm box, not the sandbox):
#   ./scripts/bench-hip-graph-replay.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CANDLE_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$CANDLE_ROOT"

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
export ROCBLAS_TENSILE_LIBPATH="${ROCBLAS_TENSILE_LIBPATH:-${ROCM_PATH}/lib/rocblas/library}"
export HIP_OFFLOAD_ARCH="${HIP_OFFLOAD_ARCH:-gfx906}"
export RUST_MIN_STACK="${RUST_MIN_STACK:-16777216}"

echo "=== HIP graph replay benchmark ==="
echo "ROCm path: $ROCM_PATH"
echo

# Build the test binary first so the test launch only does the work.
cargo test -p candle-core --features hip --release --lib graph_smoke --no-run 2>&1 | tail -3
echo

# The bench is `#[ignore]` so it only runs with --ignored. The test
# itself prints baseline / replay / speedup on stderr — capture both
# stdout and stderr so the user sees everything.
cargo test -p candle-core --features hip --release --lib \
    capture_replay_chained_bench -- --nocapture --ignored 2>&1 | tail -40

echo
echo "=== bench complete ==="
echo
echo "If 'speedup' is >= 3×: the launch-overhead theory is correct, and the"
echo "  multi-day model-side graph integration is worth investing in."
echo "If speedup is < 1.5×: the bottleneck is somewhere else (kernel work,"
echo "  rocBLAS dispatch, or memory traffic) and graphs are the wrong lever."
