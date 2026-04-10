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

# Two `#[ignore]`d variants:
#
#   capture_replay_kernel_only_floor   — chain of `hipMemsetAsync`
#       calls against a pre-allocated buffer. No alloc/free nodes
#       in the captured graph. Measures the floor: how fast can
#       graph replay be on this hardware when the graph is *just*
#       kernels?
#
#   capture_replay_chained_bench       — chain of candle Tensor
#       (matmul → relu) pairs. Each op allocates an output tensor
#       so the captured graph contains many `hipMallocAsync` /
#       `hipFreeAsync` nodes. Measures the realistic case for
#       capturing a candle forward as-is.
cargo test -p candle-core --features hip --release --lib \
    graph_smoke -- --nocapture --include-ignored 2>&1 | tail -50

echo
echo "=== bench complete ==="
echo
echo "Decision matrix:"
echo "  kernel_only_floor speedup  >>  chained_bench speedup  →"
echo "    The hardware *can* benefit from graph replay, but the candle"
echo "    Tensor model is allocating too many temporaries inside the"
echo "    captured forward. To unlock graphs, we'd need a pre-allocated"
echo "    intermediate buffer pool that the captured forward writes into,"
echo "    so the captured graph contains only kernel nodes."
echo
echo "  Both speedups < 1.5×  →"
echo "    HIP graphs are not the right lever on this hardware. Pivot"
echo "    to operator fusion / GQA-aware matmul instead."
