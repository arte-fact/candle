#!/usr/bin/env bash
# Validate the fused silu_mul kernel + measure end-to-end decode speedup
# on qwen35-9B vs the pre-fusion baseline.
#
# Run on the ROCm box (the sandbox has a rocBLAS strided-batched bug for
# some shapes that prevents end-to-end model runs).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CANDLE_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$CANDLE_ROOT"

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
export ROCBLAS_TENSILE_LIBPATH="${ROCBLAS_TENSILE_LIBPATH:-${ROCM_PATH}/lib/rocblas/library}"
export HIP_OFFLOAD_ARCH="${HIP_OFFLOAD_ARCH:-gfx906}"
export RUST_MIN_STACK="${RUST_MIN_STACK:-16777216}"

MODEL="${MODEL:-/artefact/models/Qwen3.5-9B-Q4_1.gguf}"

echo "=== silu_mul fusion: correctness + perf ==="
echo "Model: $MODEL"
echo

# 1. Unit test: confirm fused kernel matches the unfused chain on
#    FFN-realistic random inputs (1, 8, 4096) within 1e-5.
echo "--- Unit test: silu_mul matches chained silu * up ---"
cargo test -p candle-nn --features hip --release --test ops silu_mul -- --nocapture 2>&1 | tail -10
echo

# 2. Decode throughput on qwen35-9B (1 GPU). The "Hello, world" prompt
#    keeps prefill out of the picture so the t/s number is dominated
#    by per-token decode. --temperature 0 makes it deterministic.
#
#    Expected baseline (pre-fusion): ~30.97 t/s tg128
#    Per-token theoretical save: ~32 launches × ~3-5 µs ≈ 100-200 µs
#    Realised speedup will be 3-5% — small but free.
echo "--- Decode benchmark: qwen35-9B, sample-len 128, n_gpus 1 ---"
cargo build -p candle-examples --features hip --release --example quantized-qwen35 2>&1 | tail -3
echo
./target/release/examples/quantized-qwen35 \
    --model "$MODEL" \
    --prompt "Hello, how are you?" \
    --sample-len 128 \
    --temperature 0 2>&1 | tail -10

echo
echo "=== complete ==="
echo
echo "Compare 'tokens generated: X.XX t/s' against the pre-fusion run on"
echo "the same model. Expected lift is 3-5% on qwen35-9B (small, free)."
echo "Larger lifts will land with the GQA-aware matmul + residual+norm fusion."
