#!/usr/bin/env bash
# P0 — Fused gated_delta_net HIP kernel.
#
# Validates the Phase 1 fused kernel (S_v=128, f32, KDA=false) against
# the CPU tensor-op reference and measures the end-to-end decode
# speedup on qwen35-9B.
#
# Run on the ROCm box — the sandbox doesn't have libhip so the HIP
# tests are linker-gated out there.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CANDLE_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$CANDLE_ROOT"

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/core-7.13/lib:${ROCM_PATH}/core-7.13/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}"
export ROCBLAS_TENSILE_LIBPATH="${ROCBLAS_TENSILE_LIBPATH:-${ROCM_PATH}/core-7.13/lib/rocblas/library}"
export HIP_OFFLOAD_ARCH="${HIP_OFFLOAD_ARCH:-gfx906}"
export RUST_MIN_STACK="${RUST_MIN_STACK:-16777216}"

MODEL="${MODEL:-/artefact/models/Qwen3.5-9B-Q4_1.gguf}"

echo "=== P0: fused gated_delta_net kernel ==="
echo "Model: $MODEL"
echo

# ---------------------------------------------------------------------
# 1. Correctness: CPU tensor-op reference vs fused HIP kernel.
#    Two tests cover:
#      - single recurrent step at B=1, H=2, L=1, S_v=128
#      - 5 sequential steps with compounding state at B=1, H=4, S_v=128
#    Both compare attn output and updated state element-wise within
#    FMA-level tolerance (1e-4 single-step, 5e-4 multi-step).
# ---------------------------------------------------------------------
echo "--- Unit tests: CPU reference vs fused HIP kernel ---"
cargo test -p candle-transformers --features hip --release \
    quantized_blocks::delta_net::tests::hip_gated_delta_net_step_matches_cpu \
    -- --nocapture 2>&1 | tail -30
echo

# ---------------------------------------------------------------------
# 2. Qwen3.5-9B short-prompt decode. With only the GDN fast path on
#    (P0 Phase 1), each GDN step still runs L=1 per call — so the win
#    comes from collapsing ~8 tensor-op launches into 1 fused kernel,
#    not from amortizing the state load/store across L tokens (that's
#    Phase 2).
#
#    Expected baseline (post-pool, pre-P0): ~35 t/s tg128 on qwen35-9B.
#    Expected lift: ~15-25 % (1.15-1.25×) — fewer launches, same kernel
#    work. The bigger 2.5-3× win lands in Phase 2 when forward_prefill
#    batches q/k/v/gate/beta across L tokens.
# ---------------------------------------------------------------------
echo "--- Decode benchmark: qwen35-9B, sample-len 128, 1 GPU ---"
cargo build -p candle-examples --features hip --release \
    --example quantized-qwen35 2>&1 | tail -3
echo
./target/release/examples/quantized-qwen35 \
    --model "$MODEL" \
    --prompt "Hello, how are you?" \
    --sample-len 128 \
    --temperature 0 2>&1 | tail -15

echo
echo "=== complete ==="
echo
echo "Compare decode 't/s' against the pre-P0 run on the same model:"
echo "  pre-P0 baseline: ~35 t/s tg128 qwen35-9B"
echo "  post-P0 Phase 1: expect +15-25 % (i.e. ~40-44 t/s)"
echo "  post-P0 Phase 2: expect +150-200 % (i.e. ~90+ t/s)  — not yet"
echo
echo "Also spot-check the generated text for coherence — the kernel"
echo "correctness tests above verify math parity with the CPU path to"
echo "1e-4, but string-level assertions are in test-suite.sh T4/T5 and"
echo "rely on a specific Tensile kernel selection so flakes there are"
echo "expected."
