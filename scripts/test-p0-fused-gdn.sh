#!/usr/bin/env bash
# P0 — Fused gated_delta_net HIP kernel (Phase 1 + Phase 2).
#
# Phase 1: validates the fused kernel (S_v=128, f32, KDA=false) at L=1
#          against the CPU tensor-op reference.
# Phase 2: validates the kernel's internal n_tokens loop by comparing a
#          single L=5 batched call to five L=1 calls, and measures the
#          end-to-end qwen35-9B prefill / decode speedup.
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
# 1. Correctness: CPU tensor-op reference vs fused HIP kernel, and the
#    kernel's internal n_tokens loop vs an external single-step loop.
#    Three tests cover:
#      - hip_gated_delta_net_step_matches_cpu_s128_single_step
#        (Phase 1: L=1, B=1, H=2; 1e-4 tolerance)
#      - hip_gated_delta_net_step_matches_cpu_s128_multi_step
#        (Phase 1: 5 sequential L=1 calls with compounding state,
#         B=1, H=4; 5e-4 tolerance)
#      - hip_gated_delta_net_batched_matches_looped_s128_l5
#        (Phase 2: one L=5 batched call vs five L=1 calls through the
#         same kernel; 1e-5 tolerance — same math, same hardware)
# ---------------------------------------------------------------------
echo "--- Unit tests: fused kernel correctness ---"
cargo test -p candle-transformers --features hip --release \
    quantized_blocks::delta_net::tests::hip_gated_delta_net \
    -- --nocapture 2>&1 | tail -40
echo

# ---------------------------------------------------------------------
# 2. Qwen3.5-9B — both prefill (long prompt) and decode (short prompt)
#    benchmarks. Phase 2's batched forward_prefill is where the large
#    speedup lives: the GDN recurrence is now one kernel launch per
#    layer instead of ~8 × seq_len launches, and the (S_v, S_v) state
#    stays register-resident across every prompt token.
# ---------------------------------------------------------------------
echo "--- Building quantized-qwen35 ---"
cargo build -p candle-examples --features hip --release \
    --example quantized-qwen35 2>&1 | tail -3
echo

# Long prompt → prefill-dominated. Expected pre-P0: pp ~78 t/s.
# Post-P0-Phase-2 projection: pp ~200-300 t/s on qwen35-9B 1GPU.
PROMPT_LONG="$(printf 'Explain in detail how gated delta net linear attention works, including the state update equations, the role of the decay gate, the chunked delta-net prefill algorithm, and how it differs from standard quadratic attention. %s' 'Focus on the mathematical formulation and numerical stability of the recurrence.')"
echo "--- Prefill benchmark: qwen35-9B, long prompt, sample-len 64, 1 GPU ---"
./target/release/examples/quantized-qwen35 \
    --model "$MODEL" \
    --prompt "$PROMPT_LONG" \
    --sample-len 64 \
    --temperature 0 2>&1 | tail -15
echo

# Short prompt → decode-dominated.
echo "--- Decode benchmark: qwen35-9B, short prompt, sample-len 128, 1 GPU ---"
./target/release/examples/quantized-qwen35 \
    --model "$MODEL" \
    --prompt "Hello, how are you?" \
    --sample-len 128 \
    --temperature 0 2>&1 | tail -15

# A/B check: force the legacy per-token forward_step path for one run
# so we can compare against the new batched forward_prefill.
echo
echo "--- A/B: CANDLE_GDN_PER_TOKEN=1 (legacy per-token, same prompt) ---"
CANDLE_GDN_PER_TOKEN=1 ./target/release/examples/quantized-qwen35 \
    --model "$MODEL" \
    --prompt "$PROMPT_LONG" \
    --sample-len 64 \
    --temperature 0 2>&1 | tail -15

echo
echo "=== complete ==="
echo
echo "Compare pp (prefill) and tg (decode) t/s against the pre-P0 baseline:"
echo "  pre-P0  baseline:   pp ~78 t/s  / tg ~35 t/s    (qwen35-9B 1GPU)"
echo "  post-P0 Phase 1:    pp ~90-120 t/s / tg ~40-44 t/s  (L=1 fused only)"
echo "  post-P0 Phase 2:    pp ~200-300 t/s / tg ~75-90 t/s (batched prefill)"
echo "  turbo-reference:    pp  450 t/s / tg  58 t/s"
echo
echo "The A/B run with CANDLE_GDN_PER_TOKEN=1 gives the legacy per-token"
echo "path as a baseline for the same prompt — expect it to be ~3× slower"
echo "on prefill than the new batched path."
echo
echo "Also spot-check the generated text for coherence. Kernel"
echo "correctness is verified by the three unit tests above; string-level"
echo "assertions in test-suite.sh T4/T5 depend on rocBLAS Tensile kernel"
echo "selection and are known to flake across Phase 1 / Phase 2 switches."
