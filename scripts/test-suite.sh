#!/usr/bin/env bash
# End-to-end test suite for candle quantized GGUF inference on MI50 (gfx906).
#
# Validates:
#   1. Multi-GPU pipeline-parallel infrastructure (Test 0)
#   2. gemma4 E4B at 1, 2, 4 GPUs                          (Tests 1-3)
#   3. qwen35-9B dense (GDN + GatedAttention + DenseMlp) at 1, 4 GPUs (Tests 4-5)
#   4. qwen35moe-35B MoE (GDN + GatedAttention + MoeExperts + shared expert + F16 mixed)
#      at 4 GPUs                                            (Test 6)
#   5. Unit tests for the modular blocks                   (Test 7)
#
# Each model run uses temperature=0 (deterministic argmax) so the exact output
# string is reproducible across runs.
#
# Required env (matches /artefact/candle/SESSION-RESUME.md driver setup):
#   ROCM_PATH, LD_LIBRARY_PATH, ROCBLAS_TENSILE_LIBPATH, HIP_OFFLOAD_ARCH

set -uo pipefail

# ---------------- ROCm 7.1.1 + gfx906 driver paths ----------------
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=/opt/rocm/core-7.13/lib:/opt/rocm/core-7.13/lib/rocm_sysdeps/lib:/opt/rocm/lib
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm/core-7.13/lib/rocblas/library
export HIP_OFFLOAD_ARCH=gfx906

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Pick the model paths — fall back to common locations.
GEMMA4="${GEMMA4:-/artefact/models/gemma-4-E4B-it-Q4_0.gguf}"
GEMMA4_31B_Q4="${GEMMA4_31B_Q4:-/artefact/models/gemma-4-31B-it-Q4_0.gguf}"
GEMMA4_31B_Q8="${GEMMA4_31B_Q8:-/artefact/models/gemma-4-31B-it-Q8_0.gguf}"
QWEN35_9B="${QWEN35_9B:-/artefact/models/Qwen3.5-9B-Q4_1.gguf}"
QWEN35_MOE="${QWEN35_MOE:-/artefact/models/qwen3.5-35b-a3b/Qwen3.5-35B-A3B-UD-Q8_K_XL.gguf}"
QWEN3NEXT="${QWEN3NEXT:-/artefact/models/Qwen3-Coder-Next-Q4_0.gguf}"
QWEN3MOE_30B="${QWEN3MOE_30B:-/artefact/models/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf}"
QWEN3NEXT_Q4_1_SPLIT="${QWEN3NEXT_Q4_1_SPLIT:-/artefact/models/Qwen3-Coder-Next-Q4_1-00001-of-00003.gguf}"

PASS=0
FAIL=0
RESULTS=()

bold() { printf '\033[1m%s\033[0m' "$*"; }
green() { printf '\033[32m%s\033[0m' "$*"; }
red() { printf '\033[31m%s\033[0m' "$*"; }
yellow() { printf '\033[33m%s\033[0m' "$*"; }

run_test() {
    local label="$1"; shift
    local needle="$1"; shift
    echo
    echo "$(bold "▶ $label")"
    echo "  cmd: $*"
    local out
    out=$(timeout 600 "$@" 2>&1)
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "$out" | tail -8
        echo "  $(red FAIL) (exit=$rc)"
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL  $label")
        return 1
    fi
    # Print the last few lines (model output + perf)
    echo "$out" | tail -10
    if echo "$out" | grep -q "$needle"; then
        echo "  $(green PASS) (matched: $needle)"
        PASS=$((PASS + 1))
        RESULTS+=("PASS  $label")
    else
        echo "  $(red FAIL) (expected to find: $needle)"
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL  $label")
    fi
}

echo "$(bold '═══════════════════════════════════════════════════════════════')"
echo "$(bold '  Candle quantized GGUF inference — full test suite (MI50/HIP)')"
echo "$(bold '═══════════════════════════════════════════════════════════════')"
echo
echo "ROCM_PATH=$ROCM_PATH"
echo "models:"
echo "  gemma4     = $GEMMA4"
echo "  qwen35_9B  = $QWEN35_9B"
echo "  qwen35_moe = $QWEN35_MOE"

# Sanity check files exist
for f in "$GEMMA4" "$GEMMA4_31B_Q4" "$GEMMA4_31B_Q8" "$QWEN35_9B" "$QWEN35_MOE" "$QWEN3NEXT" "$QWEN3MOE_30B" "$QWEN3NEXT_Q4_1_SPLIT"; do
    if [[ ! -f "$f" ]]; then
        echo "$(red MISSING): $f"
    fi
done

# ---------------- Build everything once ----------------
echo
echo "$(bold 'Building examples...')"
if cargo build --release --features hip \
    --example quantized-gemma4 \
    --example quantized-qwen35 \
    --example hip_multi_gpu_test 2>&1 | tail -5
then
    echo "$(green 'build OK')"
else
    echo "$(red 'BUILD FAILED — aborting')"
    exit 1
fi

# ---------------- Test 0: multi-GPU + RCCL infrastructure ----------------
run_test "T0: multi-GPU detection + RCCL AllReduce" \
    "All multi-GPU tests passed" \
    ./target/release/examples/hip_multi_gpu_test

# ---------------- Test 1-3: gemma4 E4B at 1/2/4 GPUs ----------------
run_test "T1: gemma4 E4B (1 GPU) — 'Hello, how are you?'" \
    "I'm doing well" \
    ./target/release/examples/quantized-gemma4 \
        --model "$GEMMA4" \
        --prompt "Hello, how are you?" \
        --sample-len 30 --temperature 0

run_test "T2: gemma4 E4B (2 GPU split) — '7 times 9'" \
    "63" \
    ./target/release/examples/quantized-gemma4 \
        --model "$GEMMA4" --n-gpus 2 \
        --prompt "What is 7 times 9?" \
        --sample-len 20 --temperature 0

run_test "T3: gemma4 E4B (4 GPU split) — 'Hello'" \
    "I'm doing well" \
    ./target/release/examples/quantized-gemma4 \
        --model "$GEMMA4" --n-gpus 4 \
        --prompt "Hello, how are you?" \
        --sample-len 30 --temperature 0

# ---------------- Test 4-5: qwen35-9B at 1/4 GPUs ----------------
run_test "T4: qwen35-9B dense (1 GPU) — 'What is 2+2?'" \
    "equals \*\*4\*\*" \
    ./target/release/examples/quantized-qwen35 \
        --model "$QWEN35_9B" \
        --prompt "What is 2+2?" \
        --sample-len 20 --temperature 0

run_test "T5: qwen35-9B dense (4 GPU split) — 'What is 2+2?'" \
    "equals \*\*4\*\*" \
    ./target/release/examples/quantized-qwen35 \
        --model "$QWEN35_9B" --n-gpus 4 \
        --prompt "What is 2+2?" \
        --sample-len 20 --temperature 0

# ---------------- Test 6: qwen35moe-35B-A3B at 4 GPUs ----------------
run_test "T6: qwen35moe-35B-A3B-Q8_K_XL (4 GPU, ~42 GB) — chat-format reasoning" \
    "Thinking Process" \
    ./target/release/examples/quantized-qwen35 \
        --model "$QWEN35_MOE" --n-gpus 4 \
        --prompt "What is 2+2?" \
        --sample-len 30 --temperature 0

# ---------------- Test 10: qwen3next Coder (MoE + GDN + ssm_ba) at 4 GPUs ----------------
run_test "T10: qwen3next Coder Q4_0 (4 GPU, ~45 GB) — 'What is 2+2?'" \
    "2 + 2" \
    ./target/release/examples/quantized-qwen35 \
        --model "$QWEN3NEXT" --n-gpus 4 \
        --prompt "What is 2+2?" \
        --sample-len 30 --temperature 0

# ---------------- Test 11: qwen3moe (Qwen3-Coder-30B Q4_K_XL) at 4 GPUs ----------------
# Multi-token prefill (no --split-prompt) — exercises the dequant+gemm fallback
# in QHipStorage::dequantize_matmul that replaces the broken mul_mat_via_q8_1
# integer-MMQ path on gfx906 (task #22).
run_test "T11: qwen3moe Coder-30B Q4_K_XL (4 GPU, ~17.6 GB) — 'What is 2+2?'" \
    "= 4" \
    ./target/release/examples/quantized-qwen35 \
        --model "$QWEN3MOE_30B" --n-gpus 4 \
        --prompt "What is 2+2?" \
        --sample-len 20 --temperature 0

# ---------------- Test 12: qwen3next Q4_1 split GGUF (3 files, ~50 GB) at 4 GPUs ----------------
run_test "T12: qwen3next Coder Q4_1 SPLIT (3 files, 4 GPU, ~50 GB)" \
    "4" \
    ./target/release/examples/quantized-qwen35 \
        --model "$QWEN3NEXT_Q4_1_SPLIT" --n-gpus 4 \
        --prompt "What is 2+2?" \
        --sample-len 30 --temperature 0

# ---------------- Test 8-9: gemma4-31B at 4 GPUs (large dense) ----------------
run_test "T8: gemma4-31B-Q4_0 (4 GPU, ~17 GB) — 'Hello'" \
    "I'm doing well" \
    ./target/release/examples/quantized-gemma4 \
        --model "$GEMMA4_31B_Q4" --n-gpus 4 \
        --prompt "Hello, how are you?" \
        --sample-len 30 --temperature 0

run_test "T9: gemma4-31B-Q8_0 (4 GPU, ~32 GB) — '7 times 9'" \
    "63" \
    ./target/release/examples/quantized-gemma4 \
        --model "$GEMMA4_31B_Q8" --n-gpus 4 \
        --prompt "What is 7 times 9?" \
        --sample-len 25 --temperature 0

# ---------------- Test 7: unit tests ----------------
echo
echo "$(bold '▶ T7: cargo test (quantized_blocks unit tests)')"
if cargo test --release -p candle-transformers --lib quantized_blocks 2>&1 | tail -15
then
    echo "  $(green PASS)"
    PASS=$((PASS + 1))
    RESULTS+=("PASS  T7: cargo test quantized_blocks")
else
    echo "  $(red FAIL)"
    FAIL=$((FAIL + 1))
    RESULTS+=("FAIL  T7: cargo test quantized_blocks")
fi

# ---------------- Summary ----------------
echo
echo "$(bold '═══════════════════════════════════════════════════════════════')"
echo "$(bold '  SUMMARY')"
echo "$(bold '═══════════════════════════════════════════════════════════════')"
for r in "${RESULTS[@]}"; do
    if [[ "$r" == PASS* ]]; then
        echo "  $(green "$r")"
    else
        echo "  $(red "$r")"
    fi
done
echo
echo "  $(bold "Total: $PASS pass, $FAIL fail")"

if [[ $FAIL -eq 0 ]]; then
    exit 0
else
    exit 1
fi
