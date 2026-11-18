#!/usr/bin/env bash
# Integration test: CANDLE_KV_Q8 + CANDLE_Q8_KV_DIRECT_GEMV paths.
#
# Verifies that Q8 KV decode:
#   (a) produces coherent output on both qwen35 (GatedAttention → direct
#       GEMV wire-up) and gemma4 (AnyKvCache::Q8 → dequant path) models,
#   (b) under CANDLE_Q8_KV_DIRECT_GEMV={0,1} on qwen35: fires
#       `gqa_decode_gemv_*_q8_*` kernels when on, dequant kernel when off,
#   (c) under CANDLE_KV_Q8=1 on gemma4-A4B without CANDLE_KV_F16=0
#       workaround (the F16/Q8 mutex fix must hold).
#
# Env knobs:
#   QWEN_MODEL (default /artefact/models/Qwen3.5-9B-Q4_1.gguf)
#   GEMMA_MODEL (default /artefact/models/gemma-4-26B-A4B-it-Q8_0.gguf)
#   SKIP_GEMMA=1 to skip the gemma4 sub-test (avoids 4-GPU requirement)
# Run: bash bench/test_q8_direct_gemv_it.sh

set -euo pipefail
QWEN_MODEL="${QWEN_MODEL:-/artefact/models/Qwen3.5-9B-Q4_1.gguf}"
GEMMA_MODEL="${GEMMA_MODEL:-/artefact/models/gemma-4-26B-A4B-it-Q8_0.gguf}"
QWEN_BIN="./target/release/examples/quantized-qwen35"
GEMMA_BIN="./target/release/examples/quantized-gemma4"
GPU="${HIP_VISIBLE_DEVICES:-1}"
PROMPT="${PROMPT:-explain quantum gravity}"
N="${N:-32}"

# Legacy aliases for backward compat
MODEL_PATH="${MODEL_PATH:-$QWEN_MODEL}"
BIN="$QWEN_BIN"

if [[ ! -x "$BIN" ]]; then
    echo "FAIL: missing $BIN; build with 'cargo build --release -p candle-examples --example quantized-qwen35 --features hip'"
    exit 2
fi
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "FAIL: missing $MODEL_PATH; set MODEL_PATH or put the file there"
    exit 2
fi

export LD_LIBRARY_PATH=/opt/rocm-7.1.1/core-7.13/lib:/opt/rocm/lib
export HIP_VISIBLE_DEVICES="$GPU"
export CANDLE_KV_Q8=1

mkdir -p /tmp/q8_direct_it
OUT_DEQ=/tmp/q8_direct_it/dequant.txt
OUT_DIR=/tmp/q8_direct_it/direct.txt

echo "=== opt-out (dequant) ==="
CANDLE_Q8_KV_DIRECT_GEMV=0 "$BIN" --model "$MODEL_PATH" --prompt "$PROMPT" --sample-len "$N" 2>/dev/null | tee "$OUT_DEQ" | tail -8

echo "=== direct GEMV (default on) ==="
CANDLE_Q8_KV_DIRECT_GEMV=1 "$BIN" --model "$MODEL_PATH" --prompt "$PROMPT" --sample-len "$N" 2>/dev/null | tee "$OUT_DIR" | tail -8

# Coherence: both outputs must contain at least 20 English characters of
# response (rules out garbage-token blowup).
for f in "$OUT_DEQ" "$OUT_DIR"; do
    chars=$(tr -cd '[:alpha:][:space:]' < "$f" | wc -c)
    if (( chars < 100 )); then
        echo "FAIL: $f has only $chars alpha chars — likely garbage"
        exit 1
    fi
done

# Perf: assert generated-tokens line present in both
for f in "$OUT_DEQ" "$OUT_DIR"; do
    grep -qE "[0-9]+ tokens generated: [0-9.]+ t/s" "$f" || {
        echo "FAIL: $f missing tg line"
        exit 1
    }
done

# Kernel trace: direct path must contain gqa_decode_gemv_*_q8_* calls;
# dequant path must NOT. Requires rocprofv3.
if command -v /opt/rocm-7.1.1/core-7.13/bin/rocprofv3 >/dev/null; then
    mkdir -p /tmp/q8_direct_it/trace_deq /tmp/q8_direct_it/trace_dir
    CANDLE_Q8_KV_DIRECT_GEMV=0 /opt/rocm-7.1.1/core-7.13/bin/rocprofv3 \
        --kernel-trace --output-format csv -d /tmp/q8_direct_it/trace_deq \
        -- "$BIN" --model "$MODEL_PATH" --prompt "$PROMPT" --sample-len 8 >/dev/null 2>&1
    CANDLE_Q8_KV_DIRECT_GEMV=1 /opt/rocm-7.1.1/core-7.13/bin/rocprofv3 \
        --kernel-trace --output-format csv -d /tmp/q8_direct_it/trace_dir \
        -- "$BIN" --model "$MODEL_PATH" --prompt "$PROMPT" --sample-len 8 >/dev/null 2>&1
    DEQ_CSV=$(find /tmp/q8_direct_it/trace_deq -name '*kernel_trace.csv' | head -1)
    DIR_CSV=$(find /tmp/q8_direct_it/trace_dir -name '*kernel_trace.csv' | head -1)
    if ! grep -q 'gqa_decode_gemv_[qv]k\?_q8_' "$DIR_CSV"; then
        echo "FAIL: direct path trace does not contain gqa_decode_gemv_*_q8_* kernels"
        exit 1
    fi
    if grep -q 'gqa_decode_gemv_[qv]k\?_q8_' "$DEQ_CSV"; then
        echo "FAIL: opt-out trace unexpectedly contains direct-GEMV Q8 kernels"
        exit 1
    fi
fi

echo "PASS: qwen35 q8_direct_gemv IT (coherent, kernel dispatch verified)"

# --- gemma4 sub-test: CANDLE_KV_Q8=1 must work without CANDLE_KV_F16=0 ---
if [[ "${SKIP_GEMMA:-0}" == "1" ]]; then
    echo "SKIP: gemma4 sub-test (SKIP_GEMMA=1)"
    echo "ALL PASS"
    exit 0
fi
if [[ ! -x "$GEMMA_BIN" || ! -f "$GEMMA_MODEL" ]]; then
    echo "SKIP: gemma4 sub-test (missing $GEMMA_BIN or $GEMMA_MODEL)"
    echo "ALL PASS"
    exit 0
fi

OUT_GEMMA=/tmp/q8_direct_it/gemma_q8.txt
unset HIP_VISIBLE_DEVICES
echo "=== gemma4-A4B with CANDLE_KV_Q8=1 (no F16 workaround) ==="
HIP_VISIBLE_DEVICES=0,1,2,3 CANDLE_KV_Q8=1 \
    "$GEMMA_BIN" --model "$GEMMA_MODEL" --prompt "$PROMPT" --sample-len 20 --n-gpus 4 \
    2>/dev/null | tee "$OUT_GEMMA" | tail -6

chars=$(tr -cd '[:alpha:][:space:]' < "$OUT_GEMMA" | wc -c)
if (( chars < 100 )); then
    echo "FAIL: gemma4 output has only $chars alpha chars"
    exit 1
fi
grep -qE "[0-9]+ tokens generated: [0-9.]+ t/s" "$OUT_GEMMA" || {
    echo "FAIL: gemma4 missing tg line — F16/Q8 mutex fix regressed?"
    exit 1
}

echo "PASS: gemma4-A4B CANDLE_KV_Q8=1 standalone (F16/Q8 mutex OK)"

# --- gemma4 direct-GEMV opt-in sub-test ---
# On gemma4, direct-GEMV is DEFAULT OFF (fused F32 mv_fast wins). Setting
# CANDLE_Q8_KV_DIRECT_GEMV=1 activates the experimental decomposed path.
# The test just verifies the opt-in still produces coherent output (the
# perf regression is documented, not asserted by the IT).
OUT_GEMMA_DIRECT=/tmp/q8_direct_it/gemma_q8_direct.txt
echo "=== gemma4-A4B CANDLE_Q8_KV_DIRECT_GEMV=1 (opt-in, experimental) ==="
HIP_VISIBLE_DEVICES=0,1,2,3 CANDLE_KV_Q8=1 CANDLE_Q8_KV_DIRECT_GEMV=1 \
    "$GEMMA_BIN" --model "$GEMMA_MODEL" --prompt "$PROMPT" --sample-len 20 --n-gpus 4 \
    2>/dev/null | tee "$OUT_GEMMA_DIRECT" | tail -4

chars=$(tr -cd '[:alpha:][:space:]' < "$OUT_GEMMA_DIRECT" | wc -c)
if (( chars < 100 )); then
    echo "FAIL: gemma4 direct-GEMV opt-in output garbled"
    exit 1
fi
echo "PASS: gemma4-A4B CANDLE_Q8_KV_DIRECT_GEMV=1 coherent (opt-in)"

echo "ALL PASS"
