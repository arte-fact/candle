#!/usr/bin/env bash
# Test quantized model inference on HIP/ROCm with available GGUF models.
# Requires: ROCm + gfx906 GPU, built candle with hip feature.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CANDLE_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${CANDLE_MODELS_DIR:-/home/artefact/models}"

echo "=== Candle HIP Quantized Inference Test ==="
echo "Models dir: $MODELS_DIR"

# Pick a compatible model — prefer smaller ones.
# Note: Qwen3.5 uses different metadata keys than Qwen3, skip for now.
MODEL=""
for candidate in \
    "$MODELS_DIR/gemma-4-E4B-it-Q4_0.gguf" \
    "$MODELS_DIR/gemma-4-31B-it-Q4_0.gguf" \
    "$MODELS_DIR/Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf" \
    "$MODELS_DIR/Qwen3.5-9B-Q4_1.gguf" \
    "$MODELS_DIR/Qwen3.5-27B-Q4_0.gguf" \
; do
    if [ -f "$candidate" ]; then
        MODEL="$candidate"
        break
    fi
done

if [ -z "$MODEL" ]; then
    echo "ERROR: No suitable GGUF model found in $MODELS_DIR"
    echo "Available files:"
    ls "$MODELS_DIR"/*.gguf 2>/dev/null || echo "  (none)"
    exit 1
fi

echo "Model: $MODEL"
echo ""

# Detect the right example based on model name
EXAMPLE="quantized"
case "$MODEL" in
    *[Qq]wen3*) EXAMPLE="quantized-qwen3" ;;
    *[Qq]wen2*) EXAMPLE="quantized-qwen2-instruct" ;;
    *gemma*)    EXAMPLE="quantized-gemma" ;;
esac
echo "Example: $EXAMPLE"
echo ""

# Build the example with hip feature
echo "=== Building $EXAMPLE ==="
cd "$CANDLE_ROOT"
cargo build --release --features hip --example "$EXAMPLE" 2>&1 | tail -5
echo "[OK] Build succeeded"
echo ""

# Run inference with a short prompt
echo "=== Running inference (10 tokens) ==="
cargo run --release --features hip --example "$EXAMPLE" -- \
    --model "$MODEL" \
    --prompt "Hello, I am a language model running on AMD GPU. " \
    --sample-len 10 \
    2>&1

echo ""
echo "=== Quantized inference test complete ==="
