#!/bin/bash
# Test Qwen3.5 quantized inference on HIP (AMD MI50 gfx906)
#
# Usage:
#   ./scripts/test-hip-qwen35.sh [model_path]
#
# Defaults to Qwen3.5-9B-Q4_1 if no model specified.
# Tokenizer is read from the GGUF file automatically.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

MODEL="${1:-/artefact/models/Qwen3.5-9B-Q4_1.gguf}"

echo "=== Qwen3.5 HIP Test ==="
echo "Model: $MODEL"
echo ""

# Step 1: Run unit tests
echo "--- Unit tests ---"
cargo test -p candle-transformers --lib quantized_blocks --features hip 2>&1 | tail -5
echo ""

# Step 2: Build example
echo "--- Building quantized-qwen35 example ---"
cargo build --example quantized-qwen35 --release --features hip 2>&1 | tail -3
echo ""

# Step 3: Run inference (short, greedy)
echo "--- Running inference (10 tokens, greedy) ---"
./target/release/examples/quantized-qwen35 \
    --model "$MODEL" \
    --prompt "What is 2+2?" \
    --sample-len 10 \
    --temperature 0 \
    --split-prompt

echo ""
echo "=== Test complete ==="
