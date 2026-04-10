#!/bin/bash
# Test Gemma4 quantized inference on HIP (AMD MI50 gfx906)
#
# Usage:
#   ./scripts/test-hip-gemma4.sh [model_path]
#
# Defaults to gemma-4-E4B if no model specified.
# Tokenizer is read from the GGUF file automatically.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

MODEL="${1:-/artefact/models/gemma-4-E4B-it-Q4_0.gguf}"

echo "=== Gemma4 HIP Test ==="
echo "Model: $MODEL"
echo ""

# Step 1: Run unit tests
echo "--- Unit tests ---"
cargo test -p candle-transformers --lib quantized_blocks --features hip 2>&1 | tail -5
echo ""

# Step 2: Build example
echo "--- Building quantized-gemma4 example ---"
cargo build --example quantized-gemma4 --release --features hip 2>&1 | tail -3
echo ""

# Step 3: Run inference (short, greedy)
echo "--- Running inference (10 tokens, greedy) ---"
./target/release/examples/quantized-gemma4 \
    --model "$MODEL" \
    --prompt "What is the capital of France?" \
    --sample-len 10 \
    --temperature 0

echo ""
echo "=== Test complete ==="
