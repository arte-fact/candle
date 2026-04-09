#!/usr/bin/env bash
# Compare CPU vs GPU output for the same prompt to find correctness issues.
set -euo pipefail
MODEL="${1:-/home/artefact/models/tinyllama-1.1b-q4_0.gguf}"
PROMPT="The capital of France is"

echo "=== CPU ==="
cargo run --release --example quantized -- \
  --model "$MODEL" --cpu --prompt "$PROMPT" --sample-len 10 --temperature 0 2>&1 | tail -5

echo ""
echo "=== GPU (--num-gpus 1) ==="
cargo run --release --features hip --example quantized -- \
  --model "$MODEL" --num-gpus 1 --prompt "$PROMPT" --sample-len 10 --temperature 0 2>&1 | tail -5
