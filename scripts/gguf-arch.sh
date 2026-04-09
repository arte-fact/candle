#!/usr/bin/env bash
# Print the architecture and key metadata from a GGUF file.
# Usage: ./gguf-arch.sh model.gguf
set -euo pipefail
MODEL="${1:?Usage: $0 model.gguf}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CANDLE_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$CANDLE_ROOT"

# Build and run the gguf tokenizer tool which dumps metadata
cargo run --release --example gguf-tokenizer -- --gguf-file "$MODEL" 2>&1 | head -30
