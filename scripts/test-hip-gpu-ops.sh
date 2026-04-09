#!/usr/bin/env bash
# Test HIP GPU tensor operations directly.
# This validates that the HIP backend actually runs kernels on the MI50.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CANDLE_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Candle HIP GPU Tensor Ops Test ==="
cd "$CANDLE_ROOT"

# Build and run the hip_basics example
cargo build --release --features hip --example hip_basics 2>&1 | tail -3
echo "[OK] Build succeeded"
echo ""

cargo run --release --features hip --example hip_basics 2>&1

echo ""
echo "=== GPU tensor ops test complete ==="
