#!/usr/bin/env bash
# Smoke test for HIP backend: basic tensor operations.
# Run on a machine with ROCm + gfx906 GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CANDLE_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Candle HIP Backend Smoke Test ==="
echo "ROCm path: ${ROCM_PATH:-/opt/rocm}"
echo ""

# 1. Check ROCm is available
if ! command -v hipcc &>/dev/null; then
    echo "ERROR: hipcc not found. Ensure ROCm is installed and in PATH."
    echo "  export ROCM_PATH=/opt/rocm"
    echo "  export PATH=\$ROCM_PATH/bin:\$PATH"
    exit 1
fi
echo "[OK] hipcc found: $(which hipcc)"
echo ""
rocm-smi 2>/dev/null | head -10 || true
echo ""

# 2. Build candle-core with hip feature
echo "=== Building candle-core with hip feature ==="
cd "$CANDLE_ROOT"
cargo build --features hip -p candle-core 2>&1 | tail -5
echo "[OK] Build succeeded"
echo ""

# 3. Run unit tests (CPU comparison tests also exercise backend dispatch)
echo "=== Running candle-core tests with hip feature ==="
cargo test --features hip -p candle-core -- --test-threads=1 2>&1 | tail -20
echo ""

echo "=== Smoke test complete ==="
