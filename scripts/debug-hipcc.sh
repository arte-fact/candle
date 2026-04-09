#!/usr/bin/env bash
# Debug hipcc compilation errors for individual kernel files.
# Usage: ./scripts/debug-hipcc.sh [filename.cu]
# Default: unary.cu
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")/candle-hip-kernels/src"
FILE="${1:-unary.cu}"

echo "=== Compiling $FILE with hipcc (verbose) ==="
echo "Source dir: $SRC_DIR"
echo ""

cd "$SRC_DIR"
hipcc --cuda-device-only -c \
    --offload-arch=gfx906 \
    -std=c++17 -O3 \
    -I . \
    -D__HIP_PLATFORM_AMD__ \
    -DWARP_SIZE=64 \
    -o /tmp/test_kernel.hsaco \
    "$FILE" \
    2>&1

echo "Format check:"
file /tmp/test_kernel.hsaco

echo ""
echo "[OK] Compilation succeeded"
