#!/usr/bin/env bash
# Compile each HIP kernel individually to find all errors.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")/candle-hip-kernels/src"
OUT_DIR="/tmp/hipcc_debug"
mkdir -p "$OUT_DIR"

PASS=0
FAIL=0
FAILED_FILES=""

cd "$SRC_DIR"

for f in *.cu; do
    echo -n "  $f ... "
    OUTPUT=$(hipcc --cuda-device-only -c \
        --offload-arch=gfx906 \
        -std=c++17 -O3 \
        -I . \
        -D__HIP_PLATFORM_AMD__ \
        -DWARP_SIZE=64 \
        -o "$OUT_DIR/${f%.cu}.hsaco" \
        "$f" 2>&1)

    if [ $? -eq 0 ]; then
        echo "OK"
        PASS=$((PASS + 1))
    else
        echo "FAILED"
        echo "$OUTPUT" | head -30
        echo ""
        FAIL=$((FAIL + 1))
        FAILED_FILES="$FAILED_FILES $f"
    fi
done

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
if [ $FAIL -gt 0 ]; then
    echo "Failed:$FAILED_FILES"
    exit 1
fi
