#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

OUT_DIR=$(find target -path "*/candle-hip-kernels-*/out" -type d 2>/dev/null | head -1)

for hsaco in "$OUT_DIR"/*.hsaco; do
    name=$(basename "$hsaco")
    echo "=== $name ==="
    file "$hsaco"
    xxd "$hsaco" | head -3
    echo ""
done
