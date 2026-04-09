#!/usr/bin/env bash
# Test that each HSACO can be loaded and function symbols found.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

# Find the build output dir
OUT_DIR=$(find target -path "*/candle-hip-kernels-*/out" -type d 2>/dev/null | head -1)
if [ -z "$OUT_DIR" ]; then
    echo "Build first: cargo build --features hip -p candle-core"
    exit 1
fi

echo "HSACO dir: $OUT_DIR"
echo ""

for hsaco in "$OUT_DIR"/*.hsaco; do
    name=$(basename "$hsaco" .hsaco)
    size=$(stat -c%s "$hsaco")
    echo -n "  $name.hsaco (${size}B) ... "
    # Check it's a valid ELF
    if file "$hsaco" | grep -q "ELF"; then
        # List exported symbols
        syms=$(readelf -s "$hsaco" 2>/dev/null | grep "FUNC.*GLOBAL" | wc -l)
        echo "OK ($syms exported functions)"
    else
        echo "INVALID (not ELF)"
    fi
done
