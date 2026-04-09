#!/usr/bin/env bash
# Test hiprand directly to isolate the segfault.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Compiling hiprand test ==="
hipcc -o /tmp/test_hiprand test_hiprand.cpp -lhiprand 2>&1 | grep -v "warning:" || true
echo ""

echo "=== Running hiprand test ==="
/tmp/test_hiprand
