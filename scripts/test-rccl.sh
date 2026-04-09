#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Compiling RCCL test ==="
hipcc -o /tmp/test_rccl test_rccl.cpp -lrccl -lpthread 2>&1 | grep -v "warning:" || true

echo "=== Running ==="
HSA_FORCE_FINE_GRAIN_PCIE=1 /tmp/test_rccl
