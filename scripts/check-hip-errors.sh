#!/usr/bin/env bash
# Capture hip build errors for analysis.
cd /artefact/candle
cargo check --features hip -p candle-core 2>&1 | grep "^error\[" | sed 's/:.*//' | sort | uniq -c | sort -rn
echo "---"
cargo check --features hip -p candle-core 2>&1 | grep "^error\[" | head -30
echo "---"
# First 5 unique errors with context
cargo check --features hip -p candle-core 2>&1 | grep -A3 "^error\[" | head -60
