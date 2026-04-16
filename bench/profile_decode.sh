#!/usr/bin/env bash
# Run a candle or turbo decode benchmark under rocprofv3 and emit a kernel
# summary + bash-time stats. Artefacts are written to a fresh /tmp/prof_<tag>/.
#
# Used as the first step of the W8 regression-check workflow
# (ROADMAP-DECODE-GAP-2026-04-16.md §Validation harness).
#
# Usage:
#   bench/profile_decode.sh <tag> <backend> [extra args]
#
#   backend = candle | turbo
#   extra args go to the underlying invocation (e.g. --n-gpus, -ts, ...)
#
# Output:
#   /tmp/prof_<tag>/summary.txt   (rocprofv3 summary)
#   /tmp/prof_<tag>/time.txt      (bash time real/user/sys)
#   /tmp/prof_<tag>/stdout.log    (backend stdout/stderr)
#
# Defaults (override with env):
#   MODEL=/artefact/models/gemma-4-26B-A4B-it-Q8_0.gguf
#   PROMPT_FILE=/tmp/prompt_gemma4_512.txt
#   GEN_TOKENS=128
#   GPUS=0,1,2,3
#   CANDLE_BIN=quantized-gemma4
set -u

TAG=${1:?usage: profile_decode.sh <tag> <candle|turbo> [extra args]}
BACKEND=${2:?backend must be candle or turbo}
shift 2

MODEL=${MODEL:-/artefact/models/gemma-4-26B-A4B-it-Q8_0.gguf}
PROMPT_FILE=${PROMPT_FILE:-/tmp/prompt_gemma4_512.txt}
GEN_TOKENS=${GEN_TOKENS:-128}
GPUS=${GPUS:-0,1,2,3}
CANDLE_BIN=${CANDLE_BIN:-quantized-gemma4}

OUT=/tmp/prof_${TAG}
rm -rf "$OUT"
mkdir -p "$OUT/rocprof"

export LD_LIBRARY_PATH=/opt/rocm-7.1.1/core-7.13/lib:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export HIP_VISIBLE_DEVICES=$GPUS

case "$BACKEND" in
  candle)
    PROMPT=$(cat "$PROMPT_FILE")
    CMD=(
      /artefact/candle/target/release/examples/"$CANDLE_BIN"
      --model "$MODEL"
      --n-gpus "$(echo "$GPUS" | tr ',' '\n' | wc -l)"
      --prompt "$PROMPT"
      --sample-len "$GEN_TOKENS"
      --temperature 0
      --seed 42
      "$@"
    )
    ;;
  turbo)
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/artefact/llamacpp-turbo/llama-cpp-gfx906-turbo/build/bin
    TS=$(echo "$GPUS" | tr ',' '/' | sed 's/[0-9]/1/g')
    # Prompt length: just use the prompt-file's approximate token count (N≈512)
    # via `llama-bench -p $len -n $gen -r 1` (deterministic, non-interactive).
    PROMPT_LEN=${PROMPT_LEN:-512}
    CMD=(
      /artefact/llamacpp-turbo/llama-cpp-gfx906-turbo/build/bin/llama-bench
      -m "$MODEL"
      -ngl 999
      -ts "$TS"
      -t 2
      -p "$PROMPT_LEN" -n "$GEN_TOKENS" -r 1
      "$@"
    )
    ;;
  *)
    echo "unknown backend: $BACKEND" >&2; exit 2 ;;
esac

echo "=== $(date -Iseconds) $BACKEND: ${CMD[*]}" | tee "$OUT/stdout.log"

{
  time /opt/rocm-7.1.1/core-7.13/bin/rocprofv3 \
    --runtime-trace \
    --stats --summary --summary-per-domain --summary-units usec \
    --summary-output-file "$OUT/rocprof/summary.txt" \
    -d "$OUT/rocprof" \
    -- "${CMD[@]}"
} >> "$OUT/stdout.log" 2> "$OUT/time.txt"

# rocprofv3 writes the summary to a nested path; flatten it.
NESTED=$(find "$OUT/rocprof" -name "summary.txt.txt" 2>/dev/null | head -1)
if [[ -n "$NESTED" ]]; then
  cp "$NESTED" "$OUT/summary.txt"
  # DB files are huge; keep only what the family-parser needs.
  find "$OUT/rocprof" -name "*_results.db" -delete
fi

echo "=== artefacts in $OUT:" ; ls "$OUT"
