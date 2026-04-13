#!/usr/bin/env bash
# ============================================================================
# bench-models-cuda.sh — Candle CUDA benchmark sweep for the pinned GGUF set.
#
# Runs each model in `models/` through its matching candle example
# (quantized-gemma4 / quantized-qwen35 / quantized-qwen3-moe / quantized),
# measures prefill and decode tokens/s on the first CUDA device, and prints a
# summary table.
#
# VRAM notes (RTX 3090 = 24 GiB):
#   - Q4_*, Q5_*, Q6K, Q8_0, BF16 quants stay on GPU in their native layout
#     (MMVQ path); files under ~19 GiB fit.
#   - MXFP4 has no MMVQ kernel on either backend — `QMatMul::from_arc` forces
#     dequantize → F32 at load. A 16.6 GiB MXFP4 MoE expands to ~130 GiB F32
#     and OOMs instantly. Those models are skipped by default; override with
#     `--include-mxfp4` if you've got more VRAM (e.g. multi-GPU or A100 80GB).
#
# Usage:
#   scripts/bench-models-cuda.sh                       # bench everything runnable
#   scripts/bench-models-cuda.sh --quick               # 1 run × 64 decode tokens
#   scripts/bench-models-cuda.sh --models=gemma-4-E4B  # substring filter
#   scripts/bench-models-cuda.sh --include-mxfp4       # attempt the MoE MXFP4 files too
#   scripts/bench-models-cuda.sh --build-only          # build release examples, then exit
#
# Env:
#   MODELS_DIR       (default: <repo>/models)
#   CANDLE_FEATURES  extra feature flags, e.g. "cuda,cudnn"
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
BIN_DIR="$REPO_ROOT/target/release/examples"
FEATURES="${CANDLE_FEATURES:-cuda}"

SAMPLE_LEN=128
RUNS=2          # timed runs per model (plus one warmup that's discarded)
PROMPT='Write a concise technical essay on the principal design tradeoffs of transformer language models, touching on attention cost, KV-cache memory, MoE routing, quantization formats (Q4, Q8, MXFP4), and why decode throughput is dominated by weight bandwidth rather than compute. Keep it dense and precise.'

QUICK=0
INCLUDE_MXFP4=0
BUILD_ONLY=0
MODEL_FILTER=""
CONTEXT_LEN=""          # optional --context-len plumbed to runners that support it
LLAMACPP_BIN="${LLAMACPP_CUDA_BIN:-}"   # optional: path to llama-cli for baseline rows

for arg in "$@"; do
    case "$arg" in
        --quick)          QUICK=1 ;;
        --include-mxfp4)  INCLUDE_MXFP4=1 ;;
        --build-only)     BUILD_ONLY=1 ;;
        --models=*)       MODEL_FILTER="${arg#--models=}" ;;
        --context-len=*)  CONTEXT_LEN="${arg#--context-len=}" ;;
        --runs=*)         RUNS="${arg#--runs=}" ;;
        --sample-len=*)   SAMPLE_LEN="${arg#--sample-len=}" ;;
        --llamacpp=*)     LLAMACPP_BIN="${arg#--llamacpp=}" ;;
        -h|--help)        sed -n '3,25p' "$0"; exit 0 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

if [[ "$QUICK" -eq 1 ]]; then
    SAMPLE_LEN=64
    RUNS=1
fi

# ---- Catalog: filename | runner | extra-flags ---------------------------------
# extra-flags is a space-separated list of additional CLI args for the runner.
# Keep MXFP4 MoE files listed; the `runnable` check filters by VRAM feasibility.
declare -a CATALOG=(
    "gemma-4-E4B-it-Q8_0.gguf|quantized-gemma4|"
    "gemma-4-31B-it-Q4_K_M.gguf|quantized-gemma4|"
    "gemma-4-26B-A4B-it-MXFP4_MOE.gguf|quantized-gemma4|"
    "Qwen3.5-9B-BF16.gguf|quantized-qwen35|"
    "Qwen3.5-27B-Q4_1.gguf|quantized-qwen35|"
    "Qwen3.5-35B-A3B-MXFP4_MOE.gguf|quantized-qwen35|--split_prompt"
    "Qwen3-Coder-30B-A3B-Instruct-1M-Q4_0.gguf|quantized-qwen3-moe|"
    "Devstral-Small-2-24B-Instruct-2512-UD-Q4_K_XL.gguf|quantized|--which 7b-mistral-instruct-v0.2"
)

is_mxfp4() { [[ "$1" == *MXFP4* ]]; }

# ---- Build all needed binaries in one cargo invocation ------------------------
echo "── build (release, --features $FEATURES) ──────────────────────────────"
cd "$REPO_ROOT"
export PATH="$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH"

BUILD_ARGS=(
    --release --no-default-features --features "$FEATURES"
    --example quantized-gemma4
    --example quantized-qwen35
    --example quantized-qwen3-moe
    --example quantized
    -p candle-examples
)
if ! cargo build "${BUILD_ARGS[@]}" 2>&1 | tail -20; then
    echo "[fatal] cargo build failed" >&2
    exit 1
fi

for bin in quantized-gemma4 quantized-qwen35 quantized-qwen3-moe quantized; do
    if [[ ! -x "$BIN_DIR/$bin" ]]; then
        echo "[fatal] missing built binary: $BIN_DIR/$bin" >&2; exit 1
    fi
done

if [[ "$BUILD_ONLY" -eq 1 ]]; then
    echo "build-only: done"; exit 0
fi

# ---- GPU info -----------------------------------------------------------------
echo ""
echo "── gpu ────────────────────────────────────────────────────────────────"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true

# ---- Helpers ------------------------------------------------------------------
extract_metric() {  # args: <output> <pattern>
    local out="$1" pat="$2"
    printf '%s\n' "$out" | grep -F "$pat" | grep -oE '[0-9]+\.[0-9]+' | tail -1
}

run_one() {  # args: <file> <runner> <extra...>; echoes "prefill|decode|peak_vram_MiB|ok"
    local file="$1" runner="$2"; shift 2
    local extra=("$@")
    local path="$MODELS_DIR/$file"
    local bin="$BIN_DIR/$runner"
    local peak_file; peak_file=$(mktemp)

    # Background VRAM poller — tracks the max `memory.used` seen while the
    # runner is alive. Polls every 200 ms; stops as soon as the fifo `done`
    # file appears.
    local done_file; done_file=$(mktemp)
    rm -f "$done_file"
    (
        local peak=0
        while [[ ! -e "$done_file" ]]; do
            local cur
            cur=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
            if [[ -n "$cur" ]] && (( cur > peak )); then peak="$cur"; echo "$peak" > "$peak_file"; fi
            sleep 0.2
        done
    ) &
    local poller_pid=$!

    local out rc=0
    out=$("$bin" \
        --model "$path" \
        --prompt "$PROMPT" \
        --sample-len "$SAMPLE_LEN" \
        --temperature 0 \
        --seed 42 \
        "${extra[@]}" 2>&1) || rc=$?

    touch "$done_file"
    wait "$poller_pid" 2>/dev/null || true

    local peak; peak=$(cat "$peak_file" 2>/dev/null || echo 0)
    rm -f "$peak_file" "$done_file"

    local prefill decode load_s
    prefill=$(extract_metric "$out" "prompt") ; prefill="${prefill:-}"
    decode=$(extract_metric "$out" "generated") ; decode="${decode:-}"
    if [[ -z "$prefill" ]]; then
        prefill=$(printf '%s\n' "$out" | grep -oE 'prompt processed: [0-9.]+ token/s' | grep -oE '[0-9]+\.[0-9]+' | tail -1)
    fi
    # load time: "loaded in X.XXs" (quantized-gemma4/qwen35) or
    # "loaded N tensors (XGB) in X.XXs" (quantized/quantized-qwen3-moe)
    load_s=$(printf '%s\n' "$out" | grep -oE 'loaded[^0-9]*in [0-9]+\.[0-9]+s' | grep -oE '[0-9]+\.[0-9]+' | tail -1)
    load_s="${load_s:-0}"

    if [[ "$rc" -ne 0 || -z "${prefill:-}" || -z "${decode:-}" ]]; then
        {
            echo "    ---- run output (tail) ----"
            printf '%s\n' "$out" | tail -8 | sed 's/^/    /'
            echo "    ---------------------------"
        } >&2
        echo "||${peak}|${load_s}|fail(rc=$rc)"
        return 1
    fi
    echo "${prefill}|${decode}|${peak}|${load_s}|ok"
    return 0
}

wait_vram_settle() {  # wait until VRAM drops back below $1 MiB (default 2048)
    local threshold="${1:-2048}" i=0
    while (( i < 30 )); do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
        if [[ -n "$used" ]] && (( used < threshold )); then return 0; fi
        sleep 1; i=$((i+1))
    done
    return 1
}

# ---- Results accumulators -----------------------------------------------------
STAMP="$(date +%Y%m%d-%H%M%S)"
RESULTS_FILE="/tmp/bench-models-cuda-${STAMP}.md"
CSV_FILE="/tmp/bench-models-cuda-${STAMP}.csv"
{
    echo "# Candle CUDA bench — RTX 3090 — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo ""
    echo "prompt: ~$(echo "$PROMPT" | wc -w) words · sample-len=$SAMPLE_LEN · runs=$RUNS"
    echo ""
    echo "| model | runner | prefill t/s (mean±σ) | decode t/s (mean±σ) | peak MiB | load s | status |"
    echo "|---|---|---:|---:|---:|---:|---|"
} > "$RESULTS_FILE"
# CSV: one row per (model, run). Parseable by downstream tooling.
echo "model,runner,run,prefill_tps,decode_tps,peak_mib,load_s,status" > "$CSV_FILE"

# --- stats helper ---
stats() {  # args: value1 value2 ... ; echoes "mean|stddev|min|max|n"
    local -a vals=("$@")
    local n=${#vals[@]}
    if (( n == 0 )); then echo "0|0|0|0|0"; return; fi
    printf '%s\n' "${vals[@]}" | awk -v n="$n" '
        { v=$1; sum+=v; sumsq+=v*v; if (NR==1||v<mn) mn=v; if (NR==1||v>mx) mx=v }
        END {
            mean = sum/n;
            var  = (n>1) ? (sumsq/n - mean*mean) : 0;
            if (var<0) var=0;
            printf "%.2f|%.2f|%.2f|%.2f|%d", mean, sqrt(var), mn, mx, n
        }
    '
}

# ---- Main loop ----------------------------------------------------------------
echo ""
echo "── bench ──────────────────────────────────────────────────────────────"
for entry in "${CATALOG[@]}"; do
    IFS='|' read -r file runner extra <<< "$entry"
    if [[ -n "$MODEL_FILTER" ]] && ! [[ "$file" == *$MODEL_FILTER* ]]; then
        continue
    fi
    if [[ ! -f "$MODELS_DIR/$file" ]]; then
        echo "[skip] $file — not present"
        continue
    fi
    if is_mxfp4 "$file" && [[ "$INCLUDE_MXFP4" -eq 0 ]]; then
        echo "[skip] $file — MXFP4 dequantizes to F32 on GPU; 24 GiB insufficient"
        echo "| $file | $runner | — | — | — | skipped (MXFP4 > 24 GiB when dequant) |" >> "$RESULTS_FILE"
        continue
    fi

    echo ""
    echo "▸ $file  ($runner $extra)"
    read -ra extra_arr <<< "$extra"
    wait_vram_settle 2048 || echo "  [warn] VRAM not settled before run" >&2

    # warmup (discarded — first call has JIT/kernel-load overhead)
    echo "  warmup…"
    warmup_result=$(run_one "$file" "$runner" "${extra_arr[@]}" || true)
    IFS='|' read -r _ _ warmup_peak warmup_load warmup_status <<< "$warmup_result"
    echo "$file,$runner,0,,,${warmup_peak:-},${warmup_load:-},warmup_${warmup_status:-fail}" >> "$CSV_FILE"
    if [[ "$warmup_status" != "ok" ]]; then
        echo "  warmup FAILED — skipping timed runs"
        echo "| $file | $runner | — | — | ${warmup_peak:-?} | ${warmup_load:-—} | ${warmup_status:-fail} |" >> "$RESULTS_FILE"
        wait_vram_settle 2048 || true
        continue
    fi

    declare -a prefill_vals=() decode_vals=() load_vals=()
    local_peak_max="${warmup_peak:-0}"
    ok_count=0
    for r in $(seq 1 "$RUNS"); do
        result=$(run_one "$file" "$runner" "${extra_arr[@]}") || true
        IFS='|' read -r p d m l s <<< "$result"
        echo "$file,$runner,$r,${p:-},${d:-},${m:-},${l:-},${s:-fail}" >> "$CSV_FILE"
        if [[ "$s" == "ok" ]]; then
            printf '  run %d/%d  prefill=%s t/s  decode=%s t/s  peak=%s MiB  load=%ss\n' "$r" "$RUNS" "$p" "$d" "$m" "$l"
            prefill_vals+=("$p"); decode_vals+=("$d"); load_vals+=("$l")
            ok_count=$((ok_count+1))
            if awk -v a="$m" -v b="$local_peak_max" 'BEGIN{exit !(a>b)}'; then local_peak_max="$m"; fi
        else
            printf '  run %d/%d  FAIL (peak=%s MiB)\n' "$r" "$RUNS" "${m:-?}"
        fi
    done

    if [[ "$ok_count" -gt 0 ]]; then
        IFS='|' read -r pm ps _ _ _ <<< "$(stats "${prefill_vals[@]}")"
        IFS='|' read -r dm ds _ _ _ <<< "$(stats "${decode_vals[@]}")"
        IFS='|' read -r lm _  _ _ _ <<< "$(stats "${load_vals[@]}")"
        echo "  avg     prefill=${pm}±${ps} t/s   decode=${dm}±${ds} t/s   peak=${local_peak_max} MiB   load=${lm}s"
        echo "| $file | $runner | ${pm}±${ps} | ${dm}±${ds} | $local_peak_max | ${lm} | ok ($ok_count/$RUNS) |" >> "$RESULTS_FILE"
    else
        echo "| $file | $runner | — | — | ${local_peak_max:-?} | — | all runs failed |" >> "$RESULTS_FILE"
    fi
done

# ---- optional llama.cpp baseline ---------------------------------------------
if [[ -n "$LLAMACPP_BIN" && -x "$LLAMACPP_BIN" ]]; then
    echo ""
    echo "── llama.cpp baseline (via $LLAMACPP_BIN) ─────────────────────────────"
    echo "" >> "$RESULTS_FILE"
    echo "### llama.cpp-CUDA baseline (same prompt, same GPU)" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    echo "| model | prefill t/s | decode t/s |" >> "$RESULTS_FILE"
    echo "|---|---:|---:|" >> "$RESULTS_FILE"
    for entry in "${CATALOG[@]}"; do
        IFS='|' read -r file _ _ <<< "$entry"
        [[ -n "$MODEL_FILTER" ]] && ! [[ "$file" == *$MODEL_FILTER* ]] && continue
        [[ ! -f "$MODELS_DIR/$file" ]] && continue
        is_mxfp4 "$file" && [[ "$INCLUDE_MXFP4" -eq 0 ]] && continue
        echo "▸ llama.cpp: $file"
        out=$("$LLAMACPP_BIN" -m "$MODELS_DIR/$file" -p "$PROMPT" -n "$SAMPLE_LEN" -ngl 99 --temp 0 --seed 42 2>&1) || true
        pp=$(printf '%s\n' "$out" | grep -oE 'prompt eval.*[0-9.]+ tokens per second' | grep -oE '[0-9]+\.[0-9]+' | tail -1)
        tg=$(printf '%s\n' "$out" | grep -vE 'prompt eval' | grep -oE 'eval.*[0-9.]+ tokens per second' | grep -oE '[0-9]+\.[0-9]+' | tail -1)
        echo "| $file | ${pp:-—} | ${tg:-—} |" >> "$RESULTS_FILE"
        wait_vram_settle 2048 || true
    done
fi

echo ""
echo "── summary ────────────────────────────────────────────────────────────"
cat "$RESULTS_FILE"
echo ""
echo "results (markdown): $RESULTS_FILE"
echo "results (csv):      $CSV_FILE"
