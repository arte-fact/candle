#!/usr/bin/env bash
# ============================================================================
# bench-candle-vs-turbo.sh — Full comparison: Candle vs llamacpp-turbo
#
# Models tested (if available):
#   1. Qwen3.5-9B Q4_1       (dense, 9B params)
#   2. Qwen3.5-27B Q4_0      (dense, 27B params — needs multi-GPU or 2x16GB)
#   3. gemma4-E4B Q4_0       (MoE, 4B active / ~12B total)
#   4. gemma4-26B-A4B Q8_0   (MoE, 4B active / 26B total — needs 2x16GB)
#   5. TinyLlama-1.1B Q4_0   (dense, baseline reference)
#
# Measures:
#   - Prefill speed (t/s)  — long prompt (~500 tokens)
#   - Decode speed (t/s)   — 256 token generation
#   - Wall-clock time
#
# Usage:
#   bash scripts/bench-candle-vs-turbo.sh [--flash-attn] [--profile] [--models MODEL1,MODEL2,...]
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CANDLE_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${CANDLE_MODELS_DIR:-/artefact/models}"
TURBO_BIN="/artefact/llamacpp-turbo/llama-cpp-gfx906-turbo/build/bin"
CANDLE_BIN="$CANDLE_ROOT/target/release/examples"
SAMPLE_LEN=256
RUNS=3  # average over N runs

# Library paths
export LD_LIBRARY_PATH="${TURBO_BIN}:/opt/rocm-7.1.1/core-7.13/lib:/opt/rocm/lib:/opt/rocm-7.1.1/lib:${LD_LIBRARY_PATH:-}"

# Parse args
FLASH_ATTN=0
PROFILE=0
SELECTED_MODELS=""
for arg in "$@"; do
    case "$arg" in
        --flash-attn) FLASH_ATTN=1 ;;
        --profile) PROFILE=1 ;;
        --models=*) SELECTED_MODELS="${arg#--models=}" ;;
    esac
done

# Long prompt for prefill benchmarking (~500 tokens)
PROMPT='Below is a detailed technical analysis of distributed systems architecture patterns used in modern cloud-native applications. We will examine the core principles of microservices, event-driven architectures, and the CAP theorem implications for system design.

Microservices architecture decomposes applications into small, independently deployable services. Each service owns its data and communicates via well-defined APIs. This approach offers several benefits: independent scaling, technology diversity, fault isolation, and organizational alignment with Conways Law. However, it introduces complexity in service discovery, distributed tracing, data consistency, and operational overhead.

Event-driven architecture uses events as the primary mechanism for communication between services. Events represent state changes and are published to event streams or message brokers. Consumers subscribe to relevant events and react accordingly. This pattern enables loose coupling, temporal decoupling, and natural audit trails. Apache Kafka, RabbitMQ, and AWS SNS/SQS are popular implementations.

The CAP theorem states that a distributed system can provide at most two of three guarantees: Consistency, Availability, and Partition tolerance. In practice, since network partitions are inevitable, the real choice is between consistency and availability during partitions. Systems like MongoDB and Cassandra offer tunable consistency levels, allowing developers to make this tradeoff per-operation.

Saga patterns manage distributed transactions across microservices. Unlike traditional two-phase commit protocols, sagas break transactions into a sequence of local transactions, each followed by a compensating transaction in case of failure. Choreography-based sagas use events to coordinate, while orchestration-based sagas use a central coordinator.

Circuit breaker patterns prevent cascading failures in distributed systems. When a service detects that a downstream dependency is failing, it opens the circuit, immediately returning errors instead of waiting for timeouts. After a configurable period, the circuit enters a half-open state, allowing a limited number of test requests to determine if the dependency has recovered.

Service mesh architectures like Istio and Linkerd provide infrastructure-level solutions for service-to-service communication. They handle load balancing, traffic management, security policies, and observability without requiring changes to application code. The sidecar proxy pattern intercepts all network traffic, enabling fine-grained control over communication patterns.

Summarize the key architectural patterns described above and explain how they work together in a modern cloud-native application. Provide specific examples of when you would choose each pattern.'

# ---- Model definitions ----
# Format: name|candle_binary|candle_extra_args|gguf_path|turbo_gguf_path|gpu_layers
declare -a MODELS=()

add_model() {
    local name="$1" candle_bin="$2" candle_args="$3" gguf="$4" turbo_gguf="$5" ngl="$6"
    if [ -f "$gguf" ]; then
        MODELS+=("${name}|${candle_bin}|${candle_args}|${gguf}|${turbo_gguf}|${ngl}")
    else
        echo "SKIP: $name — model file not found: $gguf"
    fi
}

# TinyLlama 1.1B (dense, baseline)
add_model "TinyLlama-1.1B-Q4_0" \
    "quantized" "--which 7b" \
    "$MODELS_DIR/tinyllama-1.1b-q4_0.gguf" \
    "$MODELS_DIR/tinyllama-1.1b-q4_0.gguf" \
    99

# Qwen3.5-9B (dense)
add_model "Qwen3.5-9B-Q4_1" \
    "quantized-qwen35" "" \
    "$MODELS_DIR/Qwen3.5-9B-Q4_1.gguf" \
    "$MODELS_DIR/Qwen3.5-9B-Q4_1.gguf" \
    99

# Qwen3.5-27B (dense, large — may OOM on single GPU)
add_model "Qwen3.5-27B-Q4_0" \
    "quantized-qwen35" "" \
    "$MODELS_DIR/Qwen3.5-27B-Q4_0.gguf" \
    "$MODELS_DIR/Qwen3.5-27B-Q4_0.gguf" \
    99

# gemma4-E4B (MoE, small active)
add_model "gemma4-E4B-Q4_0" \
    "quantized-gemma4" "" \
    "$MODELS_DIR/gemma-4-E4B-it-Q4_0.gguf" \
    "$MODELS_DIR/gemma-4-E4B-it-Q4_0.gguf" \
    99

# gemma4-26B-A4B (MoE, larger)
add_model "gemma4-26B-A4B-Q8_0" \
    "quantized-gemma4" "" \
    "$MODELS_DIR/gemma-4-26B-A4B-it-Q8_0.gguf" \
    "$MODELS_DIR/gemma-4-26B-A4B-it-Q8_0.gguf" \
    99

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     Candle vs llamacpp-turbo — gfx906 (MI50) Benchmark         ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Prompt: ~556 tokens   Generate: $SAMPLE_LEN tokens               ║"
echo "║  Runs per config: $RUNS (warmup + average)                        ║"
echo "║  Flash attention: $([ $FLASH_ATTN -eq 1 ] && echo 'ENABLED (v2)' || echo 'DISABLED  ')                           ║"
echo "║  Profiling: $([ $PROFILE -eq 1 ] && echo 'ENABLED ' || echo 'DISABLED')                                      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

if [ $FLASH_ATTN -eq 1 ]; then
    export CANDLE_FLASH_ATTN_V2_ENABLE=1
else
    unset CANDLE_FLASH_ATTN_V2_ENABLE 2>/dev/null || true
    unset CANDLE_FLASH_ATTN_V1_ENABLE 2>/dev/null || true
fi

# ---- Result storage ----
RESULTS_FILE="/tmp/bench-candle-vs-turbo-$(date +%Y%m%d-%H%M%S).md"
echo "# Candle vs Turbo Benchmark — $(date +%Y-%m-%d)" > "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "| Model | Framework | Prefill (t/s) | Decode (t/s) | Wall (s) |" >> "$RESULTS_FILE"
echo "|-------|-----------|--------------|-------------|----------|" >> "$RESULTS_FILE"

# ---- Helper: extract candle metrics ----
extract_candle_metrics() {
    local output="$1"
    local prefill decode wall
    prefill=$(echo "$output" | grep -oP '[\d.]+(?= token/s)' | head -1)
    decode=$(echo "$output" | grep -oP '[\d.]+(?= token/s)' | tail -1)
    wall=$(echo "$output" | grep -oP 'real\s+\Km[\d.s]+' | head -1)
    echo "${prefill:-0}|${decode:-0}|${wall:-n/a}"
}

# ---- Helper: extract turbo metrics ----
extract_turbo_metrics() {
    local output="$1"
    local prefill decode
    prefill=$(echo "$output" | grep 'prompt eval' | grep -oP '[\d.]+(?=\s+tokens per second)')
    decode=$(echo "$output" | grep 'eval.*tokens per second' | grep -v 'prompt' | grep -oP '[\d.]+(?=\s+tokens per second)')
    echo "${prefill:-0}|${decode:-0}"
}

# ---- Run benchmarks ----
for model_spec in "${MODELS[@]}"; do
    IFS='|' read -r name candle_bin candle_args gguf turbo_gguf ngl <<< "$model_spec"

    # Skip if model filter is set and doesn't match
    if [ -n "$SELECTED_MODELS" ] && ! echo "$SELECTED_MODELS" | grep -qi "$name"; then
        continue
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "MODEL: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ---- Candle ----
    echo ""
    echo "▸ Candle$([ $FLASH_ATTN -eq 1 ] && echo ' (flash-attn v2)' || echo ' (rocBLAS)')"

    best_prefill=0
    best_decode=0
    for run in $(seq 1 $RUNS); do
        output=$(
            { time "$CANDLE_BIN/$candle_bin" \
                --model "$gguf" \
                $candle_args \
                --prompt "$PROMPT" \
                --sample-len $SAMPLE_LEN \
                --temperature 0 \
                --seed 42 ; } 2>&1
        ) || true

        prefill=$(echo "$output" | grep -oP '[\d.]+(?= token/s)' | head -1)
        decode=$(echo "$output" | grep -oP '[\d.]+(?= token/s)' | tail -1)
        wall=$(echo "$output" | grep -oP '(?<=real\t)[\dm.s]+' | head -1)
        echo "  Run $run: prefill=${prefill:-err} t/s  decode=${decode:-err} t/s  wall=${wall:-?}"

        # Keep best (warmup run 1 is slower)
        if [ -n "$prefill" ] && [ "$(echo "$prefill > $best_prefill" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
            best_prefill="$prefill"
            best_decode="$decode"
            best_wall="$wall"
        fi
    done
    echo "  ★ Best: prefill=$best_prefill t/s  decode=$best_decode t/s"
    echo "| $name | Candle | $best_prefill | $best_decode | ${best_wall:-n/a} |" >> "$RESULTS_FILE"

    # ---- llamacpp-turbo ----
    echo ""
    echo "▸ llamacpp-turbo"

    turbo_best_prefill=0
    turbo_best_decode=0
    for run in $(seq 1 $RUNS); do
        output=$(
            timeout 120 "$TURBO_BIN/llama-cli" \
                -m "$turbo_gguf" \
                -p "$PROMPT" \
                -n $SAMPLE_LEN \
                --temp 0 \
                -s 42 \
                -ngl "$ngl" \
                --no-display-prompt \
                2>&1
        ) || true

        prefill=$(echo "$output" | grep 'prompt eval' | grep -oP '[\d.]+(?=\s+tokens per second)')
        decode=$(echo "$output" | grep 'eval.*tokens per second' | grep -v 'prompt' | grep -oP '[\d.]+(?=\s+tokens per second)')
        echo "  Run $run: prefill=${prefill:-err} t/s  decode=${decode:-err} t/s"

        if [ -n "$prefill" ] && [ "$(echo "$prefill > $turbo_best_prefill" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
            turbo_best_prefill="$prefill"
            turbo_best_decode="$decode"
        fi
    done
    echo "  ★ Best: prefill=$turbo_best_prefill t/s  decode=$turbo_best_decode t/s"
    echo "| $name | Turbo | $turbo_best_prefill | $turbo_best_decode | n/a |" >> "$RESULTS_FILE"

    # ---- Ratio ----
    if [ "$turbo_best_prefill" != "0" ] && [ "$best_prefill" != "0" ]; then
        ratio_prefill=$(echo "scale=0; $best_prefill * 100 / $turbo_best_prefill" | bc -l 2>/dev/null || echo "?")
        ratio_decode=$(echo "scale=0; $best_decode * 100 / $turbo_best_decode" | bc -l 2>/dev/null || echo "?")
        echo "  → Candle/Turbo: prefill=${ratio_prefill}%  decode=${ratio_decode}%"
        echo "| $name | **Ratio** | **${ratio_prefill}%** | **${ratio_decode}%** | — |" >> "$RESULTS_FILE"
    fi

    echo ""

    # ---- Optional profiling ----
    if [ $PROFILE -eq 1 ]; then
        echo "▸ rocprofv3 kernel trace (candle)"
        PROF_OUT="/tmp/bench-profile-${name}-$(date +%H%M%S)"
        rocprofv3 --kernel-trace -o "$PROF_OUT" -- \
            "$CANDLE_BIN/$candle_bin" \
                --model "$gguf" \
                $candle_args \
                --prompt "$PROMPT" \
                --sample-len 32 \
                --temperature 0 \
                --seed 42 2>&1 | tail -5
        echo "  Profile saved: ${PROF_OUT}*"
        echo ""
    fi
done

echo "" >> "$RESULTS_FILE"
echo "Flash attention: $([ $FLASH_ATTN -eq 1 ] && echo 'v2 enabled' || echo 'disabled')" >> "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Results saved to: $RESULTS_FILE"
echo ""
cat "$RESULTS_FILE"
