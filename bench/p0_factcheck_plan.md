# P0 fact-check commands (2026-04-19)

Every bench uses `LD_LIBRARY_PATH=/opt/rocm-7.1.1/core-7.13/lib:/opt/rocm/lib`.

## P0.1 — Baseline perf snapshot

Models: `gemma-4-E4B-it-Q4_0.gguf`, `Qwen3.5-9B-Q4_1.gguf`, `Qwen3-Coder-Next-Q4_0.gguf`.

For each × `{CANDLE_KV_Q8=0, CANDLE_KV_Q8=1}`:
```
candle-bench bench <model> --prompt-len 512 --gen-len 64 --json
```

Record: pp512 t/s, tg64 t/s. Save to `bench/p0_baseline_<date>.json`.

## P0.2 — rocprofv3 trace confirming Q8-KV dequant overhead

```
LD_LIBRARY_PATH=... CANDLE_KV_Q8=1 rocprofv3 --kernel-trace -o p02_trace -- \
  candle-bench bench /artefact/models/gemma-4-E4B-it-Q4_0.gguf --prompt-len 32 --gen-len 16
```

Then: jq-filter kernel trace to confirm:
- `dequantize_q8_0_kv_f32` appears every decode layer
- `gqa_decode_gemv_v_q8` does NOT appear
- measure total µs of dequant as % of decode time

## P0.3 — Q8 GEMV correctness

```
cargo run --release -p candle-core --example hip_q8_attn_pipeline_test --features hip
cargo run --release -p candle-core --example hip_q8_qk_attn_test --features hip
cargo run --release -p candle-core --example hip_q8_v_attn_test --features hip
cargo run --release -p candle-core --example hip_q8_decode_loop_test --features hip
```

All must exit 0.

## P0.5 — Occupancy PMC

```
LD_LIBRARY_PATH=... rocprofv3 --pmc SQ_WAVES SQ_BUSY_CYCLES GRBM_GUI_ACTIVE VALU_INSTS \
  -- candle-bench bench /artefact/models/Qwen3-Coder-30B-A3B-Instruct-1M-Q4_0.gguf \
  --prompt-len 8 --gen-len 4
```

Derive waves/EU for mul_mat_vec_q4_K_q8_1 kernel.
