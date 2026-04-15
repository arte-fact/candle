---
name: candle-bench
description: Rust CLI tool for every recurring candle-investigation task — GGUF metadata extraction, model-folder listing, GGUF download via curl, single-shot bench of candle / llama.cpp / turbo, A/B env-var comparison, catalog matrix over multiple models, and HSACO kernel symbol + VGPR inspection.  Use whenever the user asks about model stats, wants a bench, wants to compare two env configs, wants a coverage matrix, or wants to know which kernels are in a compiled `.hsaco`.  Each subcommand has `--json` for machine-readable output.
argument-hint: "[subcommand] [args]"
allowed-tools: Bash Read Grep Glob Write Edit
---

# candle-bench — investigation tool

Binary: `./target/release/examples/candle-bench`
Source: `candle-examples/examples/candle-bench/`

Build once per session (or after kernel-source changes):

```
cargo build --release --features hip -p candle-examples --example candle-bench
```

All invocations require the ROCm LD path because they dynamically load
candle binaries and rocBLAS; put this at the start of any Bash call:

```
LD_LIBRARY_PATH=/opt/rocm-7.1.1/core-7.13/lib:/opt/rocm/lib
```

(`Bash(candle-bench *)` is pre-authorised in `.claude/settings.local.json`.)

## When to use which subcommand

| Goal | Subcommand |
|---|---|
| "What arch / experts / size / context is this GGUF?" | `meta` |
| "What GGUFs do we have?" / "list the MoE ones" | `list` |
| "Download `Qwen3-Coder-Next-Q4_0.gguf` from unsloth" | `fetch` |
| "Run pp128/pp512/tg64 on candle or llama.cpp" | `bench` |
| "Measure the speedup of env X=1 vs X=0" | `compare` |
| "Re-run coverage across the built-in model catalog" | `matrix` |
| "List kernels + VGPR counts in a `.hsaco`" | `kernels` |

## Subcommand reference

### `meta <MODEL>`

Dumps the GGUF header fields that we care about for kernel planning.
Use `--json` and pipe through `jq` when you want a specific field.

```
candle-bench meta /artefact/models/gemma-4-26B-A4B-it-Q8_0.gguf --json \
  | jq '.result | {arch, n_experts: .expert_count, topk: .expert_used_count, size_label}'
```

Output fields: `arch`, `name`, `size_label`, `quantized_by`, `license`,
`block_count`, `embedding_length`, `feed_forward_length`, `head_count`,
`head_count_kv`, `context_length`, `expert_count`, `expert_used_count`,
`expert_feed_forward_length`, `is_hybrid`, `dtype_histogram`.

### `list [--dir DIR]`

Enumerates every `.gguf` under `DIR` (default `$MODELS_DIR` or
`/artefact/models`) and emits the same per-file metadata.  Useful first
step when you don't remember what variants are on disk.

### `fetch <URL> [--expected-bytes N] [--out-name NAME]`

Wraps `curl -L --fail --retry 5 -C - --progress-bar` (fallback `wget -c`).
Verifies `Content-Length` if `--expected-bytes` is passed.  Resume-capable.

```
candle-bench fetch 'https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/resolve/main/Qwen3-Coder-Next-Q4_0.gguf' \
  --expected-bytes 45330098208
```

### `bench --model M --backend {candle,llamacpp,turbo} [flags]`

One bench, pp + tg.  Flags:

- `--bin NAME` — candle example (`quantized-gemma4`, `quantized-qwen35`, …).
  Defaults to `quantized-gemma4`; set when the model is not a gemma4.
- `--gpus 0,1,2,3` — passed via `HIP_VISIBLE_DEVICES`; candle also gets
  `--n-gpus N` (pipeline-parallel split).
- `--prompt-len 128 --prompt-len 512` — repeatable.  One pp run per length.
- `--tg-len 64` — decode bench; 0 disables.
- `--env KEY=VAL` — repeatable.  e.g. `--env CANDLE_MMQ_TURBO_PORT=1`.
- `--extra-arg --split-prompt` — pass-through to candle (needed for
  qwen3next and other hybrid-GDN archs).
- `--reps N` — repeats.  We keep the best run per key.
- `--timeout N` — per-run timeout in seconds (not strictly enforced yet;
  child inherits the default and we report rc=-1 on signal).
- `--json` / `--log-dir`.

Example (gemma-4-26B-A4B Q8_0 on 4 GPUs with M7 turbo MoE MMQ):

```
candle-bench bench --model /artefact/models/gemma-4-26B-A4B-it-Q8_0.gguf \
  --backend candle --bin quantized-gemma4 --gpus 0,1,2,3 \
  --prompt-len 128 --prompt-len 512 --tg-len 64 \
  --env CANDLE_MMQ_TURBO_PORT=1 --reps 1 --json
```

### `compare --a 'K=V [...]' --b 'K=V [...]' [bench flags]`

Runs `bench` twice with extra env pairs applied to each side.  Prints
the per-key ratio `b / a`.

```
candle-bench compare --model /artefact/models/gemma-4-26B-A4B-it-Q8_0.gguf \
  --bin quantized-gemma4 --gpus 0,1,2,3 --prompt-len 512 --tg-len 0 \
  --a 'CANDLE_MMQ_TURBO_PORT=0' --b 'CANDLE_MMQ_TURBO_PORT=1' \
  --a-label baseline --b-label m7 --json
```

### `matrix [--tag T] [--only L] [--catalog FILE]`

Iterates the built-in catalog (in `catalog.rs` — mirrors
`scripts/download-models.sh` plus per-model candle-binary + GPU + known
blockers).  Filters by `--tag` (e.g. `moe`, `q4_0`, `gemma4`) or `--only`
(exact label).  Skips entries whose model file is missing.

The catalog marks **known-broken cells** with a `blockers` dict so a
regression vs an expected-fail is visible.  Current blockers (kept in
sync with SESSION memory):

- `qwen35-9B-BF16` pp128/pp512: no BF16 MMQ path in candle → rocBLAS.
- `qwen3-coder-30B-A3B-Q4_0` pp128/pp512: rocBLAS Tensile gfx906 gap
  (missing `Cijk_Alik...ISA906...WG16_16_1` kernel), candle crashes
  with `hipErrorInvalidImage` at prompt > ~100 tokens on multi-GPU.
- `qwen35-35B-A3B-MXFP4` all keys: no MXFP4 kernel family in candle yet.

### `kernels <HSACO> [--filter PATTERN]`

Wraps `/opt/rocm/llvm/bin/llvm-readobj --symbols --notes`.  Prints
kernel name, VGPR / SGPR / AGPR count, private scratch size.

```
candle-bench kernels \
  target/release/build/candle-hip-kernels-*/out/mmq_turbo.hsaco \
  --filter turbo_moe
```

Use this after a kernel change to sanity-check register pressure and
scratch spills before running a full bench.

## JSON output schema

Every subcommand's `--json` emits:

```json
{ "tool": "candle-bench", "version": 1, "cmd": "<name>",
  "ts": "<rfc3339>", "result": <per-cmd>, "errors": {} }
```

`result` structure is stable per subcommand.  Parsing with `jq` is the
intended downstream path.

## Typical error patterns

- `hipErrorInvalidImage`: rocBLAS Tensile kernel variant missing for
  gfx906.  Not a candle bug.  Look for `Cijk_...ISA906...` in the stderr.
  Workaround: force flash-attn-v2 path via `CANDLE_FLASH_V2_MAX_LK=8192`
  or fall back to MMVQ by clearing `CANDLE_MMQ_TURBO_PORT`.

- `libroctx64.so.4: cannot open shared object file`: the caller didn't
  set `LD_LIBRARY_PATH`.  Always prefix the Bash call with
  `LD_LIBRARY_PATH=/opt/rocm-7.1.1/core-7.13/lib:/opt/rocm/lib`.

- `rc=-1` with the run ending mid-way through `reading tokenizer from GGUF
  metadata...`: a prior process left stranded VRAM (ROCm 7.1.1 leaks on
  SIGKILL / coredump).  Check with
  `/opt/rocm-7.1.1/core-7.13/bin/rocm-smi --showmeminfo vram`.  If the
  totals sum near full without a running `KFD process`, the driver is
  wedged and needs a reset (root).

- `timeout: the monitored command dumped core`: same stranded-VRAM
  symptom; `candle-bench bench` passes `--timeout` through but doesn't
  kill the child from Rust yet, so a shell `timeout` wrapper will report
  coredump when the child hits the allocator.

## When NOT to use this tool

- Fine-grained ROCProf / roctx profiling — not in scope; see
  `bench/prefill_3way.py` which integrates `rocprofv3`.
- vLLM backend — not implemented; `bench/vllm_prefill.py` does it.
- Per-kernel micro-benches — write a dedicated Rust test example like
  `candle-core/examples/hip_moe_mmq_test.rs`; candle-bench is for
  end-to-end runs.
