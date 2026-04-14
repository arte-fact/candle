"""vLLM prefill/decode bench — invoked by prefill_3way.py.

Usage: vllm_prefill.py <model.gguf> <output.json>
Env:   VLLM_BENCH_REPS  (default 3) — reps per bench slot
       VLLM_BENCH_KEYS  (default "pp128,pp512,pp2048,tg64") — benches to run
Writes {"pp128": tps, "pp512": tps, "pp2048": tps, "tg64": tps,
        "error": str|None}
"""
import json
import os
import sys
import time
import traceback

os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPS = int(os.environ.get("VLLM_BENCH_REPS", "3"))
KEYS = set(os.environ.get("VLLM_BENCH_KEYS",
                          "pp128,pp512,pp2048,tg64").split(","))


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: vllm_prefill.py <model.gguf> <output.json>", file=sys.stderr)
        return 2
    model = sys.argv[1]
    out_path = sys.argv[2]
    result = {"pp128": None, "pp512": None, "pp2048": None,
              "tg64": None, "error": None}
    try:
        from vllm import LLM, SamplingParams  # type: ignore

        llm = LLM(
            model=model,
            quantization="gguf",
            dtype="half",
            gpu_memory_utilization=0.75,
            max_model_len=2048,
            enforce_eager=True,
            max_num_batched_tokens=2048,
            disable_log_stats=True,
            # Turn prefix caching OFF so each rep does a real prefill.
            # With it on (v1 default), the warmup call populates KV cache
            # and subsequent reps report ~20k t/s phantom "prefill" that
            # is really a cache pass-through. See README / 2026-04-14 note.
            enable_prefix_caching=False,
        )

        def bench_prefill(n_prompt: int) -> float:
            prompt = " ".join(["word"] * n_prompt)
            sp = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
            llm.generate([prompt], sp)  # warmup
            best = float("inf")
            for _ in range(REPS):
                t = time.perf_counter()
                out = llm.generate([prompt], sp)
                dt = time.perf_counter() - t
                best = min(best, dt)
            actual = len(out[0].prompt_token_ids)
            return actual / best

        def bench_decode(n_gen: int) -> float:
            sp = SamplingParams(max_tokens=n_gen, temperature=0.0, ignore_eos=True)
            llm.generate(["hi"], sp)  # warmup
            best = float("inf")
            for _ in range(REPS):
                t = time.perf_counter()
                out = llm.generate(["hi"], sp)
                dt = time.perf_counter() - t
                best = min(best, dt)
            prefill_sp = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
            t = time.perf_counter()
            llm.generate(["hi"], prefill_sp)
            prefill_dt = time.perf_counter() - t
            n_gen_actual = len(out[0].outputs[0].token_ids)
            return n_gen_actual / max(best - prefill_dt, 1e-9)

        if "pp128" in KEYS:
            result["pp128"] = bench_prefill(128)
        if "pp512" in KEYS:
            result["pp512"] = bench_prefill(512)
        if "pp2048" in KEYS:
            # 2046 words -> 2047 tokens + 1 gen = 2048 = max_model_len.
            result["pp2048"] = bench_prefill(2046)
        if "tg64" in KEYS:
            result["tg64"] = bench_decode(64)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    with open(out_path, "w") as f:
        json.dump(result, f)
    return 0 if result["error"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
