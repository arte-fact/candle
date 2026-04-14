"""vLLM prefill/decode bench — invoked by prefill_3way.py.

Usage: vllm_prefill.py <model.gguf> <output.json>
Writes {"pp128": tps, "pp512": tps, "tg64": tps, "error": str|None}
"""
import json
import os
import sys
import time
import traceback

# Must be set BEFORE importing vllm.
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: vllm_prefill.py <model.gguf> <output.json>", file=sys.stderr)
        return 2
    model = sys.argv[1]
    out_path = sys.argv[2]
    result = {"pp128": None, "pp512": None, "tg64": None, "error": None}
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
        )

        def bench_prefill(n_prompt: int) -> float:
            prompt = " ".join(["word"] * n_prompt)
            sp = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
            llm.generate([prompt], sp)  # warmup
            best = float("inf")
            for _ in range(3):
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
            for _ in range(3):
                t = time.perf_counter()
                out = llm.generate(["hi"], sp)
                dt = time.perf_counter() - t
                best = min(best, dt)
            prefill_sp = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
            t = time.perf_counter()
            llm.generate(["hi"], prefill_sp)
            prefill_dt = time.perf_counter() - t
            n_gen_actual = len(out[0].outputs[0].token_ids)
            tg = n_gen_actual / max(best - prefill_dt, 1e-9)
            return tg

        result["pp128"] = bench_prefill(128)
        result["pp512"] = bench_prefill(512)
        result["tg64"] = bench_decode(64)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    with open(out_path, "w") as f:
        json.dump(result, f)
    return 0 if result["error"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
