#!/usr/bin/env python3
"""3-way 2-GPU prefill/decode bench: candle vs llamacpp-vanilla vs llamacpp-turbo.

Candle runs on its own GPU pair in parallel; llamacpp-vanilla and turbo share
a GPU pair (run sequentially since they can't both use the same GPUs
concurrently). Each backend uses pipeline/layer-split across 2 GPUs.

Usage:  bench/gemma_2gpu.py [--model PATH] [--reps N] [--timeout SECONDS]
                            [--candle-gpus 0,1] [--llama-gpus 2,3]
                            [--keys pp128,pp512,pp2048,tg64]
                            [--only candle|llamacpp|turbo]
Output: stdout markdown + bench/results/gemma_2gpu_{latest,history}.*
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as _dt
import json
import logging
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Optional

CANDLE_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "/artefact/models/gemma-4-31B-it-Q4_0.gguf"
ROCM_LIB = "/opt/rocm-7.1.1/core-7.13/lib"
ROCM_EXTRA_LIB = "/opt/rocm/lib"  # roctx lives here
TURBO_BIN_DIR = "/artefact/llamacpp-turbo/llama-cpp-gfx906-turbo/build/bin"
LLAMACPP_BIN_DIR = "/artefact/llama.cpp/build/bin"
RESULTS_DIR = CANDLE_ROOT / "bench" / "results"

SUBBENCHES = [
    # (key,    prompt_len, sample_len, turbo_p)
    ("pp128",  128,        1,          "128"),
    ("pp512",  512,        1,          "512"),
    ("pp2048", 2046,       1,          "2048"),
    ("tg64",   None,       64,         None),
]
ALL_KEYS = [k for k, *_ in SUBBENCHES]

logger = logging.getLogger("gemma_2gpu")


def configure_logging(log_dir: pathlib.Path) -> None:
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(log_dir / "bench.log")
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)


def env_for_gpus(gpus: str, extra: Optional[dict[str, str]] = None) -> dict[str, str]:
    e = os.environ.copy()
    # Use HIP_VISIBLE_DEVICES only — setting ROCR_VISIBLE_DEVICES as well
    # double-filters (ROCR remaps, then HIP tries original indices from the
    # remapped set), which breaks non-zero-starting selections like "2,3".
    e["HIP_VISIBLE_DEVICES"] = gpus
    e.pop("ROCR_VISIBLE_DEVICES", None)
    # roctx symlink path via /opt/rocm/lib (links to 7.2.1 but ABI compatible).
    e["LD_LIBRARY_PATH"] = f"{ROCM_LIB}:{ROCM_EXTRA_LIB}"
    if extra:
        e.update(extra)
    return e


def run(cmd: list[str], env: dict[str, str], timeout: int) -> tuple[int, str]:
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                           timeout=timeout)
        out = r.stdout + "\n" + r.stderr
        return r.returncode, out
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + "\n" + (e.stderr or "")
        logger.debug("timeout after %.1fs: %s", time.time() - t0, " ".join(cmd[:3]))
        return -1, out + f"\n[TIMEOUT after {timeout}s]"


def _parse_llama_bench_json(out: str) -> list[dict]:
    """llama-bench -o json prints a JSON array to stdout (mixed with
    stderr device-init chatter). Extract the [...] array and parse it."""
    # Find the outermost [ ... ] in the output.
    start = out.find("[\n")
    if start < 0:
        return []
    depth = 0
    end = -1
    for i in range(start, len(out)):
        c = out[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return []
    try:
        return json.loads(out[start:end + 1])
    except Exception:
        return []


def extract_err(out: str, rc: int) -> str:
    for line in reversed(out.splitlines()):
        s = line.strip()
        if s and not s.startswith(">") and len(s) < 300:
            return f"rc={rc} {s[:200]}"
    return f"rc={rc}"


def bench_candle(model: str, gpus: str, log_dir: pathlib.Path,
                 timeout: int, reps: int, keys: set[str],
                 bin_name: str = "quantized-gemma4",
                 extra_args: Optional[list[str]] = None) -> dict:
    bin_path = str(CANDLE_ROOT / "target" / "release" / "examples" / bin_name)
    log = log_dir / "candle.log"
    log.write_text("")
    result: dict = {k: None for k in ALL_KEYS}
    result["sub_errors"] = {}
    result["log"] = str(log)
    n_gpus = len(gpus.split(","))
    logger.info("[candle] bin=%s gpus=%s (n_gpus=%d) reps=%d",
                bin_name, gpus, n_gpus, reps)
    env = env_for_gpus(gpus, extra={"CANDLE_MMQ_TURBO_PORT": "1"})

    for key, prompt_len, sample_len, _ in SUBBENCHES:
        if key not in keys:
            continue
        prompt = " ".join(["word"] * prompt_len) if prompt_len else "hi"
        runs: list[float] = []
        last_err: Optional[str] = None
        for rep in range(reps):
            cmd = [bin_path, "--model", model, "--prompt", prompt,
                   "--sample-len", str(sample_len), "--temperature", "0",
                   "--n-gpus", str(n_gpus)]
            if extra_args:
                cmd.extend(extra_args)
            rc, out = run(cmd, env, timeout)
            with log.open("a") as f:
                f.write(f"[{key} rep{rep}] rc={rc}\n{out}\n---\n")
            if rc != 0:
                last_err = extract_err(out, rc)
                logger.debug("[candle %s rep%d] fail: %s", key, rep, last_err)
                continue
            m_pp = re.search(r"(\d+)\s+prompt tokens(?:\s+processed)?:\s+([\d.]+)\s+(?:t|token)/s", out)
            m_tg = re.search(r"(\d+)\s+tokens generated:\s+([\d.]+)\s+(?:t|token)/s", out)
            if prompt_len is None and m_tg:
                runs.append(float(m_tg.group(2)))
            elif prompt_len is not None and m_pp:
                runs.append(float(m_pp.group(2)))
            else:
                last_err = f"rc=0 no rate found; tail={out[-200:].strip()}"
        if runs:
            result[key] = max(runs)
            logger.info("[candle %s] %.1f t/s (best of %d)", key, result[key], len(runs))
        else:
            result["sub_errors"][key] = last_err or "no runs"
            logger.info("[candle %s] failed: %s", key, last_err)
    return result


def bench_llama_bench(tag: str, bin_dir: str, model: str, gpus: str,
                      log_dir: pathlib.Path, timeout: int, reps: int,
                      keys: set[str]) -> dict:
    bin_path = f"{bin_dir}/llama-bench"
    log = log_dir / f"{tag}.log"
    log.write_text("")
    result: dict = {k: None for k in ALL_KEYS}
    result["sub_errors"] = {}
    result["log"] = str(log)
    logger.info("[%s] gpus=%s reps=%d", tag, gpus, reps)
    env = env_for_gpus(gpus, extra={
        "LD_LIBRARY_PATH":
            f"{ROCM_LIB}:{ROCM_EXTRA_LIB}:{bin_dir}:"
            f"{bin_dir}/../src:{bin_dir}/../ggml/src"
    })

    pp_prompts = [str(p) for k, p, _, tp in SUBBENCHES
                  if k in keys and tp is not None]
    do_tg = "tg64" in keys

    if pp_prompts:
        args = [bin_path, "-m", model, "-ngl", "99", "-sm", "layer",
                "-r", str(reps), "-o", "json", "-n", "0"]
        for p in pp_prompts:
            args += ["-p", p]
        rc, out = run(args, env, timeout)
        with log.open("a") as f:
            f.write(f"[pp] rc={rc}\n{out}\n---\n")
        if rc == 0:
            data = _parse_llama_bench_json(out)
            for rec in data:
                if rec.get("n_gen", 1) == 0:
                    p = rec.get("n_prompt")
                    if p == 128:
                        result["pp128"] = rec["avg_ts"]
                    elif p == 512:
                        result["pp512"] = rec["avg_ts"]
                    elif p == 2048:
                        result["pp2048"] = rec["avg_ts"]
            for key in ("pp128", "pp512", "pp2048"):
                if key in keys and result[key] is not None:
                    logger.info("[%s %s] %.1f t/s", tag, key, result[key])
                elif key in keys:
                    result["sub_errors"][key] = extract_err(out, rc)
        else:
            for k in ("pp128", "pp512", "pp2048"):
                if k in keys:
                    result["sub_errors"][k] = extract_err(out, rc)

    if do_tg:
        args = [bin_path, "-m", model, "-ngl", "99", "-sm", "layer",
                "-r", str(reps), "-o", "json", "-p", "0", "-n", "64"]
        rc, out = run(args, env, timeout)
        with log.open("a") as f:
            f.write(f"[tg] rc={rc}\n{out}\n---\n")
        if rc == 0:
            for rec in _parse_llama_bench_json(out):
                if rec.get("n_gen") == 64:
                    result["tg64"] = rec["avg_ts"]
                    logger.info("[%s tg64] %.1f t/s", tag, rec["avg_ts"])
                    break
            if result["tg64"] is None:
                result["sub_errors"]["tg64"] = extract_err(out, rc)
        else:
            result["sub_errors"]["tg64"] = extract_err(out, rc)

    return result


def render_md(results: dict[str, dict], keys: list[str],
              model: str, elapsed: float, candle_gpus: str,
              llama_gpus: str) -> str:
    lines: list[str] = []
    lines.append(f"# gemma-2gpu bench — {pathlib.Path(model).name}  ({elapsed:.1f}s)")
    lines.append("")
    header = f"| bench  | candle ({candle_gpus}) | llama.cpp ({llama_gpus}) | turbo ({llama_gpus}) | c/llama | c/turbo |"
    sep = "|--------|---------------|----------------|---------------|---------|---------|"
    lines.append(header)
    lines.append(sep)
    for k in keys:
        c = results.get("candle", {}).get(k)
        ll = results.get("llamacpp", {}).get(k)
        tu = results.get("turbo", {}).get(k)
        def cell(v): return f"{v:>7.1f}" if v else f"{'—':>7}"
        def ratio(a, b):
            return f"{a/b:.2f}×" if (a and b) else "—"
        lines.append(f"| {k:<6} | {cell(c)}       | {cell(ll)}        | {cell(tu)}       | "
                     f"{ratio(c, ll):>7} | {ratio(c, tu):>7} |")
    errs = []
    for name, r in results.items():
        for k, v in r.get("sub_errors", {}).items():
            errs.append(f"- **{name}/{k}**: {v}")
    if errs:
        lines.append("")
        lines.append("## Errors")
        lines.extend(errs)
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--candle-gpus", default="0,1")
    p.add_argument("--llama-gpus", default="2,3",
                   help="GPUs shared by vanilla llama.cpp + turbo (sequential)")
    p.add_argument("--keys", default=",".join(ALL_KEYS),
                   help="Comma-separated subset of: " + ",".join(ALL_KEYS))
    p.add_argument("--only", choices=["candle", "llamacpp", "turbo"],
                   default=None)
    p.add_argument("--candle-bin", default="quantized-gemma4",
                   help="Candle example binary name "
                        "(e.g. quantized-gemma4, quantized-qwen35).")
    p.add_argument("--candle-extra", default="",
                   help="Extra CLI args for candle (whitespace-separated).")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    keys = set(k.strip() for k in args.keys.split(",") if k.strip())
    keys &= set(ALL_KEYS)
    if not keys:
        print("no valid keys", file=sys.stderr)
        return 2

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path(args.out_dir) if args.out_dir \
              else pathlib.Path(f"/tmp/gemma_2gpu_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(out_dir)
    logger.info("start ts=%s  model=%s", ts, args.model)
    logger.info("candle_gpus=%s llama_gpus=%s reps=%d keys=%s",
                args.candle_gpus, args.llama_gpus, args.reps, sorted(keys))

    def run_llama_sequence() -> dict[str, dict]:
        out: dict[str, dict] = {}
        if args.only in (None, "llamacpp"):
            out["llamacpp"] = bench_llama_bench(
                "llamacpp", LLAMACPP_BIN_DIR, args.model, args.llama_gpus,
                out_dir, args.timeout, args.reps, keys)
        if args.only in (None, "turbo"):
            out["turbo"] = bench_llama_bench(
                "turbo", TURBO_BIN_DIR, args.model, args.llama_gpus,
                out_dir, args.timeout, args.reps, keys)
        return out

    extra_args = args.candle_extra.split() if args.candle_extra else []

    t0 = time.time()
    results: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futures: dict[str, concurrent.futures.Future] = {}
        if args.only in (None, "candle"):
            futures["candle"] = ex.submit(bench_candle, args.model,
                                          args.candle_gpus, out_dir,
                                          args.timeout, args.reps, keys,
                                          args.candle_bin, extra_args)
        if args.only in (None, "llamacpp", "turbo"):
            futures["_llama_seq"] = ex.submit(run_llama_sequence)
        for name, fut in futures.items():
            if name == "_llama_seq":
                results.update(fut.result())
            else:
                results[name] = fut.result()
    elapsed = time.time() - t0
    logger.info("wall: %.1fs", elapsed)

    md = render_md(results, [k for k in ALL_KEYS if k in keys],
                   args.model, elapsed, args.candle_gpus, args.llama_gpus)
    print("\n" + md)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "gemma_2gpu_latest.md").write_text(md)
    summary = {"ts": ts, "model": args.model, "elapsed": elapsed,
               **{name: r for name, r in results.items()}}
    (RESULTS_DIR / "gemma_2gpu_latest.json").write_text(
        json.dumps(summary, indent=2))
    with (RESULTS_DIR / "gemma_2gpu_history.jsonl").open("a") as f:
        f.write(json.dumps(summary) + "\n")
    (out_dir / "summary.md").write_text(md)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("artifacts: %s   stable: %s", out_dir,
                RESULTS_DIR / "gemma_2gpu_latest.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
