#!/usr/bin/env python3
"""2-way 2-GPU prefill/decode bench: candle (GPUs 0,1) vs llamacpp-turbo (GPUs 2,3).

Intended for models that don't fit on a single 16 GB MI50 — default model
is gemma-4-31B-it-Q4_0 (17 GB). Each backend gets 2 GPUs with pipeline /
layer-split, and they run in parallel on disjoint devices so the wall time
is max(candle, turbo) rather than sum.

Usage:  bench/gemma_2gpu.py [--model PATH] [--reps N] [--timeout SECONDS]
                            [--candle-gpus 0,1] [--turbo-gpus 2,3]
                            [--keys pp128,pp512,pp2048,tg64] [--only X]
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
    e["HIP_VISIBLE_DEVICES"] = gpus
    e["ROCR_VISIBLE_DEVICES"] = gpus
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
                 timeout: int, reps: int, keys: set[str]) -> dict:
    bin_path = str(CANDLE_ROOT / "target" / "release" / "examples" / "quantized-gemma4")
    log = log_dir / "candle.log"
    log.write_text("")
    result: dict = {k: None for k in ALL_KEYS}
    result["sub_errors"] = {}
    result["log"] = str(log)
    n_gpus = len(gpus.split(","))
    logger.info("[candle] gpus=%s (n_gpus=%d) reps=%d", gpus, n_gpus, reps)
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


def bench_turbo(model: str, gpus: str, log_dir: pathlib.Path,
                timeout: int, reps: int, keys: set[str]) -> dict:
    bin_path = f"{TURBO_BIN_DIR}/llama-bench"
    log = log_dir / "turbo.log"
    log.write_text("")
    result: dict = {k: None for k in ALL_KEYS}
    result["sub_errors"] = {}
    result["log"] = str(log)
    logger.info("[turbo] gpus=%s reps=%d", gpus, reps)
    env = env_for_gpus(gpus, extra={
        "LD_LIBRARY_PATH":
            f"{ROCM_LIB}:{ROCM_EXTRA_LIB}:{TURBO_BIN_DIR}:"
            f"{TURBO_BIN_DIR}/../src:{TURBO_BIN_DIR}/../ggml/src"
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
                    logger.info("[turbo %s] %.1f t/s", key, result[key])
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
                    logger.info("[turbo tg64] %.1f t/s", rec["avg_ts"])
                    break
            if result["tg64"] is None:
                result["sub_errors"]["tg64"] = extract_err(out, rc)
        else:
            result["sub_errors"]["tg64"] = extract_err(out, rc)

    return result


def render_md(candle: dict, turbo: dict, keys: list[str],
              model: str, elapsed: float) -> str:
    lines: list[str] = []
    lines.append(f"# gemma-2gpu bench — {pathlib.Path(model).name}  ({elapsed:.1f}s)")
    lines.append("")
    lines.append("| bench  | candle (0,1) | turbo (2,3) | c/turbo |")
    lines.append("|--------|--------------|-------------|---------|")
    for k in keys:
        c = candle.get(k)
        t = turbo.get(k)
        ratio = f"{c/t:.2f}×" if (c and t) else "—"
        c_s = f"{c:>7.1f}" if c else f"{'—':>7}"
        t_s = f"{t:>7.1f}" if t else f"{'—':>7}"
        lines.append(f"| {k:<6} | {c_s}      | {t_s}     | {ratio:>7} |")
    errs = []
    for name, r in (("candle", candle), ("turbo", turbo)):
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
    p.add_argument("--turbo-gpus", default="2,3")
    p.add_argument("--keys", default=",".join(ALL_KEYS),
                   help="Comma-separated subset of: " + ",".join(ALL_KEYS))
    p.add_argument("--only", choices=["candle", "turbo"], default=None)
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
    logger.info("candle_gpus=%s turbo_gpus=%s reps=%d keys=%s",
                args.candle_gpus, args.turbo_gpus, args.reps, sorted(keys))

    t0 = time.time()
    backends: dict[str, concurrent.futures.Future] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        if args.only != "turbo":
            backends["candle"] = ex.submit(bench_candle, args.model,
                                           args.candle_gpus, out_dir,
                                           args.timeout, args.reps, keys)
        if args.only != "candle":
            backends["turbo"] = ex.submit(bench_turbo, args.model,
                                          args.turbo_gpus, out_dir,
                                          args.timeout, args.reps, keys)
        results = {name: fut.result() for name, fut in backends.items()}
    elapsed = time.time() - t0
    logger.info("wall: %.1fs", elapsed)

    candle = results.get("candle", {k: None for k in ALL_KEYS})
    turbo = results.get("turbo",   {k: None for k in ALL_KEYS})
    md = render_md(candle, turbo, [k for k in ALL_KEYS if k in keys],
                   args.model, elapsed)
    print("\n" + md)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "gemma_2gpu_latest.md").write_text(md)
    summary = {"ts": ts, "model": args.model, "elapsed": elapsed,
               "candle": candle, "turbo": turbo}
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
