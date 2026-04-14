#!/usr/bin/env python3
"""3-way prefill bench: candle vs llamacpp-turbo vs vLLM, parallel across GPUs.

Runs pp128, pp512, tg64 on TinyLlama-1.1B Q4_0. Each backend is pinned to
its own GPU (HIP_VISIBLE_DEVICES=N) so the three runs don't interfere.

Usage:  bench/prefill_3way.py [--model PATH] [--timeout SECONDS]
Output: a markdown table + JSON log in /tmp/prefill_3way_YYYYMMDD_HHMMSS/

Reusable for successive prompt-processing improvement arcs — re-run after
each kernel change to see the turbo/vLLM gap shrink.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as _dt
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time
from typing import Optional

CANDLE_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "/artefact/models/tinyllama-1.1b-q4_0.gguf"
ROCM_LIB = "/opt/rocm-7.1.1/core-7.13/lib"
TURBO_BIN_DIR = "/artefact/llamacpp-turbo/llama-cpp-gfx906-turbo/build/bin"
VLLM_PY = "/artefact/mobydick-venv/bin/python"
VLLM_SCRIPT = str(CANDLE_ROOT / "bench" / "vllm_prefill.py")


def run(cmd: list[str], env: dict[str, str], timeout: int) -> tuple[int, str]:
    """Invoke cmd; return (rc, combined_stdout_stderr)."""
    try:
        r = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        return 124, f"[TIMEOUT after {timeout}s]\n{e.stdout or ''}{e.stderr or ''}"
    except Exception as e:
        return 125, f"[ERROR {type(e).__name__}: {e}]"


def _env_for_gpu(gpu: int, extra: Optional[dict[str, str]] = None) -> dict[str, str]:
    e = os.environ.copy()
    e["HIP_VISIBLE_DEVICES"] = str(gpu)
    e["LD_LIBRARY_PATH"] = ROCM_LIB + ":" + e.get("LD_LIBRARY_PATH", "")
    if extra:
        e.update(extra)
    return e


def bench_candle(model: str, gpu: int, log_dir: pathlib.Path,
                 timeout: int) -> dict:
    """Run candle quantized example thrice each for pp128, pp512, tg64."""
    bin_path = str(CANDLE_ROOT / "target" / "release" / "examples" / "quantized")
    env = _env_for_gpu(gpu)
    log = log_dir / "candle.log"
    result = {"pp128": None, "pp512": None, "tg64": None, "error": None,
              "log": str(log)}

    def run_bench(prompt_len: Optional[int], sample_len: int) -> list[float]:
        prompt = " ".join(["word"] * prompt_len) if prompt_len else "hi"
        runs: list[float] = []
        for _ in range(3):
            cmd = [bin_path, "--model", model, "--prompt", prompt,
                   "--sample-len", str(sample_len), "--temperature", "0"]
            rc, out = run(cmd, env, timeout)
            with log.open("a") as f:
                f.write(f"$ {shlex.join(cmd)}\n(rc={rc})\n{out}\n---\n")
            if rc != 0:
                continue
            m_pp = re.search(r"(\d+)\s+prompt tokens processed:\s+([\d.]+)\s+token/s",
                             out)
            m_tg = re.search(r"(\d+)\s+tokens generated:\s+([\d.]+)\s+token/s",
                             out)
            if prompt_len is None and m_tg:
                runs.append(float(m_tg.group(2)))
            elif prompt_len is not None and m_pp:
                runs.append(float(m_pp.group(2)))
        return runs

    try:
        log.write_text("")
        pp128 = run_bench(128, 10)
        if pp128:
            result["pp128"] = max(pp128)
        pp512 = run_bench(512, 10)
        if pp512:
            result["pp512"] = max(pp512)
        tg = run_bench(None, 64)
        if tg:
            result["tg64"] = max(tg)
    except Exception as e:  # pragma: no cover
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def bench_turbo(model: str, gpu: int, log_dir: pathlib.Path,
                timeout: int) -> dict:
    """Invoke llama-bench (turbo) once; it handles reps+stats natively."""
    bin_path = f"{TURBO_BIN_DIR}/llama-bench"
    env = _env_for_gpu(gpu, {"LD_LIBRARY_PATH":
                              f"{TURBO_BIN_DIR}:{ROCM_LIB}"})
    log = log_dir / "turbo.log"
    result = {"pp128": None, "pp512": None, "tg64": None, "error": None,
              "log": str(log)}
    cmd = [bin_path, "-m", model, "-p", "128,512", "-n", "64",
           "-ngl", "99", "-t", "1", "-r", "3"]
    rc, out = run(cmd, env, timeout)
    log.write_text(f"$ {shlex.join(cmd)}\n(rc={rc})\n{out}\n")
    if rc != 0 and "token/s" not in out:
        result["error"] = f"turbo exit={rc}"
        return result
    for line in out.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 3:
            continue
        label = cells[-3] if len(cells) >= 3 else ""
        val_str = cells[-2] if len(cells) >= 2 else ""
        m = re.match(r"([\d.]+)\s*±", val_str)
        if not m:
            continue
        val = float(m.group(1))
        if label == "pp128":
            result["pp128"] = val
        elif label == "pp512":
            result["pp512"] = val
        elif label == "tg64":
            result["tg64"] = val
    return result


def bench_vllm(model: str, gpu: int, log_dir: pathlib.Path,
               timeout: int) -> dict:
    """Invoke vLLM via the helper script; captures JSON output."""
    env = _env_for_gpu(gpu, {"HF_HUB_OFFLINE": "1"})
    log = log_dir / "vllm.log"
    json_out = log_dir / "vllm_result.json"
    result = {"pp128": None, "pp512": None, "tg64": None, "error": None,
              "log": str(log)}
    cmd = [VLLM_PY, VLLM_SCRIPT, model, str(json_out)]
    rc, out = run(cmd, env, timeout)
    log.write_text(f"$ {shlex.join(cmd)}\n(rc={rc})\n{out}\n")
    if json_out.exists():
        try:
            parsed = json.loads(json_out.read_text())
            result.update({k: parsed.get(k) for k in
                           ("pp128", "pp512", "tg64", "error")})
        except Exception as e:  # pragma: no cover
            result["error"] = f"vllm json parse: {e}"
    elif rc == 124:
        result["error"] = "timeout"
    else:
        result["error"] = f"vllm exit={rc}, no json"
    return result


def format_table(results: dict[str, dict]) -> str:
    def cell(v):
        return f"{v:.0f}" if isinstance(v, (int, float)) else "—"
    header = "| bench  | candle | turbo  | vllm   | candle/turbo |\n"
    header += "|--------|--------|--------|--------|--------------|\n"
    rows = []
    for k in ("pp128", "pp512", "tg64"):
        c = results["candle"].get(k)
        t = results["turbo"].get(k)
        v = results["vllm"].get(k)
        ratio = (f"{c / t:.2f}×"
                 if isinstance(c, (int, float)) and isinstance(t, (int, float))
                    and t > 0 else "—")
        rows.append(f"| {k:<6} | {cell(c):>6} | {cell(t):>6} | "
                    f"{cell(v):>6} | {ratio:>12} |")
    return header + "\n".join(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--timeout", type=int, default=600,
                   help="per-backend timeout in seconds")
    p.add_argument("--out-dir", default=None,
                   help="log directory (default /tmp/prefill_3way_<ts>)")
    p.add_argument("--only", choices=["candle", "turbo", "vllm"],
                   help="restrict to one backend (for debugging)")
    args = p.parse_args()

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = pathlib.Path(args.out_dir or f"/tmp/prefill_3way_{ts}")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"log dir: {log_dir}", flush=True)

    if not pathlib.Path(args.model).exists():
        print(f"model missing: {args.model}", file=sys.stderr)
        return 1

    jobs = {
        "candle": (bench_candle, 0),
        "turbo":  (bench_turbo,  1),
        "vllm":   (bench_vllm,   2),
    }
    if args.only:
        jobs = {args.only: jobs[args.only]}

    t0 = time.perf_counter()
    results: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        future_to_name = {
            pool.submit(fn, args.model, gpu, log_dir, args.timeout): name
            for name, (fn, gpu) in jobs.items()
        }
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
            except Exception as e:  # pragma: no cover
                results[name] = {"error": f"{type(e).__name__}: {e}"}
            done = results[name]
            err = done.get("error")
            if err:
                print(f"[{name}] FAILED: {err}", flush=True)
            else:
                print(f"[{name}] OK pp128={done.get('pp128')} "
                      f"pp512={done.get('pp512')} tg64={done.get('tg64')}",
                      flush=True)

    # Default missing keys to None so the table renderer works with --only.
    for name in ("candle", "turbo", "vllm"):
        results.setdefault(name, {"pp128": None, "pp512": None, "tg64": None,
                                   "error": "skipped"})

    wall = time.perf_counter() - t0
    summary = {
        "timestamp": ts, "model": args.model,
        "wall_seconds": round(wall, 1), "results": results,
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    table = format_table(results)
    (log_dir / "summary.md").write_text(f"# Prefill 3-way bench {ts}\n\n"
                                         f"Model: `{args.model}`\n"
                                         f"Wall: {wall:.1f}s\n\n{table}\n")
    print()
    print(table)
    print(f"\nlogs + summary.{{json,md}} in {log_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
