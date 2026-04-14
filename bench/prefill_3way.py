#!/usr/bin/env python3
"""4-way prefill/decode bench: candle vs llama.cpp (vanilla) vs llamacpp-turbo vs vLLM.

Each backend is pinned to a single GPU (apples-to-apples; no multi-GPU
overhead inside candle). Backends run in parallel on different GPUs.
candle sub-benches (pp128/pp512/pp2048/tg64) are serial inside the
backend for measurement stability.

Usage:  bench/prefill_3way.py [--model PATH] [--timeout SECONDS]
                              [--only X] [--gpus 0,1,2,3]
                              [--reps N] [--quick]
Output: /tmp/prefill_3way_YYYYMMDD_HHMMSS/
        - bench.log          structured timestamped log (DEBUG)
        - {candle,turbo,vllm}.log  full stdout/stderr of each subprocess
        - summary.{md,json}  table + per-cell errors
        bench/results/latest.{md,json} + history.jsonl mirror the run.

Error-recovery: every subprocess invocation is wrapped; timeouts and
non-zero exits are caught and recorded per sub-bench. A backend crash
doesn't abort the other two; a sub-bench failure doesn't abort peers.
Each result carries sub_errors[key] -> short reason string, appended
as an "Errors" section in the markdown summary.

VRAM leak detection: rocm-smi is sampled before and after each backend
runs; a >100 MB delta after backend exit is flagged in bench.log.

Env-var notes baked into the script:
  - candle/turbo:  LD_LIBRARY_PATH -> /opt/rocm-7.1.1/core-7.13/lib
  - vLLM (mobydick): LD_LIBRARY_PATH -> /opt/rocm-6.3.4/lib, ROCM_PATH,
    PYTORCH_ROCM_ARCH=gfx906, FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE.
    Mixing 7.1.1 libs into the mobydick venv segfaults torch._dynamo
    during import. See BENCH-3WAY-2026-04-11.md.
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
import shlex
import subprocess
import sys
import time
import traceback
from typing import Optional

CANDLE_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "/artefact/models/tinyllama-1.1b-q4_0.gguf"
ROCM_LIB = "/opt/rocm-7.1.1/core-7.13/lib"
ROCM_SMI = "/opt/rocm-7.1.1/core-7.13/bin/rocm-smi"
TURBO_BIN_DIR = "/artefact/llamacpp-turbo/llama-cpp-gfx906-turbo/build/bin"
LLAMACPP_BIN_DIR = "/artefact/llama.cpp/build/bin"
VLLM_PY = "/artefact/mobydick-venv/bin/python"
VLLM_SCRIPT = str(CANDLE_ROOT / "bench" / "vllm_prefill.py")
RESULTS_DIR = CANDLE_ROOT / "bench" / "results"
VRAM_LEAK_MB = 100  # warn if a backend leaks this much above baseline

# sample_len=1 for prefill keeps the total under TinyLlama's 2048 ctx at
# pp2048 (prompt 2046 -> 2047 tokens + BOS + 1 gen = 2048). turbo runs
# its own -p N prefill-only, ignoring sample_len.
SUBBENCHES = [
    # (key,    prompt_len, sample_len, turbo_p)
    ("pp128",  128,        1,          "128"),
    ("pp512",  512,        1,          "512"),
    ("pp2048", 2046,       1,          "2048"),
    ("tg64",   None,       64,         None),
]
QUICK_KEYS = {"pp128", "pp512", "tg64"}  # --quick drops the long one
ALL_KEYS = [k for k, *_ in SUBBENCHES]

logger = logging.getLogger("prefill_3way")


def configure_logging(log_dir: pathlib.Path) -> None:
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(log_dir / "bench.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-5s %(threadName)-12s %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(message)s",
                                       datefmt="%H:%M:%S"))
    logger.addHandler(ch)


_ERROR_PATTERNS = [
    re.compile(r"^(Error:.*)$", re.M),
    re.compile(r"^(thread '[^']+' panicked at.*)$", re.M),
    re.compile(r"^(rocBLAS error.*)$", re.M),
    re.compile(r"^(RuntimeError:.*)$", re.M),
    re.compile(r"^(ValueError:.*)$", re.M),
    re.compile(r"^(.*out of memory.*)$", re.M | re.I),
    re.compile(r"^(.*Segmentation fault.*)$", re.M),
    re.compile(r"^(\[TIMEOUT.*?\])", re.M),
]


def extract_error(out: str, rc: int) -> str:
    for pat in _ERROR_PATTERNS:
        m = pat.search(out)
        if m:
            line = m.group(1).strip()
            return line[:200] + ("…" if len(line) > 200 else "")
    if rc == -11:
        return f"rc={rc} (SIGSEGV)"
    if rc == -6:
        return f"rc={rc} (SIGABRT)"
    if rc == 124:
        return f"rc={rc} (TIMEOUT)"
    return f"rc={rc}"


def run(cmd: list[str], env: dict[str, str], timeout: int) -> tuple[int, str]:
    """Invoke cmd; return (rc, combined_stdout_stderr). Never raises."""
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


def vram_snapshot() -> dict[int, int]:
    """{physical_gpu_idx: used_MB} via rocm-smi. Empty dict on failure."""
    try:
        r = subprocess.run([ROCM_SMI, "--showmeminfo", "vram", "--json"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return {}
        data = json.loads(r.stdout)
    except Exception:
        return {}
    out: dict[int, int] = {}
    for k, v in data.items():
        m = re.search(r"card(\d+)", k)
        if not m:
            continue
        try:
            used_b = int(v.get("VRAM Total Used Memory (B)", 0))
        except (ValueError, TypeError):
            continue
        out[int(m.group(1))] = used_b // (1024 * 1024)
    return out


def _empty_result(log: pathlib.Path) -> dict:
    r: dict = {k: None for k in ALL_KEYS}
    r["error"] = None
    r["sub_errors"] = {}
    r["log"] = str(log)
    return r


def _candle_example_for(model: str) -> str:
    name = pathlib.Path(model).name.lower()
    examples_dir = CANDLE_ROOT / "target" / "release" / "examples"
    if "gemma-4" in name or "gemma4" in name:
        return str(examples_dir / "quantized-gemma4")
    if "qwen3.5" in name or "qwen35" in name or "qwen3-next" in name \
            or "qwen3next" in name:
        return str(examples_dir / "quantized-qwen35")
    if "qwen3-moe" in name or "qwen3_moe" in name or "qwen3-coder" in name:
        return str(examples_dir / "quantized-qwen3-moe")
    if "qwen3" in name:
        return str(examples_dir / "quantized-qwen3")
    return str(examples_dir / "quantized")


def _iter_subbenches(keys: set[str]):
    for entry in SUBBENCHES:
        if entry[0] in keys:
            yield entry


def bench_candle(model: str, gpu: int, log_dir: pathlib.Path,
                 timeout: int, reps: int, keys: set[str]) -> dict:
    """Run candle's quantized example on one GPU, sub-benches serial."""
    bin_path = _candle_example_for(model)
    log = log_dir / "candle.log"
    log.write_text("")
    result = _empty_result(log)
    logger.info("[candle] binary=%s gpu=%d reps=%d",
                pathlib.Path(bin_path).name, gpu, reps)

    env = _env_for_gpu(gpu)

    for key, prompt_len, sample_len, _ in _iter_subbenches(keys):
        prompt = " ".join(["word"] * prompt_len) if prompt_len else "hi"
        runs: list[float] = []
        last_err: Optional[str] = None
        for rep in range(reps):
            cmd = [bin_path, "--model", model, "--prompt", prompt,
                   "--sample-len", str(sample_len), "--temperature", "0"]
            rc, out = run(cmd, env, timeout)
            display_cmd = shlex.join([bin_path, "--model", model, "--prompt",
                                      f"<{len(prompt)}ch>", "--sample-len",
                                      str(sample_len), "--temperature", "0"])
            with log.open("a") as f:
                f.write(f"[gpu{gpu} {key} rep{rep}]$ {display_cmd}\n"
                        f"(rc={rc})\n{out}\n---\n")
            if rc != 0:
                last_err = extract_error(out, rc)
                logger.debug("[candle %s rep%d] fail: %s", key, rep, last_err)
                continue
            # Two output styles in candle examples:
            #   llama-style: "128 prompt tokens processed: 1420 token/s"
            #   qwen35-style: "136 prompt tokens: 265 t/s"
            m_pp = re.search(
                r"(\d+)\s+prompt tokens(?:\s+processed)?:\s+([\d.]+)\s+(?:t|token)/s",
                out)
            m_tg = re.search(
                r"(\d+)\s+tokens generated:\s+([\d.]+)\s+(?:t|token)/s",
                out)
            if prompt_len is None and m_tg:
                runs.append(float(m_tg.group(2)))
            elif prompt_len is not None and m_pp:
                runs.append(float(m_pp.group(2)))
            else:
                last_err = "output parsed but no rate line"
        if runs:
            result[key] = max(runs)
            logger.info("[candle %s] %.1f t/s (best of %d)", key,
                        result[key], len(runs))
        else:
            result["sub_errors"][key] = last_err or "no successful run"
            logger.warning("[candle %s] failed: %s", key,
                           result["sub_errors"][key])
    return result


def bench_turbo(model: str, gpu: int, log_dir: pathlib.Path,
                timeout: int, reps: int, keys: set[str]) -> dict:
    """llama-bench handles multiple -p values and reps natively."""
    bin_path = f"{TURBO_BIN_DIR}/llama-bench"
    env = _env_for_gpu(gpu, {"LD_LIBRARY_PATH":
                              f"{TURBO_BIN_DIR}:{ROCM_LIB}"})
    log = log_dir / "turbo.log"
    result = _empty_result(log)
    p_list = ",".join(p for k, _, _, p in SUBBENCHES
                      if p is not None and k in keys)
    run_tg = "tg64" in keys
    cmd = [bin_path, "-m", model, "-ngl", "99", "-t", "1", "-r", str(reps)]
    if p_list:
        cmd += ["-p", p_list]
    cmd += ["-n", "64" if run_tg else "0"]
    logger.info("[turbo] gpu=%d prompts=%s tg=%s reps=%d",
                gpu, p_list or "-", run_tg, reps)
    rc, out = run(cmd, env, timeout)
    log.write_text(f"$ {shlex.join(cmd)}\n(rc={rc})\n{out}\n")
    if rc != 0 and "token/s" not in out:
        result["error"] = extract_error(out, rc)
        logger.error("[turbo] %s", result["error"])
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
        if label in keys:
            result[label] = val
            logger.info("[turbo %s] %.1f t/s", label, val)
    for k in keys:
        if result[k] is None:
            result["sub_errors"][k] = "not in llama-bench output"
    return result


def bench_llamacpp(model: str, gpu: int, log_dir: pathlib.Path,
                   timeout: int, reps: int, keys: set[str]) -> dict:
    """Vanilla llama.cpp (upstream) llama-bench — identical protocol to turbo."""
    bin_path = f"{LLAMACPP_BIN_DIR}/llama-bench"
    env = _env_for_gpu(gpu, {"LD_LIBRARY_PATH":
                              f"{LLAMACPP_BIN_DIR}:{ROCM_LIB}"})
    log = log_dir / "llamacpp.log"
    result = _empty_result(log)
    p_list = ",".join(p for k, _, _, p in SUBBENCHES
                      if p is not None and k in keys)
    run_tg = "tg64" in keys
    cmd = [bin_path, "-m", model, "-ngl", "99", "-t", "1", "-r", str(reps)]
    if p_list:
        cmd += ["-p", p_list]
    cmd += ["-n", "64" if run_tg else "0"]
    logger.info("[llamacpp] gpu=%d prompts=%s tg=%s reps=%d",
                gpu, p_list or "-", run_tg, reps)
    rc, out = run(cmd, env, timeout)
    log.write_text(f"$ {shlex.join(cmd)}\n(rc={rc})\n{out}\n")
    if rc != 0 and "token/s" not in out:
        result["error"] = extract_error(out, rc)
        logger.error("[llamacpp] %s", result["error"])
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
        if label in keys:
            result[label] = val
            logger.info("[llamacpp %s] %.1f t/s", label, val)
    for k in keys:
        if result[k] is None:
            result["sub_errors"][k] = "not in llama-bench output"
    return result


def bench_vllm(model: str, gpu: int, log_dir: pathlib.Path,
               timeout: int, reps: int, keys: set[str]) -> dict:
    env = _env_for_gpu(gpu, {
        "HF_HUB_OFFLINE": "1",
        "LD_LIBRARY_PATH": "/opt/rocm-6.3.4/lib",
        "ROCM_PATH": "/opt/rocm-6.3.4",
        "PYTORCH_ROCM_ARCH": "gfx906",
        "FLASH_ATTENTION_TRITON_AMD_ENABLE": "TRUE",
        "VLLM_BENCH_REPS": str(reps),
        "VLLM_BENCH_KEYS": ",".join(sorted(keys)),
    })
    log = log_dir / "vllm.log"
    json_out = log_dir / "vllm_result.json"
    result = _empty_result(log)
    cmd = [VLLM_PY, VLLM_SCRIPT, model, str(json_out)]
    logger.info("[vllm] gpu=%d reps=%d keys=%s", gpu, reps, sorted(keys))
    rc, out = run(cmd, env, timeout)
    log.write_text(f"$ {shlex.join(cmd)}\n(rc={rc})\n{out}\n")
    if json_out.exists():
        try:
            parsed = json.loads(json_out.read_text())
            for k in keys:
                v = parsed.get(k)
                if v is not None:
                    result[k] = v
                    logger.info("[vllm %s] %.1f t/s", k, v)
                else:
                    result["sub_errors"][k] = "not reported by vllm_prefill.py"
            if parsed.get("error"):
                result["error"] = parsed["error"].splitlines()[0][:200]
        except Exception as e:
            result["error"] = f"vllm json parse: {e}"
            logger.exception("[vllm] JSON parse failure")
    else:
        result["error"] = extract_error(out, rc)
        logger.error("[vllm] %s", result["error"])
    return result


def format_table(results: dict[str, dict], keys: list[str]) -> str:
    def cell(v):
        return f"{v:.0f}" if isinstance(v, (int, float)) else "—"

    def ratio(a, b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b > 0:
            return f"{a / b:.2f}×"
        return "—"

    header = ("| bench  | candle | llama.cpp | turbo  | vllm   "
              "| c/llama | c/turbo | c/vllm |\n")
    header += ("|--------|--------|-----------|--------|--------"
               "|---------|---------|--------|\n")
    rows = []
    for k in keys:
        c  = results["candle"].get(k)
        lc = results["llamacpp"].get(k)
        t  = results["turbo"].get(k)
        v  = results["vllm"].get(k)
        rows.append(
            f"| {k:<6} | {cell(c):>6} | {cell(lc):>9} | {cell(t):>6} | "
            f"{cell(v):>6} | {ratio(c, lc):>7} | {ratio(c, t):>7} | "
            f"{ratio(c, v):>6} |")
    return header + "\n".join(rows)


def format_errors(results: dict[str, dict]) -> str:
    lines = []
    for name in ("candle", "llamacpp", "turbo", "vllm"):
        r = results.get(name, {})
        be = r.get("error")
        if be == "skipped":
            continue  # --only X leaves peers as "skipped"; not an error
        subs = r.get("sub_errors", {}) or {}
        if not be and not subs:
            continue
        lines.append(f"### {name}")
        if be:
            lines.append(f"- **backend error**: {be}")
        for k, msg in subs.items():
            lines.append(f"- `{k}`: {msg}")
    if not lines:
        return ""
    return "## Errors\n\n" + "\n".join(lines) + "\n"


def render_and_save(results: dict[str, dict], keys: list[str],
                    log_dir: pathlib.Path, ts: str,
                    model: str, wall: float) -> str:
    table = format_table(results, keys)
    errors = format_errors(results)
    body = (f"# Prefill 3-way bench {ts}\n\n"
            f"Model: `{model}`\n"
            f"Wall: {wall:.1f}s\n\n"
            f"{table}\n\n"
            f"{errors}")
    summary_json = {
        "timestamp": ts, "model": model,
        "wall_seconds": round(wall, 1), "results": results,
    }
    (log_dir / "summary.json").write_text(json.dumps(summary_json, indent=2))
    (log_dir / "summary.md").write_text(body)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "latest.md").write_text(body)
    (RESULTS_DIR / "latest.json").write_text(json.dumps(summary_json, indent=2))
    with (RESULTS_DIR / "history.jsonl").open("a") as f:
        f.write(json.dumps(summary_json) + "\n")
    return table + ("\n\n" + errors if errors else "")


def run_backend(name: str, fn, model: str, gpu: int,
                log_dir: pathlib.Path, timeout: int, reps: int,
                keys: set[str]) -> dict:
    """Wrap a backend invocation: VRAM snapshot, try/except, log."""
    logger.info("[%s] starting (gpu=%d)", name, gpu)
    before = vram_snapshot()
    if before:
        logger.info("[%s] pre-run VRAM: %s MB", name, before)
    t0 = time.perf_counter()
    try:
        res = fn(model, gpu, log_dir, timeout, reps, keys)
    except Exception as e:
        logger.exception("[%s] backend raised", name)
        res = _empty_result(log_dir / f"{name}.log")
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()
    dt = time.perf_counter() - t0
    after = vram_snapshot()
    if after:
        logger.info("[%s] post-run VRAM: %s MB", name, after)
        # Only check THIS backend's GPU: parallel backends legitimately
        # hold VRAM on their own GPUs when we sample mid-run.
        delta = after.get(gpu, 0) - before.get(gpu, 0)
        if delta >= VRAM_LEAK_MB:
            logger.warning("[%s] VRAM leak on gpu%d: +%d MB (threshold %d)",
                           name, gpu, delta, VRAM_LEAK_MB)
            res["vram_leak_mb"] = {gpu: delta}
    logger.info("[%s] finished in %.1fs", name, dt)
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--timeout", type=int, default=600,
                   help="per-backend subprocess timeout in seconds")
    p.add_argument("--out-dir", default=None,
                   help="log directory (default /tmp/prefill_3way_<ts>)")
    p.add_argument("--only", choices=["candle", "llamacpp", "turbo", "vllm"],
                   help="restrict to one backend")
    p.add_argument("--gpus", default="0,1,2,3",
                   help="comma-separated GPU indices to assign "
                        "(one per backend, round-robin)")
    p.add_argument("--reps", type=int, default=3,
                   help="reps per sub-bench (default 3)")
    p.add_argument("--quick", action="store_true",
                   help="fast-iteration mode: reps=1, drop pp2048")
    args = p.parse_args()

    if args.quick:
        args.reps = 1
        keys = set(QUICK_KEYS)
    else:
        keys = set(ALL_KEYS)

    all_gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    if not all_gpus:
        print("no GPUs supplied via --gpus", file=sys.stderr)
        return 1

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = pathlib.Path(args.out_dir or f"/tmp/prefill_3way_{ts}")
    log_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_dir)

    logger.info("prefill_3way start ts=%s", ts)
    logger.info("log dir: %s", log_dir)
    logger.info("model:   %s", args.model)
    logger.info("keys:    %s  reps=%d  quick=%s",
                sorted(keys), args.reps, args.quick)

    if not pathlib.Path(args.model).exists():
        logger.error("model missing: %s", args.model)
        return 1

    all_backends = {
        "candle":   bench_candle,
        "llamacpp": bench_llamacpp,
        "turbo":    bench_turbo,
        "vllm":     bench_vllm,
    }
    if args.only:
        backends = {args.only: all_backends[args.only]}
    else:
        backends = all_backends
    # One GPU per backend, round-robin across --gpus.
    gpu_assign = {name: all_gpus[i % len(all_gpus)]
                  for i, name in enumerate(backends)}
    logger.info("gpu assignment: %s", gpu_assign)

    # Sanity: distinct GPUs for parallel backends (avoid contention).
    if len(set(gpu_assign.values())) < len(backends):
        logger.warning("multiple backends share a GPU — results may be noisy")

    t0 = time.perf_counter()
    results: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(backends),
            thread_name_prefix="backend") as pool:
        future_to_name = {
            pool.submit(run_backend, name, fn, args.model,
                        gpu_assign[name], log_dir, args.timeout,
                        args.reps, keys): name
            for name, fn in backends.items()
        }
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
            except Exception as e:
                logger.exception("[%s] future result raised", name)
                results[name] = _empty_result(log_dir / f"{name}.log")
                results[name]["error"] = f"{type(e).__name__}: {e}"

    for name in ("candle", "llamacpp", "turbo", "vllm"):
        results.setdefault(name,
                           _empty_result(log_dir / f"{name}.log") |
                           {"error": "skipped"})

    wall = time.perf_counter() - t0
    display_keys = [k for k in ALL_KEYS if k in keys]
    table = render_and_save(results, display_keys, log_dir, ts,
                            args.model, wall)
    logger.info("wall: %.1fs", wall)
    print()
    print(table)
    print(f"\nlogs + summary.{{json,md}} in {log_dir}")
    print(f"structured log:  {log_dir / 'bench.log'}")
    print(f"stable pointer:  {RESULTS_DIR / 'latest.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
