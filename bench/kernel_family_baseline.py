#!/usr/bin/env python3
"""Capture a per-kernel-family baseline JSON from a rocprofv3 summary.

Used in W8 of ROADMAP-DECODE-GAP-2026-04-16.md to record the starting-line
kernel-time distribution for a model, so later phases can gate their
changes against it via `kernel_family_regression.py`.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

from kernel_family import (
    PRESERVE,
    DEGRADE_TARGET,
    classify,
    dump_summary,
    parse_bash_time,
    parse_rocprof_summary,
    to_baseline_json,
)


def _parse_kv_list(s: str) -> dict[str, float]:
    """Parse 'k1=v1,k2=v2' into a dict of floats."""
    out: dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        out[k.strip()] = float(v.strip())
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True,
                   help="path to rocprofv3 summary.txt")
    p.add_argument("--time", type=Path,
                   help="path to bash-time output (optional; used for real/user/sys)")
    p.add_argument("--throughput-unrocprof", default="",
                   help="pp_t_s=X,tg_t_s=Y  (uninstrumented bench, optional)")
    p.add_argument("--throughput-rocprof", default="",
                   help="pp_t_s=X,tg_t_s=Y  (under rocprofv3, optional)")
    p.add_argument("--model", required=True)
    p.add_argument("--backend", choices=("candle", "turbo", "llamacpp"), required=True)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--bin", default="")
    p.add_argument("--prompt-tokens", type=int, required=True)
    p.add_argument("--gen-tokens", type=int, required=True)
    p.add_argument("--note", action="append", default=[],
                   help="annotations to include in the JSON (repeatable)")
    p.add_argument("--unmatched-pct-limit", type=float, default=1.0,
                   help="fail if unmatched kernels exceed this pct of GPU time")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--md", type=Path,
                   help="also write a Markdown snapshot here (default: out with .md suffix)")
    args = p.parse_args()

    rows = parse_rocprof_summary(args.summary)
    fams = classify(rows)

    # Sanity check: no unmatched kernel should contribute >limit% of total.
    total_usec = sum(fr.total_usec for fr in fams.values())
    other = fams.get("other")
    if other is not None and total_usec > 0:
        other_pct = 100.0 * other.total_usec / total_usec
        if other_pct > args.unmatched_pct_limit:
            print(
                f"FAIL: unmatched kernels contribute {other_pct:.2f}% of GPU time, "
                f"exceeding the {args.unmatched_pct_limit}% limit. "
                f"Unmatched kernel names: {other.kernels}\n"
                f"Extend FAMILY_RULES in kernel_family.py to cover these.",
                file=sys.stderr,
            )
            return 2

    meta: dict = {
        "captured_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "model": Path(args.model).name,
        "backend": args.backend,
        "config": {
            "gpus": args.gpus,
            "bin": args.bin,
            "prompt_tokens": args.prompt_tokens,
            "gen_tokens": args.gen_tokens,
        },
    }
    if args.throughput_unrocprof:
        meta["throughput_unrocprof"] = _parse_kv_list(args.throughput_unrocprof)
    if args.throughput_rocprof:
        meta["throughput_rocprof"] = _parse_kv_list(args.throughput_rocprof)
    if args.time:
        meta["cpu_seconds"] = parse_bash_time(args.time)
    if args.note:
        meta["notes"] = args.note

    baseline = to_baseline_json(fams, meta)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(baseline, indent=2) + "\n")

    md_path = args.md or args.out.with_suffix(".md")
    md_lines = [
        f"# Kernel-family baseline — {meta['model']} ({args.backend})",
        "",
        f"- captured_at: {meta['captured_at']}",
        f"- gpus: {args.gpus}",
        f"- prompt_tokens: {args.prompt_tokens}, gen_tokens: {args.gen_tokens}",
    ]
    if "throughput_unrocprof" in meta:
        md_lines.append(f"- throughput (no rocprof): "
                        f"{meta['throughput_unrocprof']}")
    if "throughput_rocprof" in meta:
        md_lines.append(f"- throughput (under rocprof): "
                        f"{meta['throughput_rocprof']}")
    if "cpu_seconds" in meta:
        md_lines.append(f"- cpu_seconds: {meta['cpu_seconds']}")
    md_lines += ["", "```", dump_summary(fams), "```", ""]
    md_path.write_text("\n".join(md_lines))

    print(f"wrote {args.out}")
    print(f"wrote {md_path}")
    print()
    print(dump_summary(fams))
    return 0


if __name__ == "__main__":
    sys.exit(main())
