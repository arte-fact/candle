#!/usr/bin/env python3
"""Check a rocprofv3 summary against a committed kernel-family baseline.

Exit 0 — every PRESERVE family within `--threshold-preserve`, aggregate within
         `--threshold-aggregate`.
Exit 1 — a PRESERVE family regressed beyond the threshold.
Exit 2 — setup error (missing baseline, unmatched kernel > limit, etc.).

Used as the per-PR gate for W1..W7 in ROADMAP-DECODE-GAP-2026-04-16.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kernel_family import (
    PRESERVE,
    DEGRADE_TARGET,
    classify,
    parse_rocprof_summary,
)


def _pct(delta: float, base: float) -> float:
    return 100.0 * delta / base if base else 0.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True,
                   help="kernel-family baseline JSON (from kernel_family_baseline.py)")
    p.add_argument("--summary", type=Path, required=True,
                   help="new rocprofv3 summary.txt to check")
    p.add_argument("--threshold-preserve", type=float, default=0.10,
                   help="per-family regression limit for PRESERVE set (fraction)")
    p.add_argument("--threshold-aggregate", type=float, default=0.05,
                   help="aggregate GPU time regression limit (fraction)")
    p.add_argument("--unmatched-pct-limit", type=float, default=1.0,
                   help="fail if unmatched kernels exceed this pct of GPU time")
    args = p.parse_args()

    if not args.baseline.exists():
        print(f"baseline not found: {args.baseline}", file=sys.stderr)
        return 2

    baseline = json.loads(args.baseline.read_text())
    base_families: dict = baseline["families"]
    base_aggregate: float = baseline["aggregate_gpu_usec"]

    rows = parse_rocprof_summary(args.summary)
    new_fams = classify(rows)
    new_aggregate = sum(fr.total_usec for fr in new_fams.values())

    # Unmatched check.
    other = new_fams.get("other")
    if other is not None and new_aggregate > 0:
        other_pct = 100.0 * other.total_usec / new_aggregate
        if other_pct > args.unmatched_pct_limit:
            print(
                f"FAIL: unmatched kernels contribute {other_pct:.2f}% (>{args.unmatched_pct_limit}%). "
                f"Unmatched kernels: {other.kernels}\n"
                f"Extend FAMILY_RULES in kernel_family.py.",
                file=sys.stderr,
            )
            return 2

    # Per-family diff.
    rows_out: list[tuple[str, float, float, float, str]] = []
    regression = False
    all_families = sorted(set(base_families) | set(new_fams))

    for family in all_families:
        base_us = float(base_families.get(family, {}).get("total_usec", 0.0))
        new_us = (new_fams[family].total_usec if family in new_fams else 0.0)
        delta_us = new_us - base_us
        delta_pct = _pct(delta_us, base_us if base_us else new_us)
        # Use a 1% noise band around zero to avoid labeling float rounding
        # as a change.
        noise_band = 0.01
        if family in PRESERVE:
            if base_us > 0 and delta_us > args.threshold_preserve * base_us:
                status = "FAIL (preserve regressed)"
                regression = True
            else:
                status = "ok (preserve)"
        elif family in DEGRADE_TARGET:
            if abs(delta_pct) < 100.0 * noise_band:
                status = "unchanged (target)"
            elif delta_us < 0:
                status = "IMPROVED (target)"
            else:
                status = "worse (target)"
        else:
            status = "neutral"
        rows_out.append((family, base_us, new_us, delta_pct, status))

    agg_delta_pct = _pct(new_aggregate - base_aggregate, base_aggregate)
    agg_status = "ok"
    if new_aggregate > base_aggregate * (1.0 + args.threshold_aggregate):
        agg_status = "FAIL (aggregate regressed)"
        regression = True

    # Print table.
    rows_out.sort(key=lambda r: r[3])  # by delta_pct ascending (improvements first)
    width = max(len(r[0]) for r in rows_out) + 2
    print(f"{'family'.ljust(width)} {'base_ms':>10} {'new_ms':>10} {'delta%':>8}   status")
    print("-" * (width + 52))
    for family, base_us, new_us, delta_pct, status in rows_out:
        print(
            f"{family.ljust(width)} {base_us/1000:>10.2f} {new_us/1000:>10.2f} "
            f"{delta_pct:>+7.1f}%   {status}"
        )
    print("-" * (width + 52))
    print(
        f"{'aggregate_gpu'.ljust(width)} "
        f"{base_aggregate/1000:>10.2f} {new_aggregate/1000:>10.2f} "
        f"{agg_delta_pct:>+7.1f}%   {agg_status}"
    )

    if regression:
        print("\nREGRESSION DETECTED.", file=sys.stderr)
        return 1
    print("\nAll preserve-families within threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
