"""Kernel-family classification for rocprofv3 summary outputs.

Parses the `ROCPROFV3 KERNEL_DISPATCH SUMMARY` section of a
rocprofv3 --summary-output-file text dump, groups each kernel into a
semantic family, and returns per-family totals (calls, total_usec).

Used by `kernel_family_baseline.py` (capture a baseline JSON) and
`kernel_family_regression.py` (gate a change by diffing vs baseline).

See ROADMAP-DECODE-GAP-2026-04-16.md §W8 for the design rationale.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

# Ordered family rules: first match wins. Order matters when a substring
# could match multiple rules.
FAMILY_RULES: list[tuple[str, re.Pattern]] = [
    ("moe_fused",        re.compile(r"indexed_moe_forward_q8_0_q8_1")),
    ("moe_combine",      re.compile(r"moe_combine_weighted_sum_f32")),
    ("moe_gelu_quant",   re.compile(r"gelu_mul_quantize_q8_1_f32")),
    ("flash_decode",     re.compile(r"gqa_decode_split_(chunk|combine)_d\d+_f32")),
    ("router_topk",      re.compile(
        r"gemma4_router_prep_f32|topk_moe_norm_f32|fast_sum_f32|topk_moe_cuda|mm_ids_helper")),
    ("softmax_attn",     re.compile(r"masked_softmax_scale_f32|\bsoft_max_f32\b|\bsoftmax_f32\b")),
    ("rocblas_fallback", re.compile(r"\bCijk_|gfx906_sgemm_tiled_fast")),
    ("mmq_q8_prefill",   re.compile(
        r"mul_mat_q8_0_gfx906_v2f_tile32|mul_mat_q<\(ggml_type\)8")),
    ("mmvq_q8",          re.compile(
        r"mul_mat_vec_q8_0_q8_1_cuda1|mul_mat_vec_q8_0_q8_1_moe_warp_coop|"
        r"mul_mat_vec_q<\(ggml_type\)8|"
        r"gfx906_mul_mat_vec_q8_0_warp_coop|mul_mat_vec_f<")),
    ("rmsnorm",          re.compile(r"\brmsnorm_|\brms_norm_f32\b")),
    ("activation",       re.compile(
        r"\bugelu_f32\b|unary_gated_op|"
        r"silu_mul_split_last_f32|gelu_mul_split_last_f32")),
    ("quant_q8_1",       re.compile(r"quantize_q8_1|quantize_mmq_q8_1")),
    ("rope",             re.compile(r"\brope_f32\b|rope_neox")),
    ("kv_index",         re.compile(r"k_set_rows|k_get_rows_float")),
    ("sampler_gpu",      re.compile(
        r"fast_argmax_f32|cast_f32_f64|cast_f64_f32|utanh_f64|affine_f64")),
    ("bookkeeping",      re.compile(
        r"\bu?copy(2d)?_f32\b|__amd_rocclr_(copyBuffer|fillBufferAligned)|"
        r"\bpad_f32\b|cpy_scalar|convert_unary|k_compute_batched_ptrs")),
    ("elementwise",      re.compile(
        r"\b(bmul|bdiv|badd|affine|is_u32|usqr|usqrt|scale|softcap)_f32\b|"
        r"k_bin_bcast")),
]

# Candle's winning families: W1-W7 work must not regress these > 10%.
PRESERVE: set[str] = {
    "mmvq_q8",
    "mmq_q8_prefill",
    "flash_decode",
    "softmax_attn",
    "rmsnorm",
    "activation",
    "quant_q8_1",
    "rope",
}

# Families that W1-W7 should actively shrink. Regressions here are fine to
# flag but not a gate failure.
DEGRADE_TARGET: set[str] = {
    "moe_fused",
    "moe_combine",
    "moe_gelu_quant",
    "bookkeeping",
    "elementwise",
    "router_topk",
}


@dataclass
class KernelRow:
    name: str
    calls: int
    total_usec: float
    avg_usec: float
    pct: float
    min_usec: float
    max_usec: float

    def warmup_trim(self, threshold: float = 100.0) -> float:
        """If a kernel has one big outlier (cold-init / first-call), subtract
        (max - avg) from total to approximate a steady-state total. Returns
        the trimmed total_usec."""
        if self.avg_usec > 0 and self.max_usec > threshold * self.avg_usec:
            return max(0.0, self.total_usec - (self.max_usec - self.avg_usec))
        return self.total_usec


@dataclass
class FamilyRow:
    family: str
    calls: int = 0
    total_usec: float = 0.0
    kernels: list[str] = field(default_factory=list)

    @property
    def avg_usec(self) -> float:
        return self.total_usec / self.calls if self.calls else 0.0


def classify_kernel(name: str) -> str:
    """Map a kernel name to a family. Returns 'other' if no rule matched."""
    for family, pattern in FAMILY_RULES:
        if pattern.search(name):
            return family
    return "other"


_TABLE_LINE = re.compile(r"^\|(.+)\|\s*$")
_SEPARATOR = re.compile(r"^\|[-\s|]+\|\s*$")


def parse_rocprof_summary(summary_path: Path) -> list[KernelRow]:
    """Return all KernelRows from the KERNEL_DISPATCH SUMMARY section."""
    text = Path(summary_path).read_text()
    # Find the KERNEL_DISPATCH section.
    m = re.search(
        r"ROCPROFV3 KERNEL_DISPATCH SUMMARY:\s*\n(.+?)(?=\n\n\w|\Z)",
        text,
        re.DOTALL,
    )
    if not m:
        raise ValueError(f"{summary_path}: no KERNEL_DISPATCH SUMMARY section")
    section = m.group(1)
    rows: list[KernelRow] = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        if _SEPARATOR.match(line):
            continue
        parts = [p.strip() for p in line.strip().split("|")]
        # Expected layout:
        #   parts[0]=""  parts[1]=NAME  parts[2]=DOMAIN  parts[3]=CALLS
        #   parts[4]=DURATION parts[5]=AVERAGE parts[6]=PERCENT
        #   parts[7]=MIN parts[8]=MAX parts[9]=STDDEV parts[10]=""
        if len(parts) < 10:
            continue
        name = parts[1]
        if not name or name.upper() == "NAME":
            continue
        try:
            calls = int(parts[3])
            duration = float(parts[4])
            average = float(parts[5])
            percent = float(parts[6])
            mn = float(parts[7])
            mx = float(parts[8])
        except ValueError:
            # header or non-data row
            continue
        rows.append(
            KernelRow(
                name=name, calls=calls, total_usec=duration,
                avg_usec=average, pct=percent, min_usec=mn, max_usec=mx,
            )
        )
    return rows


def classify(
    kernel_rows: Iterable[KernelRow], trim_warmup: bool = True,
) -> dict[str, FamilyRow]:
    out: dict[str, FamilyRow] = {}
    for row in kernel_rows:
        family = classify_kernel(row.name)
        fr = out.setdefault(family, FamilyRow(family=family))
        fr.calls += row.calls
        fr.total_usec += row.warmup_trim() if trim_warmup else row.total_usec
        if row.name not in fr.kernels:
            fr.kernels.append(row.name)
    return out


def dump_summary(family_rows: dict[str, FamilyRow]) -> str:
    total = sum(fr.total_usec for fr in family_rows.values())
    width = max(len(f) for f in family_rows) + 2
    lines = [
        f"{'family'.ljust(width)} {'calls':>10} {'total_ms':>12} "
        f"{'avg_us':>10} {'pct':>7}  role",
        "-" * (width + 56),
    ]
    for family in sorted(family_rows, key=lambda k: -family_rows[k].total_usec):
        fr = family_rows[family]
        pct = 100.0 * fr.total_usec / total if total else 0.0
        role = (
            "PRESERVE" if family in PRESERVE
            else "DEGRADE-TARGET" if family in DEGRADE_TARGET
            else ""
        )
        lines.append(
            f"{family.ljust(width)} {fr.calls:>10} "
            f"{fr.total_usec / 1000:>12.2f} {fr.avg_usec:>10.2f} "
            f"{pct:>6.2f}%  {role}"
        )
    lines.append("-" * (width + 56))
    lines.append(f"{'TOTAL'.ljust(width)} {'':>10} {total / 1000:>12.2f}")
    return "\n".join(lines)


def to_baseline_json(
    family_rows: dict[str, FamilyRow],
    meta: dict,
) -> dict:
    total = sum(fr.total_usec for fr in family_rows.values())
    families_out: dict[str, dict] = {}
    for family, fr in family_rows.items():
        families_out[family] = {
            "calls": fr.calls,
            "total_usec": round(fr.total_usec, 3),
            "avg_usec": round(fr.avg_usec, 3),
            "pct_of_gpu": round(100.0 * fr.total_usec / total, 3) if total else 0.0,
            "kernels": fr.kernels,
        }
    return {
        "tool": "kernel_family",
        "version": 1,
        **meta,
        "aggregate_gpu_usec": round(total, 3),
        "families": families_out,
        "preserve_set": sorted(PRESERVE),
        "degrade_target_set": sorted(DEGRADE_TARGET),
    }


def parse_bash_time(time_path: Path) -> dict[str, float]:
    """Extract real/user/sys from bash-time output; return seconds."""
    out: dict[str, float] = {}
    for line in Path(time_path).read_text().splitlines():
        m = re.match(r"^(real|user|sys)\s+(\d+)m([\d.]+)s\s*$", line)
        if m:
            out[m.group(1)] = int(m.group(2)) * 60 + float(m.group(3))
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write("usage: kernel_family.py SUMMARY.txt [--json]\n")
        sys.exit(1)
    rows = parse_rocprof_summary(Path(sys.argv[1]))
    fams = classify(rows)
    if "--json" in sys.argv[2:]:
        print(json.dumps(to_baseline_json(fams, meta={}), indent=2))
    else:
        print(dump_summary(fams))
