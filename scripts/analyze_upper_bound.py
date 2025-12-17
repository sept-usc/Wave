"""
Analyze upper-bound NCU CSVs.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from utils import logger


def _ensure_preprocessed(df: pd.DataFrame, *, source_path: Path) -> pd.DataFrame:
    """Ensure we have the derived columns used by the heuristics."""
    if "total_flops_with_tensor" in df.columns and "shared_ratio" in df.columns:
        return df.copy().reset_index(drop=True)

    # Treat as raw NCU export (the 2nd row is units; preprocessor drops it via iloc[1:]).
    try:
        from src.gpu_pmc_analyzer.preprocessor import PMCProcessor

        folder = str(source_path.parent)
        fname = source_path.name
        p = PMCProcessor(folder)
        out = p.process_file(fname)
        return out.reset_index(drop=True)
    except Exception as e:
        raise RuntimeError(
            f"Input does not look preprocessed (missing total_flops_with_tensor/shared_ratio), "
            f"and failed to preprocess raw NCU CSV: {source_path} ({e})"
        )


def _add_debug_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Subset of analyzer._add_debug_columns for coarse role detection."""
    working = df.copy().reset_index(drop=True)

    flops = working["total_flops_with_tensor"].astype(float)
    flops_log = flops.map(lambda x: math.log(x + 1.0))
    mu = float(flops_log.mean())
    sigma = float(flops_log.std(ddof=0))
    flops_thr = mu + 0.5 * sigma

    working["flops_log"] = flops_log
    working["flops_threshold"] = flops_thr
    working["is_matmul_candidate"] = flops_log > flops_thr

    # Keep a shared-ratio mean around for later heuristics
    working["shared_ratio_mean"] = float(working["shared_ratio"].astype(float).mean())

    return working


def _isclose_series(
    a: pd.Series, b: pd.Series, *, rtol: float = 0.0, atol: float = 0.0
) -> pd.Series:
    a = a.astype(float)
    b = b.astype(float)
    return (a - b).abs() <= (atol + rtol * b.abs())


def _detect_splitkreduce(working: pd.DataFrame) -> pd.Series:
    """Detect splitKreduce kernels"""
    prev_is_matmul = (
        working["is_matmul_candidate"].astype(bool).shift(1, fill_value=False)
    )

    k_gr = working["global_op_ld_lookup_miss_bytes"].astype(float)
    k_gw = working["global_op_st_lookup_miss_bytes"].astype(float)
    prev_gw = (
        working["global_op_st_lookup_miss_bytes"].astype(float).shift(1, fill_value=0.0)
    )

    # match previous matmul write to current read
    heavy_reduce = k_gr > (2.0 * k_gw)
    link = _isclose_series(k_gr, prev_gw, rtol=0.1, atol=1e-6)
    # split-K matmul writes partials that are much larger than the reduce's write
    ratio_ok = (prev_gw >= 2.0 * k_gw) & (prev_gw <= 32.0 * k_gw) & (k_gw > 0)
    return prev_is_matmul & link & heavy_reduce & ratio_ok


def _detect_add(working: pd.DataFrame) -> pd.Series:
    """Detect add kernels"""
    not_matmul = ~working["is_matmul_candidate"].astype(bool)

    shared_mean = working["shared_ratio"].astype(float).mean()
    shared_low = working["shared_ratio"].astype(float) < float(shared_mean)

    tf = working["total_flops_with_tensor"].astype(float)
    fma = working["total_fma_ops"].astype(float)
    flops_sig = (fma > 0) & _isclose_series(tf, 2.0 * fma, rtol=0.0, atol=1e-6)

    ld = working["global_op_ld_lookup_miss_bytes"].astype(float)
    st = working["global_op_st_lookup_miss_bytes"].astype(float)
    bytes_sig = (st > 0) & _isclose_series(ld, 2.0 * st, rtol=0.0, atol=1e-6)

    return not_matmul & shared_low & flops_sig & bytes_sig


def assign_upper_bound_roles(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a new coarse `role` column."""
    working = _add_debug_columns(df)

    # Order required by user: matmul -> splitKreduce -> add
    is_matmul = working["is_matmul_candidate"].astype(bool)
    is_splitkreduce = _detect_splitkreduce(working)
    is_add = _detect_add(working)

    role = pd.Series(["other"] * len(working), index=working.index, dtype=object)
    role.loc[is_matmul] = "matmul"
    role.loc[(role == "other") & is_splitkreduce] = "splitKreduce"
    role.loc[(role == "other") & is_add] = "add"

    out = working.copy()
    out["role"] = role.values

    return out


def _summarize(df: pd.DataFrame) -> None:
    counts = df["role"].value_counts(dropna=False).to_dict()
    logger.info(f"Upper-bound role counts: {counts}")

    # Show top kernels per class (by total_flops_with_tensor)
    if "total_flops_with_tensor" in df.columns:
        for r in ["matmul", "splitKreduce", "add"]:
            sub = df[df["role"] == r]
            if sub.empty:
                continue
            top = sub.sort_values("total_flops_with_tensor", ascending=False).head(5)
            names = top[["Kernel Name", "total_flops_with_tensor"]].to_dict("records")
            logger.info(f"Top {r} kernels: {names}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze upper-bound CSV and detect matmul/add/splitKreduce"
    )
    ap.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to upper_bound CSV (raw or preprocessed)",
    )
    ap.add_argument("-o", "--output", default=None, help="Optional output CSV path")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    df_in = pd.read_csv(in_path, low_memory=False)
    df = _ensure_preprocessed(df_in, source_path=in_path)

    out = assign_upper_bound_roles(df)
    _summarize(out)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        logger.success(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
