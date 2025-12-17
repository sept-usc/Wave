"""
Upper-bound pipeline: preprocess -> analyze roles.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run upper-bound preprocessing + analysis")
    ap.add_argument(
        "--csv-dir",
        default="data/upper_bound/csv",
        help="Directory containing raw upper-bound NCU CSVs",
    )
    ap.add_argument(
        "--preprocessed-dir",
        default="data/upper_bound/preprocessed",
        help="Directory to write preprocessed CSVs",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_role.csv outputs",
    )
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    pre_dir = Path(args.preprocessed_dir)

    if not csv_dir.exists():
        raise FileNotFoundError(str(csv_dir))

    pre_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 1: Preprocessing upper_bound CSVs...")
    print("=" * 60)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.gpu_pmc_analyzer.preprocessor",
            "-d",
            str(csv_dir),
            "-o",
            str(pre_dir),
        ],
        check=True,
    )

    print("\n" + "=" * 60)
    print("Step 2: Analyzing roles (matmul/splitKreduce/add)...")
    print("=" * 60)

    inputs = sorted(pre_dir.glob("*_processed.csv"))
    if not inputs:
        print(f"No *_processed.csv found in {pre_dir}")
        return

    script_path = Path(__file__).parent / "analyze_upper_bound.py"

    ok = 0
    skipped = 0
    for in_file in inputs:
        out_file = pre_dir.parent / "processed" / in_file.name
        if out_file.exists() and not args.overwrite:
            skipped += 1
            continue

        subprocess.run(
            [
                sys.executable,
                str(script_path),
                "-i",
                str(in_file),
                "-o",
                str(out_file),
            ],
            check=True,
        )
        ok += 1

    print("\n✅ Upper-bound pipeline complete!")
    print(f"Processed: {ok}, skipped: {skipped}, total inputs: {len(inputs)}")


if __name__ == "__main__":
    main()
