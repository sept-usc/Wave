#!/usr/bin/env python3
"""Simple pipeline: preprocess -> batch_process -> test_role_order"""

import subprocess
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <data_folder_path>")
        sys.exit(1)

    data_path = Path(sys.argv[1])

    # Derive output path from data path
    # e.g., data/lower_bound/4090/csv/ -> data/lower_bound/4090/preprocessed
    output_path = data_path.parent / "preprocessed"

    processed_path = output_path.parent / "processed"

    print("=" * 60)
    print("Step 1: Preprocessing...")
    print("=" * 60)
    subprocess.run(
        [
            "python",
            "-m",
            "src.gpu_pmc_analyzer.preprocessor",
            "-d",
            str(data_path),
            "-o",
            str(output_path),
        ],
        check=True,
    )

    print("\n" + "=" * 60)
    print("Step 2: Batch processing...")
    print("=" * 60)
    # Get the scripts directory path
    scripts_dir = Path(__file__).parent
    subprocess.run(
        ["python", str(scripts_dir / "batch_process.py"), str(output_path)], check=True
    )

    print("\n" + "=" * 60)
    print("Step 3: Testing role order...")
    print("=" * 60)
    subprocess.run(
        [
            "python",
            str(scripts_dir / "test_role_order.py"),
            "--dir",
            str(processed_path),
        ],
        check=True,
    )

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()
