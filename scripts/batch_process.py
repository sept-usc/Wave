#!/usr/bin/env python3
"""Batch process all files in preprocessed/ folder through analyzer."""

import subprocess
from pathlib import Path
import shutil
import sys


def main():
    # Get preprocessed directory from command line argument or use default
    if len(sys.argv) > 1:
        preprocessed_dir = Path(sys.argv[1])
    else:
        preprocessed_dir = Path("preprocessed")

    # Derive processed directory from preprocessed directory
    # e.g., data/lower_bound/4090/preprocessed -> data/lower_bound/4090/processed
    processed_dir = preprocessed_dir.parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(preprocessed_dir.glob("*.csv"))

    if not csv_files:
        print(f"❌ No CSV files found in {preprocessed_dir}")
        return

    print(f"Processing {len(csv_files)} files...\n")

    for i, csv_file in enumerate(csv_files, 1):
        filename = csv_file.name

        # Determine family from filename
        if filename.startswith("gpt2"):
            family = "gpt2"
        elif filename.startswith("llama"):
            family = "llama"
        elif filename.startswith("qwen"):
            family = "qwen"
        else:
            print(f"[{i}/{len(csv_files)}] ⚠️  {filename} - Unknown family, skipping")
            continue

        output_file = processed_dir / filename
        temp_output = "process.csv"

        print(f"[{i}/{len(csv_files)}] 📄 {filename}")

        # Run analyzer
        cmd = [
            "python",
            "-m",
            "src.gpu_pmc_analyzer.analyzer",
            "-i",
            str(csv_file),
            "-o",
            temp_output,
            "--family",
            family,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                # Move temp output to final location
                shutil.move(temp_output, output_file)
                print(f"  ✓ Saved to {output_file}")
            else:
                print(f"  ❌ Failed: {result.stderr[:200]}")

        except subprocess.TimeoutExpired:
            print("  ❌ Timeout after 60s")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n✅ Done! Check {processed_dir}/ for results")


if __name__ == "__main__":
    main()
