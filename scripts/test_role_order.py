import pandas as pd
from pathlib import Path


def test_role_order(
    csv_path: str, expected_order: list[str], run_analyzer: bool = False
) -> bool:
    """Test if roles in CSV file follow expected order."""
    df = pd.read_csv(csv_path)

    if "role" not in df.columns:
        if run_analyzer:
            print("  ⏭️  No role column, running analyzer...")
            # Run analyzer to generate roles
            from src.gpu_pmc_analyzer.analyzer import Analyzer, AnalyzerConfig

            # Determine family from filename
            filename = Path(csv_path).name
            if filename.startswith("gpt2"):
                family = "gpt2"
            elif filename.startswith("llama"):
                family = "llama"
            elif filename.startswith("qwen"):
                family = "qwen"
            else:
                print("  ❌ Unknown family")
                return False

            cfg = AnalyzerConfig.for_family(family)
            analyzer = Analyzer(df_path=csv_path, config=cfg)

            # Simple single-layer analysis
            df = analyzer.data.head(100)  # Just analyze first 100 rows
            df = analyzer.assign_roles(df)
        else:
            print("  ❌ Missing 'role' column (use --analyze to auto-generate)")
            return False

    # Extract roles in order, skipping "other"
    roles = [r for r in df["role"].values if r != "other"]

    # Check all expected roles are present
    missing = set(expected_order) - set(roles)
    if missing:
        print(f"  ❌ Missing roles: {missing}")
        print(f"     Found: {roles}")
        return False

    # Check order by finding indices
    indices = []
    for role in expected_order:
        if role in roles:
            indices.append(roles.index(role))

    if indices != sorted(indices):
        print("  ❌ Wrong order!")
        print(f"     Expected: {expected_order}")
        print(f"     Found:    {roles}")
        return False

    print(f"  ✓ Correct: {roles}")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test role order in processed CSV files"
    )
    parser.add_argument(
        "--dir", default="processed", help="Directory to test (default: processed)"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Auto-run analyzer on files without role column",
    )
    args = parser.parse_args()

    test_dir = Path(args.dir)

    if not test_dir.exists():
        print(f"❌ Directory not found: {test_dir}")
        return

    # Define expected role orders
    gpt2_order = ["QKV", "Attention", "Wo", "elementwise_add", "FFN1", "FFN2"]
    llama_qwen_order = [
        "Q",
        "K",
        "V",
        "Attention",
        "Wo",
        "elementwise_add",
        "FFN1",
        "FFN2",
        "FFN3",
    ]

    csv_files = sorted(test_dir.glob("*.csv"))

    if not csv_files:
        print(f"❌ No CSV files found in {test_dir}")
        return

    print(f"Testing {len(csv_files)} files in {test_dir}/\n")

    passed = 0
    failed = 0

    for csv_file in csv_files:
        filename = csv_file.name
        print(f"📄 {filename}")

        if filename.startswith("gpt2"):
            success = test_role_order(
                str(csv_file), gpt2_order, run_analyzer=args.analyze
            )
        elif filename.startswith("llama") or filename.startswith("qwen"):
            success = test_role_order(
                str(csv_file), llama_qwen_order, run_analyzer=args.analyze
            )
        else:
            print("  ⚠️  Unknown prefix, skipping")
            continue

        if success:
            passed += 1
        else:
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ {failed} test(s) failed")


if __name__ == "__main__":
    main()
