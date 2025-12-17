import re
import statistics
import argparse


def parse_overhead_stats(file_path):
    """
    Parse a GPU-PMC-Verifier stats log file and compute avg/min/max of relative overhead.
    """

    with open(file_path, "r") as f:
        text = f.read()

    # Regex to capture config name and relative overhead percentage
    pattern = re.compile(
        r"Processing results for: (.*?)\n.*?Relative overhead:\s+([\d.]+)%", re.S
    )

    results = pattern.findall(text)

    overheads = []
    for config, overhead in results:
        value = float(overhead)
        overheads.append((config.strip(), value))

    if not overheads:
        raise ValueError("No overhead data found in log file!")

    values = [v for _, v in overheads]
    avg = statistics.mean(values)
    min_val = min(values)
    max_val = max(values)

    print("📊 Overhead Statistics Across Configurations")
    print("=" * 50)
    for config, val in overheads:
        print(f"{config:60s} -> {val:.2f}%")

    print("\nSummary:")
    print(f"  Average Overhead: {avg:.2f}%")
    print(f"  Minimum Overhead: {min_val:.2f}%")
    print(f"  Maximum Overhead: {max_val:.2f}%")

    return overheads, avg, min_val, max_val


def main():
    """Main function to run the overhead analyzer."""
    parser = argparse.ArgumentParser(
        description="Parse GPU-PMC-Verifier overhead stats and compute statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_overhead.py overhead_summary.txt
  python analyze_overhead.py data/overhead/4090/hw/overhead_summary.txt
        """,
    )

    parser.add_argument(
        "file_path", help="Path to the overhead summary file to analyze"
    )

    args = parser.parse_args()

    try:
        overheads, avg, min_val, max_val = parse_overhead_stats(args.file_path)
        return 0
    except FileNotFoundError:
        print(f"Error: File not found: {args.file_path}")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
