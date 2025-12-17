#!/usr/bin/env python3
"""
PMC Processor for analyzing kernel FLOPs by roles from NVIDIA Nsight Compute data.
This script processes PMC data to calculate FLOPs for MatMuls
and creates plots showing FLOPs vs hidden dimension for each model type.
"""

import pandas as pd
import numpy as np
import re
import argparse
import os
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional


class RoleBasedPMCProcessor:
    """Processor for PMC data to calculate kernel FLOPs by specific roles."""

    def __init__(self, input_file: str):
        """
        Initialize the PMC processor.

        Args:
            input_file: Path to the PMC CSV file
        """
        self.input_file = input_file
        self.df = None

    def intify(self, value):
        if isinstance(value, str):
            value = value.replace(",", "")
            if re.match(r"^-?\d+$", value):
                return int(value)
        return value

    def load_data(self) -> pd.DataFrame:
        """
        Load PMC data from CSV file.

        Returns:
            Loaded DataFrame
        """
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        print(f"Loading PMC data from: {self.input_file}")
        # Skip the second row (units row) when reading the CSV
        self.df = pd.read_csv(self.input_file)
        print(f"Loaded {len(self.df)} kernel records (skipped units row)")

        # Convert string values to integers using modern pandas methods
        for col in self.df.columns:
            if self.df[col].dtype == "object":  # Only process string columns
                self.df[col] = self.df[col].apply(self.intify)

        return self.df

    def get_role_based_flops(self) -> Dict[str, int]:
        """
        Get FLOPs for specific roles (q, k, v, FFN1, FFN2, FFN3).

        Returns:
            Dictionary with role names as keys and total FLOPs as values
        """
        # Initialize role FLOPs
        role_flops = {"Q": 0, "K": 0, "V": 0, "FFN1": 0, "FFN2": 0, "FFN3": 0}

        # Use the existing role column to categorize kernels
        for _, row in self.df.iterrows():
            role = str(row.get("role", "")).strip()
            flops = row["total_flops_with_tensor"]

            # Map role values to our role categories
            if role in role_flops:
                role_flops[role] += flops

        return role_flops


def parse_filename(filename: str) -> Optional[Dict]:
    """
    Parse filename to extract model parameters.

    Args:
        filename: Filename to parse

    Returns:
        Dictionary with parsed parameters or None if parsing fails
    """
    # Remove file extension
    name = Path(filename).stem

    # Pattern: model_layerN_embdN_ffnN_headN_posN_maxN_batchN_promptN
    pattern = r"(\w+)_layer(\d+)_embd(\d+)_ffn(\d+)_head(\d+)_pos(\d+)_max(\d+)_batch(\d+)_prompt(\d+)"

    match = re.match(pattern, name)
    if match:
        return {
            "model_type": match.group(1),
            "n_layer": int(match.group(2)),
            "hidden_dim": int(match.group(3)),
            "ffn_dim": int(match.group(4)),
            "n_head": int(match.group(5)),
            "n_positions": int(match.group(6)),
            "max_tokens": int(match.group(7)),
            "batch_size": int(match.group(8)),
            "prompt_len": int(match.group(9)),
        }
    return None


def process_folder(folder_path: str) -> List[Dict]:
    """
    Process all CSV files in a folder and extract role-based FLOPs data.

    Args:
        folder_path: Path to folder containing CSV files

    Returns:
        List of dictionaries with file info and role-based FLOPs data
    """
    folder = Path(folder_path)
    csv_files = list(folder.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return []

    results = []

    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file.name}")

        try:
            # Parse filename
            params = parse_filename(csv_file.name)
            if params is None:
                print(f"  Skipping {csv_file.name} - could not parse filename")
                continue

            # Process file
            processor = RoleBasedPMCProcessor(str(csv_file))
            processor.load_data()
            role_flops = processor.get_role_based_flops()

            result = {"filename": csv_file.name, **role_flops, **params}

            results.append(result)
            print(f"  Successfully processed {csv_file.name}")

        except Exception as e:
            print(f"  Error processing {csv_file.name}: {e}")
            continue

    return results


def create_plots(
    results: List[Dict],
    processed_folder: str,
    csv_folder: str,
    output_dir: str = "figs",
) -> None:
    """
    Create plots showing FLOPs vs hidden dimension and FLOPs vs layers for llama models.

    Args:
        results: List of dictionaries with FLOPs data from processed folder
        processed_folder: Path to processed folder for hidden dim plot
        csv_folder: Path to csv folder for layer plot
        output_dir: Directory to save plots
    """
    if not results:
        print("No results to plot")
        return

    # Set matplotlib defaults
    plt.rcParams.update(
        {
            "font.size": 28,
            "axes.titlesize": 32,
            "axes.labelsize": 28,
            "xtick.labelsize": 26,
            "ytick.labelsize": 26,
            "legend.fontsize": 20,
            "font.family": "serif",
            "font.serif": ["Linux Libertine", "Libertine", "DejaVu Serif"],
        }
    )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_path = output_path / "motivating_example"
    output_path.mkdir(exist_ok=True)

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results)

    # Print summary
    print("\nSummary:")
    print(f"Total files processed: {len(results)}")
    print(f"Llama files: {len(df)}")
    print(f"Unique layers: {sorted(df['n_layer'].unique())}")
    print(f"Unique hidden dims: {sorted(df['hidden_dim'].unique())}")

    # Plot 1: FLOPs vs Hidden Dimension (layer=6) - using processed folder data
    plot_flops_vs_hidden_dim(df, output_path)

    # Plot 2: FLOPs vs Layer (hidden_dim=1024) - using csv folder data
    plot_flops_vs_layer_from_csv(csv_folder, output_path)


def plot_flops_vs_hidden_dim(df: pd.DataFrame, output_path: Path) -> None:
    """Plot FLOPs vs hidden dimension for layer=6, showing q, k, v, FFN1, FFN2, and FFN3."""
    # Filter for layer 6
    layer6_data = df[df["n_layer"] == 6].copy()

    if layer6_data.empty:
        print("No data found for layer 6")
        return

    # Sort by hidden dimension
    layer6_data = layer6_data.sort_values("hidden_dim")

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot q and FFN1 only
    roles_to_plot = ["Q", "FFN1"]
    colors = ["#1f77b4", "#ff7f0e"]
    markers = ["o", "s"]

    label_map = {"Q": "Observed Q", "FFN1": "Observed FFN1"}
    for i, role in enumerate(roles_to_plot):
        if role in layer6_data.columns:
            y_values = layer6_data[role].values
            x_values = layer6_data["hidden_dim"].values

            # Convert to millions of FLOPs for better readability
            y_values_millions = y_values / 1e6

            ax.plot(
                x_values,
                y_values_millions,
                color=colors[i],
                marker=markers[i],
                linewidth=3,
                markersize=10,
                label=label_map.get(role, role),
            )

    # Add reference curves with square relation (d²) using mathematical formulas
    if len(layer6_data) > 0:
        # Generate reference curves: y = 2 * d² and y = 8 * d²
        ref_x = np.array(layer6_data["hidden_dim"].values)

        # Reference curve: y = 2 * d²
        ref_2d2_y = 2 * (ref_x**2) / 1e6

        ax.plot(
            ref_x,
            ref_2d2_y,
            "b--",
            linewidth=2,
            alpha=0.7,
            label="2d^2 (theoretical Q FLOPs)",
        )

        # Reference curve: y = 8 * d²
        ref_8d2_y = 8 * (ref_x**2) / 1e6

        ax.plot(
            ref_x,
            ref_8d2_y,
            "orange",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="8d^2 (theoretical FFN FLOPs)",
        )

    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("FLOPs (Millions)")
    ax.set_title("FLOPs vs Hidden Dimension")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Format x-axis with ticks at intervals of 512
    min_dim = layer6_data["hidden_dim"].min()
    max_dim = layer6_data["hidden_dim"].max()
    tick_values = np.arange(min_dim, max_dim + 512, 512)
    ax.set_xticks(tick_values)
    ax.set_xticklabels([512, 1024, 1536, 2048, 2560, 3072, 3584, 4096])

    # Save the plot
    output_file = output_path / "motivating_example_flops_vs_hidden_dim_layer6.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_file}")

    plt.close()


def intify(value):
    if isinstance(value, str):
        value = value.replace(",", "")
        if re.match(r"^-?\d+$", value):
            return int(value)
    return value


def plot_flops_vs_layer_from_csv(csv_folder: str, output_path: Path) -> None:
    """Plot FLOPs vs layer for hidden_dim=1024, calculating FLOPs manually from CSV files."""
    csv_path = Path(csv_folder)
    if not csv_path.exists():
        print(f"CSV folder not found: {csv_folder}")
        return

    # Find all CSV files with hidden_dim=1024
    csv_files = list(csv_path.glob("*embd1024*.csv"))
    if not csv_files:
        print("No CSV files found with hidden_dim=1024")
        return

    # Parse filenames and collect data
    layer_data = []
    for csv_file in csv_files:
        params = parse_filename(csv_file.name)
        if params and params["hidden_dim"] == 1024:
            try:
                # Load CSV and calculate total FLOPs
                df = pd.read_csv(csv_file, skiprows=[1])
                df = df.map(intify)
                total_flops = calculate_total_flops(df)
                layer_data.append(
                    {"n_layer": params["n_layer"], "total_flops": total_flops}
                )
            except Exception as e:
                print(f"Error processing {csv_file.name}: {e}")
                continue

    if not layer_data:
        print("No valid data found for plotting")
        return

    # Convert to DataFrame and sort by layer
    layer_df = pd.DataFrame(layer_data)
    layer_df = layer_df.sort_values("n_layer")

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    x_values = layer_df["n_layer"].values
    y_values_millions = layer_df["total_flops"].values / 1e6

    ax.plot(
        x_values,
        y_values_millions,
        color="#1f77b4",
        marker="o",
        linewidth=3,
        markersize=10,
        label="Total FLOPs",
    )

    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("FLOPs (Millions)")
    ax.set_title("Total FLOPs vs Number of Layers")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Format x-axis to show layers clearly
    ax.set_xticks(x_values)

    # Save the plot
    output_file = output_path / "motivating_example_flops_vs_layers_hidden1024.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_file}")

    plt.close()


def calculate_total_flops(df: pd.DataFrame) -> int:
    """Calculate total FLOPs from PMC data"""
    # Calculate basic floating-point operations
    # Float operations (FP32)
    float_flops = (
        df["smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"].fillna(0)
        + df["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"].fillna(0)
        + df["smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"].fillna(0) * 2
    )

    # Double operations (FP64)
    double_flops = (
        df["smsp__sass_thread_inst_executed_op_dmul_pred_on.sum"].fillna(0)
        + df["smsp__sass_thread_inst_executed_op_dadd_pred_on.sum"].fillna(0)
        + df["smsp__sass_thread_inst_executed_op_dfma_pred_on.sum"].fillna(0) * 2
    )

    # Half operations (FP16)
    half_flops = (
        df["smsp__sass_thread_inst_executed_op_hmul_pred_on.sum"].fillna(0)
        + df["smsp__sass_thread_inst_executed_op_hadd_pred_on.sum"].fillna(0)
        + df["smsp__sass_thread_inst_executed_op_hfma_pred_on.sum"].fillna(0) * 2
    )

    # Calculate Tensor Core operations
    # Each warp-level MMA instruction performs matrix operations
    # For most precisions: 16×8×8 = 1,024 FMAs = 2,048 FLOPs per instruction
    # Using 2,048 FLOPs per warp-level MMA instruction for most precisions

    # Use safe access with .get() to handle missing columns
    tensor_fp8_ops = (
        df.get("sm__ops_path_tensor_src_fp8.sum", pd.Series([0] * len(df))) * 2048
    )
    tensor_fp16_fp16_ops = (
        df.get("sm__ops_path_tensor_src_fp16_dst_fp16.sum", pd.Series([0] * len(df)))
        * 2048
    )
    tensor_fp16_fp32_ops = (
        df.get("sm__ops_path_tensor_src_fp16_dst_fp32.sum", pd.Series([0] * len(df)))
        * 2048
    )
    tensor_tf32_fp32_ops = (
        df.get("sm__ops_path_tensor_src_tf32_dst_fp32.sum", pd.Series([0] * len(df)))
        * 2048
    )

    # FP64 might use smaller matrices, but using 2048 as conservative estimate
    tensor_fp64_ops = (
        df.get("sm__ops_path_tensor_src_fp64.sum", pd.Series([0] * len(df))) * 2048
    )
    tensor_flops = (
        tensor_fp8_ops
        + tensor_fp16_fp16_ops
        + tensor_fp16_fp32_ops
        + tensor_tf32_fp32_ops
        + tensor_fp64_ops
    )

    # Calculate total FLOPs
    total_flops = (
        float_flops.sum() + double_flops.sum() + half_flops.sum() + tensor_flops.sum()
    )

    return int(total_flops)


def main():
    """Main function to run the role-based PMC processor."""
    parser = argparse.ArgumentParser(
        description="Process PMC data to calculate role-based kernel FLOPs and create plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_motivating_example.py /path/to/data/folder
  python analyze_motivating_example.py /path/to/data/folder --output-dir plots
        """,
    )

    parser.add_argument(
        "data_folder", help="Path to data folder (e.g., data/motivating_example/5080)"
    )

    parser.add_argument(
        "--output-dir",
        default="figs",
        help="Output directory for plots (default: figs)",
    )

    args = parser.parse_args()

    # Construct paths for processed and csv folders
    processed_folder = os.path.join(args.data_folder, "processed")
    csv_folder = os.path.join(args.data_folder, "csv")

    # Process the processed folder for role-based analysis
    results = process_folder(processed_folder)

    if results:
        # Create plots using both folders
        create_plots(results, processed_folder, csv_folder, args.output_dir)

        # Print detailed results
        print("\nDetailed Results:")
        df = pd.DataFrame(results)
        # Filter for llama models only
        df = df[df["model_type"] == "llama"]
        sorted_data = df.sort_values(["n_layer", "hidden_dim"])

        for _, row in sorted_data.iterrows():
            print(
                f"\n{row['model_type']} - Layer {row['n_layer']} - Batch {row['batch_size']} - Hidden Dim {row['hidden_dim']}"
            )
            for role in ["q", "FFN1"]:
                if role in row:
                    print(f"  {role.upper()}: {row[role]:,} FLOPs")

        # Also show summary for CSV-based layer analysis
        print("\nCSV-based Layer Analysis Summary:")
        print(f"CSV folder: {csv_folder}")
        print(f"Processed folder: {processed_folder}")
    else:
        print("No results to process")


if __name__ == "__main__":
    main()
