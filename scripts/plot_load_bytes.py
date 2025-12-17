import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re
import argparse


try:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Libertine"],
            "text.latex.preamble": r"""
                \usepackage{libertine}
                \usepackage{newtxmath}
            """,
        }
    )
except Exception as e:
    print(f"LaTeX not available, using system fonts: {e}")
    plt.rcParams.update(
        {"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm"}
    )


def extract_params_from_filename(filename):
    """Extract batch and embd parameters from filename"""
    batch_match = re.search(r"batch[_-]?(\d+)", filename)
    embd_match = re.search(r"embd[_-]?(\d+)", filename)

    batch = int(batch_match.group(1)) if batch_match else None
    embd = int(embd_match.group(1)) if embd_match else None

    return batch, embd


def _normalize_role(role: str) -> str:
    return str(role).strip().lower()


def calculate_theoretical_value(embd, batch, model_type, role=None):
    """Calculate theoretical value (bytes) for the selected role.

    role:
      - 'q'   : query projection only
      - 'qkv' : fused/combined QKV projection (3 projections)
      - None  : keep historical behavior (gpt2 -> qkv, others -> q)
    """
    if role is None:
        if model_type == "gpt2":
            role = "qkv"
        else:  # llama, qwen
            role = "q"

    role_n = _normalize_role(role)
    if role_n == "q":
        factor = 1
    elif role_n == "qkv":
        factor = 3
    else:
        raise ValueError(f"Unsupported role '{role}'. Expected 'q' or 'qkv'.")

    # 4 bytes per fp32 element; (embd*embd + batch*embd) matches original formula.
    return factor * 4 * (embd * embd + batch * embd)


def get_model_type(filename):
    """Extract model type from filename"""
    if filename.startswith("gpt2_"):
        return "gpt2"
    elif filename.startswith("llama_"):
        return "llama"
    elif filename.startswith("qwen_"):
        return "qwen"
    else:
        return "unknown"


def extract_gpu_name(path):
    """Extract GPU name from path if present. Returns None if not found."""
    gpu_names = ["4090", "5080", "h100"]
    path_lower = path.lower()
    for gpu_name in gpu_names:
        if gpu_name.lower() in path_lower:
            return gpu_name.lower()
    return None


def plot_individual_model(results_df, model_name, model_type, gpu_prefix=None):
    """Plot individual model chart and save as PDF"""
    fig, ax1 = plt.subplots(figsize=(12, 5))  # Changed to flatter aspect ratio

    # Set background and grid
    ax1.set_facecolor("white")
    ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Left y-axis: actual and theoretical values (log scale)
    ax1.set_xlabel("n_embd", fontsize=16)
    ax1.set_ylabel("Bytes (Log Scale)", fontsize=16, color="blue")
    ax1.set_yscale("log")

    # Plot measured and theoretical values
    line1 = ax1.plot(
        results_df["embd"],
        results_df["measured"],
        "o-",
        color="blue",
        label="Actual Bytes",
        linewidth=2,
        markersize=6,
        alpha=0.4,
    )
    line2 = ax1.plot(
        results_df["embd"],
        results_df["theoretical"],
        "s--",
        color="green",
        label="Expected Bytes",
        linewidth=2,
        markersize=6,
        alpha=0.4,
    )

    ax1.tick_params(axis="y", labelcolor="blue", labelsize=16)
    ax1.tick_params(axis="x", rotation=45, labelsize=16)

    # Right y-axis: ratio
    ax2 = ax1.twinx()
    ax2.set_ylabel("Ratio", fontsize=16, color="red")
    line3 = ax2.plot(
        results_df["embd"],
        results_df["ratio"],
        "o--",
        color="red",
        label="Ratio (Actual / Expected)",
        linewidth=2,
        markersize=6,
        alpha=0.4,
    )
    ax2.tick_params(axis="y", labelcolor="red", labelsize=16)

    # Set ratio axis range
    ratio_min = min(results_df["ratio"]) * 0.95
    ratio_max = max(results_df["ratio"]) * 1.05
    ax2.set_ylim(ratio_min, ratio_max)

    measured_min = min(results_df["measured"]) * 0.8
    measured_max = (
        max(max(results_df["measured"]), max(results_df["theoretical"])) * 1.2
    )
    ax1.set_ylim(measured_min, measured_max)

    # Add legend above the plot
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        frameon=False,
        fontsize=16,
    )

    # Set x-axis labels
    ax1.set_xticks(results_df["embd"])
    ax1.set_xticklabels(results_df["embd"], rotation=45)

    # Add borders
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("black")

    # Adjust layout
    plt.tight_layout()

    # Save individual chart with GPU prefix if available
    prefix = f"{gpu_prefix}_" if gpu_prefix else ""
    filename = f"figs/load_bytes_observation/{prefix}{model_type}_fp32_global_ld_bytes_diff_n_embd.pdf"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")


def plot_combined_chart(
    model_data, model_types, colors, model_names, role_label="auto", gpu_prefix=None
):
    """Plot combined chart with all models and save as PDF"""
    fig, ax1 = plt.subplots(figsize=(12, 6))  # Changed to flatter aspect ratio

    # Set background and grid
    ax1.set_facecolor("white")
    ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Left y-axis: actual and theoretical values (log scale)
    ax1.set_xlabel("n_embd", fontsize=24)
    ax1.set_ylabel("Bytes (Log Scale)", fontsize=24)
    ax1.set_yscale("log")

    # Per-model markers so lines are easier to distinguish
    markers = {"gpt2": "o", "llama": "D", "qwen": "^"}
    expected_markers = {"gpt2": "s", "llama": "X", "qwen": "P"}

    # Plot all models on combined chart
    for model_type in model_types:
        if not model_data[model_type].empty:
            results_df = model_data[model_type]
            color = colors[model_type]
            name = model_names[model_type]
            marker = markers.get(model_type, "o")
            expected_marker = expected_markers.get(model_type, "s")

            # Actual values (solid lines)
            ax1.plot(
                results_df["embd"],
                results_df["measured"],
                linestyle="-",
                marker=marker,
                color=color,
                label=f"{name} Actual",
                linewidth=2,
                markersize=12,
                alpha=0.6, 
            )

            # Theoretical values (dashed lines)
            ax1.plot(
                results_df["embd"],
                results_df["theoretical"],
                linestyle="--",
                marker=expected_marker,
                color=color,
                label=f"{name} Expected",
                linewidth=2,
                markersize=5,
                alpha=0.8,
            )

    ax1.tick_params(axis="y", labelsize=24)
    ax1.tick_params(axis="x", rotation=45, labelsize=24)

    # Combined chart formatting - legend above the plot
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.5),
        ncol=3,
        frameon=False,
        fontsize=24,
    )
    # ax1.set_title('All Models fp32: Global LD Bytes: Actual vs Expected Comparison',
    #  fontsize=16, pad=30)  # Increased pad for legend space

    # Add borders to combined chart
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("black")

    # Adjust layout
    plt.tight_layout()

    # Save combined chart with GPU prefix if available
    prefix = f"{gpu_prefix}_" if gpu_prefix else ""
    filename = f"figs/load_bytes_observation/{prefix}combined_fp32_global_ld_bytes_diff_n_embd.pdf"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")


def _extract_measured_value(
    df: pd.DataFrame, file_path: str, role: str | None, model_type: str
) -> float:
    metric_col = "global_op_ld_lookup_miss_bytes"
    if metric_col not in df.columns:
        raise ValueError(f"Missing required column '{metric_col}' in {file_path}")

    # Auto role selection preserves old behavior (gpt2 -> qkv, others -> q)
    if role is None or _normalize_role(role) == "auto":
        desired_role = "qkv" if model_type == "gpt2" else "q"
    else:
        desired_role = role

    if "role" not in df.columns:
        # Fallback to old behavior if role column doesn't exist.
        return float(df[metric_col].iloc[0])

    desired_role_n = _normalize_role(desired_role)
    role_series = df["role"].astype(str).str.strip().str.lower()
    df_role = df[role_series == desired_role_n]
    if df_role.empty:
        available = sorted(set(role_series.dropna().unique().tolist()))
        raise ValueError(
            f"No rows found for role='{desired_role}' in {file_path}. "
            f"Available roles: {available}"
        )

    # In some files a role can appear multiple times; summing is more robust than iloc[0].
    return float(df_role[metric_col].sum())


def analyze_memory_misses(
    role: str,
    csv_dir: str,
    plot_mode: str = "both",
):
    # Extract GPU name from path if present
    gpu_prefix = extract_gpu_name(csv_dir)

    # Get all CSV files in the plot folder
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in '{csv_dir}', please check the path")
        return

    # Organize data by model type
    model_data = {"gpt2": [], "llama": [], "qwen": []}

    os.makedirs("figs", exist_ok=True)

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)

            # Extract parameters from filename
            filename = os.path.basename(file_path)
            model_type = get_model_type(filename)

            if model_type == "unknown":
                raise ValueError(f"Unknown model type for file {filename}")

            batch, embd = extract_params_from_filename(filename)

            if batch is None or embd is None:
                raise ValueError(
                    f"Unable to extract batch and embd parameters from filename {filename}"
                )

            # Only keep batch=4
            if batch != 4:
                continue

            # Get measured value for the requested role
            measured_value = _extract_measured_value(
                df, file_path, role=role, model_type=model_type
            )

            # Calculate theoretical value
            theoretical_value = calculate_theoretical_value(
                embd, batch, model_type, role=role if role != "auto" else None
            )

            # Calculate ratio
            ratio = measured_value / theoretical_value if theoretical_value != 0 else 0

            model_data[model_type].append(
                {
                    "embd": embd,
                    "batch": batch,
                    "measured": measured_value,
                    "theoretical": theoretical_value,
                    "ratio": ratio,
                    "filename": filename,
                }
            )

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # Convert to DataFrames and sort by embd
    for model_type in model_data:
        if model_data[model_type]:
            model_data[model_type] = pd.DataFrame(model_data[model_type]).sort_values(
                "embd"
            )

    # Model colors and names
    # A cleaner, more distinct palette (hex) than 'blue/green/orange'
    colors = {"gpt2": "#4E79A7", "llama": "#59A14F", "qwen": "#F28E2B"}
    model_names = {"gpt2": "GPT2", "llama": "LLaMA", "qwen": "Qwen"}
    model_types = ["gpt2", "llama", "qwen"]

    # Generate plots based on plot_mode
    role_label = _normalize_role(role) if role is not None else "auto"
    batch_label = f"batch{4}"

    if plot_mode in ["individual", "both"]:
        # Generate individual charts for each model
        for model_type in model_types:
            if not model_data[model_type].empty:
                plot_individual_model(
                    model_data[model_type],
                    model_names[model_type],
                    model_type,
                    gpu_prefix=gpu_prefix,
                )

    if plot_mode in ["combined", "both"]:
        # Generate combined chart
        plot_combined_chart(
            model_data,
            model_types,
            colors,
            model_names,
            role_label=f"{role_label}_{batch_label}",
            gpu_prefix=gpu_prefix,
        )

    # Print data summary
    print("\nData Summary by Model:")
    for model_type in model_types:
        if not model_data[model_type].empty:
            print(f"\n{model_names[model_type]}:")
            print(
                model_data[model_type][
                    ["embd", "batch", "measured", "theoretical", "ratio"]
                ].to_string(index=False)
            )

    print(f"\nPDF charts saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze global LD lookup miss bytes vs theoretical estimates."
    )
    parser.add_argument(
        "--role",
        default="auto",
        choices=["auto", "q", "qkv"],
        help="Which role to analyze from CSV column 'role'. 'auto' keeps old behavior (gpt2->qkv, others->q).",
    )
    parser.add_argument(
        "--csv_dir",
        required=True,
        help="Directory containing processed CSV files (the script will read all '*.csv' inside).",
    )
    parser.add_argument(
        "--plot_mode",
        default="combined",
        choices=["individual", "combined", "both"],
        help="Plot mode: 'individual' for separate plots per model, 'combined' for one combined plot, 'both' for both (default: both).",
    )

    args = parser.parse_args()
    analyze_memory_misses(
        role=args.role,
        csv_dir=args.csv_dir,
        plot_mode=args.plot_mode,
    )
