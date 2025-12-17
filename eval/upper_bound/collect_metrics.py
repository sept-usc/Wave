#!/usr/bin/env python3
import os
import argparse
import subprocess
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import sys
from utils import logger


class MetricsCollector:
    def __init__(self, executable_path: str):
        self.executable_path = executable_path
        self.python_executable = sys.executable

        self.metrics = [
            # L1 cache metrics
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss.sum",
            # Double operations
            "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
            # Float operations
            "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
            # Half precision operations
            "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum",
            # Tensor core operations
            "sm__ops_path_tensor_src_fp8.sum",  # 4090 metric
            "sm__ops_path_tensor_src_fp8_dst_fp16.sum",  # 5080 metric
            "sm__ops_path_tensor_src_fp8_dst_fp32.sum",  # 5080 metric
            "sm__ops_path_tensor_src_bf16_dst_fp32.sum",
            "sm__ops_path_tensor_src_fp16_dst_fp16.sum",
            "sm__ops_path_tensor_src_fp16_dst_fp32.sum",
            "sm__ops_path_tensor_src_tf32_dst_fp32.sum",
            "sm__ops_path_tensor_src_fp64.sum",
            # Memory operations
            "smsp__sass_inst_executed_op_global_ld.sum",
            "smsp__sass_inst_executed_op_global_st.sum",
            "smsp__sass_inst_executed_op_local_ld.sum",
            "smsp__sass_inst_executed_op_local_st.sum",
            "smsp__sass_inst_executed_op_shared_ld.sum",
            "smsp__sass_inst_executed_op_shared_st.sum",
            "sm__sass_inst_executed_op_shared_ld.sum",
        ]

    def _parse_kernel_configs(self, output: str) -> List[Dict[str, Any]]:
        """
        Parse kernel configurations from the executable output.

        Args:
            output: The stdout output from the executable

        Returns:
            List of kernel configurations, each containing:
            {
                "layer_name": str,  # Name of the layer
                "split": int,  # Split number
                "split_type": str,  # Type of split
                "shape": {
                    "b": int,  # Batch dimension
                    "m": int,  # Matrix dimension m
                    "k": int,  # Matrix dimension k
                    "n": int   # Matrix dimension n
                }
            }
        """
        kernel_configs = []
        current_layer = None
        current_split = None
        current_split_type = None

        for line in output.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Parse layer line with split info
            if ":" in line and "split=" in line and "type=" in line:
                # Example: "  q_proj: split=2 type=hstack with 2 kernels:"
                parts = line.split(":")
                layer_name = parts[0].strip()
                split_info = parts[1].strip()
                current_layer = layer_name
                current_split = int(split_info.split("split=")[1].split()[0])
                current_split_type = split_info.split("type=")[1].split()[0]

            # Parse kernel shape line
            elif "KernelShape" in line:
                # Example: "    - KernelShape(b=4, m=1, k=1024, n=256)"
                shape_str = line.split("KernelShape(")[1].rstrip(")")
                shape_parts = shape_str.split(", ")
                shape = {}
                for part in shape_parts:
                    dim, value = part.split("=")
                    shape[dim] = int(value)

                kernel_configs.append(
                    {
                        "layer_name": current_layer,
                        "split": current_split,
                        "split_type": current_split_type,
                        "shape": shape,
                    }
                )

        return kernel_configs

    def _export_to_csv(self, raw_file: str, csv_file: str) -> None:
        """
        Export a single NCU report to CSV format.

        Args:
            raw_file: Path to the raw NCU report (.ncu-rep)
            csv_file: Path to save the CSV output
        """
        try:
            with open(csv_file, "w") as f:
                subprocess.run(
                    ["ncu", "--csv", "--page", "raw", "-i", raw_file],
                    stdout=f,
                    check=True,
                )
            logger.success(f"Successfully exported {csv_file}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to export {raw_file} to CSV: {e}")

    def export_all_to_csv(self, raw_data_dir: str, csv_data_dir: str) -> None:
        """
        Export all raw NCU reports to CSV format.

        Args:
            raw_data_dir: Directory containing raw NCU reports
            csv_data_dir: Directory to save CSV exports
        """
        # Create directories if they don't exist
        os.makedirs(raw_data_dir, exist_ok=True)
        os.makedirs(csv_data_dir, exist_ok=True)

        # Walk through all subdirectories in the raw data directory
        for root, dirs, files in os.walk(raw_data_dir):
            for file in files:
                if file.endswith(".ncu-rep"):
                    raw_report_path = os.path.join(root, file)
                    csv_export_path = os.path.join(
                        csv_data_dir, file.replace(".ncu-rep", ".csv")
                    )

                    # Skip if the CSV file already exists
                    if os.path.exists(csv_export_path):
                        logger.info(f"Skipping {csv_export_path}, already exported")
                        continue

                    logger.info(f"Exporting {raw_report_path} to {csv_export_path}")
                    self._export_to_csv(raw_report_path, csv_export_path)

        logger.success("All reports successfully exported to CSV!")

    def collect_metrics(
        self,
        b: int,
        s: int,
        d: int,
        d_ffn: int,
        model_size: int,
        output_file: Optional[str] = None,
        additional_args: Optional[List[str]] = None,
        granularity: int = 256,
    ) -> None:
        """
        Collect metrics using NCU for the given parameters.

        Args:
            b: Batch size
            s: Sequence length
            d: Hidden dimension
            d_ffn: Feed-forward network dimension
            model_size: Maximum model size
            output_file: Path to save the raw metrics data (required)
            additional_args: Optional list of additional arguments to pass to the executable
            granularity: Minimum block size for matrix operations (default: 256)
        """
        # Create output directories if needed
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            # Store configs in a separate directory
            config_dir = os.path.join(
                os.path.dirname(os.path.dirname(output_file)), "configs"
            )
            os.makedirs(config_dir, exist_ok=True)
            config_file = os.path.join(
                config_dir,
                os.path.basename(output_file).replace(".ncu-rep", "_config.json"),
            )

        # First run the executable to get kernel configurations
        cmd = [
            self.python_executable,
            self.executable_path,
            "--batch-size",
            str(b),
            "--seq-len",
            str(s),
            "--hidden-dim",
            str(d),
            "--ffn-dim",
            str(d_ffn),
            "--granularity",
            str(granularity),
        ]

        if additional_args:
            cmd.extend(additional_args)

        try:
            # Now run NCU to collect metrics
            ncu_cmd = [
                "ncu",
                "--config-file",
                "off",
                "--export",
                output_file if output_file else "metrics.ncu-rep",
                "--force-overwrite",
                "--replay-mode",
                "kernel",
                "--metrics",
                ",".join(self.metrics),
                self.python_executable,  # Use Python from venv
                self.executable_path,  # The Python script path
                "--batch-size",
                str(b),
                "--seq-len",
                str(s),
                "--hidden-dim",
                str(d),
                "--ffn-dim",
                str(d_ffn),
                "--granularity",
                str(granularity),
            ]

            if additional_args:
                ncu_cmd.extend(additional_args)

            # Run the executable and capture output
            result = subprocess.run(ncu_cmd, capture_output=True, text=True, check=True)
            print(result.stdout)

            # Parse kernel configurations
            kernel_configs = self._parse_kernel_configs(result.stdout)

            if output_file:
                # Save kernel configurations alongside the metrics
                with open(config_file, "w") as f:
                    json.dump(
                        {
                            "parameters": {
                                "b": b,
                                "s": s,
                                "d": d,
                                "d_ffn": d_ffn,
                                "model_size": model_size,
                                "granularity": granularity,
                                "additional_args": additional_args,
                            },
                            "kernel_configs": kernel_configs,
                        },
                        f,
                        indent=2,
                    )

                logger.success(f"Raw metrics data saved to {output_file}")
                logger.success(f"Kernel configurations saved to {config_file}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running executable or NCU: {e}")
            if hasattr(e, "stderr"):
                logger.error(f"Error output: {e.stderr}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

    def collect_metrics_batch(
        self,
        batch_sizes: list,
        seq_lens: list,
        hidden_dims: list,
        ffn_dims: list,
        raw_data_dir: str,
        csv_data_dir: str,
        additional_args: list = None,
        granularity: int = 256,
    ):
        """Collect metrics for multiple configurations.

        Args:
            batch_sizes: List of batch sizes to test
            seq_lens: List of sequence lengths to test
            hidden_dims: List of hidden dimensions to test
            ffn_dims: List of FFN dimensions to test
            raw_data_dir: Directory to save raw NCU reports
            csv_data_dir: Directory to save CSV exports
            additional_args: Optional list of additional arguments to pass to the executable
            granularity: Minimum block size for matrix operations (default: 256)
        """
        # Create directories
        os.makedirs(raw_data_dir, exist_ok=True)
        os.makedirs(csv_data_dir, exist_ok=True)
        config_dir = os.path.join(os.path.dirname(raw_data_dir), "configs")
        os.makedirs(config_dir, exist_ok=True)

        # Collect metrics for each configuration
        for b in batch_sizes:
            for s in seq_lens:
                for d in hidden_dims:
                    for f in ffn_dims:
                        # Create timestamp for unique filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        config_name = f"b{b}_s{s}_d{d}_f{f}_{timestamp}"

                        # Set up paths
                        raw_file = os.path.join(raw_data_dir, f"{config_name}.ncu-rep")

                        logger.info(
                            f"Collecting metrics for b={b}, s={s}, d={d}, f={f}"
                        )

                        try:
                            # Collect metrics
                            self.collect_metrics(
                                b=b,
                                s=s,
                                d=d,
                                d_ffn=f,
                                model_size=25165824,  # Default value
                                output_file=raw_file,
                                additional_args=additional_args,
                                granularity=granularity,
                            )

                            logger.success(
                                f"Successfully collected metrics for {config_name}"
                            )

                        except Exception as e:
                            logger.error(
                                f"Failed to collect metrics for {config_name}: {str(e)}"
                            )
                            continue

        # Export all collected data to CSV
        logger.info("Exporting all collected data to CSV...")
        self.export_all_to_csv(raw_data_dir, csv_data_dir)


def main():
    parser = argparse.ArgumentParser(description="Collect GPU metrics using NCU")
    parser.add_argument(
        "--executable", "-e", required=True, help="Path to the executable to profile"
    )
    parser.add_argument("--batch-size", "-b", type=int, help="Batch size (single run)")
    parser.add_argument(
        "--seq-len", "-s", type=int, help="Sequence length (single run)"
    )
    parser.add_argument(
        "--hidden-dim", "-d", type=int, help="Hidden dimension (single run)"
    )
    parser.add_argument(
        "--ffn-dim",
        "-f",
        type=int,
        help="Feed-forward network dimension (single run)",
    )
    parser.add_argument(
        "--granularity",
        "-g",
        type=int,
        default=256,
        help="Minimum block size for matrix operations (default: 256)",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        help="List of batch sizes (batch run)",
    )
    parser.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        help="List of sequence lengths (batch run)",
    )
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        help="List of hidden dimensions (batch run)",
    )
    parser.add_argument(
        "--ffn-dims", nargs="+", type=int, help="List of FFN dimensions (batch run)"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/upper_bound/raw",
        help="Directory for raw NCU reports",
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default="data/upper_bound/csv",
        help="Directory for CSV exports",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="data/upper_bound/configs",
        help="Directory for kernel configurations",
    )
    parser.add_argument(
        "--additional-args",
        "-a",
        nargs="*",
        help="Additional arguments to pass to the executable",
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Enable test mode for correctness verification",
    )

    args = parser.parse_args()

    # Create metrics collector
    collector = MetricsCollector(args.executable)

    # Prepare additional arguments
    additional_args = args.additional_args or []
    if args.test:
        additional_args.append("--test")

    # Check if this is a single run or batch run
    if (
        args.batch_size is not None
        and args.seq_len is not None
        and args.hidden_dim is not None
        and args.ffn_dim is not None
    ):
        # Single run
        collector.collect_metrics(
            b=args.batch_size,
            s=args.seq_len,
            d=args.hidden_dim,
            d_ffn=args.ffn_dim,
            model_size=25165824,  # Default value
            output_file=os.path.join(
                args.raw_dir,
                f"b{args.batch_size}_s{args.seq_len}_d{args.hidden_dim}_f{args.ffn_dim}.ncu-rep",
            ),
            additional_args=additional_args,
            granularity=args.granularity,
        )
    elif args.batch_sizes and args.seq_lens and args.hidden_dims and args.ffn_dims:
        # Batch run
        collector.collect_metrics_batch(
            batch_sizes=args.batch_sizes,
            seq_lens=args.seq_lens,
            hidden_dims=args.hidden_dims,
            ffn_dims=args.ffn_dims,
            raw_data_dir=args.raw_dir,
            csv_data_dir=args.csv_dir,
            additional_args=additional_args,
            granularity=args.granularity,
        )
    else:
        logger.error(
            "Please provide either single run parameters (-b, -s, -d, -f) or batch run parameters (--batch-sizes, --seq-lens, --hidden-dims, --ffn-dims)"
        )
        return


if __name__ == "__main__":
    main()
