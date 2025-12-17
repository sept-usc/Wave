import argparse
import os
import subprocess
import sys
from utils import logger


def get_raw_data_name(
    model_type: str,
    n_layer: int,
    hidden_dim: int,
    ffn_dim: int,
    n_head: int,
    n_positions: int,
    max_tokens: int,
    batch_size: int,
    prompt_len: int,
) -> str:
    name = f"{model_type}_layer{n_layer}_embd{hidden_dim}_ffn{ffn_dim}_head{n_head}_pos{n_positions}"
    return name + f"_max{max_tokens}_batch{batch_size}_prompt{prompt_len}.ncu-rep"


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
        ]

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
        model_type: str,
        n_layer: int,
        hidden_dim: int,
        ffn_dim: int,
        n_head: int,
        n_positions: int,
        output_file: str,
        max_tokens: int = 2,
        batch_size: int = 4,
        prompt_len: int = 1,
    ) -> None:
        """
        Collect metrics using NCU for the given parameters.

        Args:
            model_type: Model type (gpt2, llama, qwen)
            n_layer: Number of layers
            hidden_dim: Hidden dimension
            ffn_dim: Feed-forward dimension
            n_head: Number of attention heads
            n_positions: Maximum sequence length
            output_file: Path to save the raw metrics data (required)
            max_tokens: Maximum number of tokens to generate
            batch_size: Batch size
            prompt_len: Prompt length
        """
        # Create output directories if needed
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

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
                "application",
                "--app-replay-mode",
                "relaxed",
                "--metrics",
                ",".join(self.metrics),
                self.python_executable,  # Use Python from venv
                self.executable_path,  # The Python script path
                "--model-type",
                model_type,
                "--n_layer",
                str(n_layer),
                "--hidden-dim",
                str(hidden_dim),
                "--ffn-dim",
                str(ffn_dim),
                "--n_head",
                str(n_head),
                "--n_positions",
                str(n_positions),
                "--max_new_tokens",
                str(max_tokens),
                "--batch-size",
                str(batch_size),
                "--prompt-len",
                str(prompt_len),
            ]

            logger.info(f"Running NCU with command: {' '.join(ncu_cmd)}")

            # Run NCU
            subprocess.run(
                ncu_cmd, stdin=subprocess.DEVNULL, cwd=os.getcwd(), check=True
            )

            logger.success(f"Raw metrics data saved to {output_file}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running executable or NCU: {e}")
            if hasattr(e, "stderr"):
                logger.error(f"Error output: {e.stderr}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_model_args(args):
    model_args = [
        f"--model-type={args.model_type}",
        f"--n_layer={args.n_layer}",
        f"--hidden-dim={args.hidden_dim}",
        f"--ffn-dim={args.ffn_dim}",
        f"--n_head={args.n_head}",
        f"--n_positions={args.n_positions}",
    ]
    return model_args


def main():
    parser = argparse.ArgumentParser(description="Collect inference metrics with ncu.")
    # Model args
    parser.add_argument(
        "--model-type", type=str, choices=["gpt2", "llama", "qwen"], default="gpt2"
    )
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--ffn-dim", type=int, default=4096)
    parser.add_argument("--n_head", type=int, default=16)
    parser.add_argument("--n_positions", type=int, default=512)
    # Sampling/inference args
    parser.add_argument("--max_tokens", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-len", type=int, default=1)
    # Output args
    parser.add_argument(
        "--target-dir",
        type=str,
        default="data/lower_bound/4090/raw",
        help="Where to save ncu output and logs",
    )
    args = parser.parse_args()

    ensure_dir(args.target_dir)

    # 1. Run model creation
    model_cmd = [sys.executable, "eval/lower_bound/create_model.py"] + build_model_args(
        args
    )
    logger.info(f"Running model creation: {' '.join(model_cmd)}")
    subprocess.run(
        model_cmd,
        cwd=os.getcwd(),
        check=True,
    )

    # 2. Collect metrics
    collector = MetricsCollector("eval/lower_bound/inference.py")
    ncu_out = os.path.join(
        args.target_dir,
        get_raw_data_name(
            model_type=args.model_type,
            n_layer=args.n_layer,
            hidden_dim=args.hidden_dim,
            ffn_dim=args.ffn_dim,
            n_head=args.n_head,
            n_positions=args.n_positions,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            prompt_len=args.prompt_len,
        ),
    )
    logger.info(f"Collecting metrics with MetricsCollector, output: {ncu_out}")
    collector.collect_metrics(
        model_type=args.model_type,
        n_layer=args.n_layer,
        hidden_dim=args.hidden_dim,
        ffn_dim=args.ffn_dim,
        n_head=args.n_head,
        n_positions=args.n_positions,
        output_file=ncu_out,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        prompt_len=args.prompt_len,
    )

    # 3. Export to CSV
    collector.export_all_to_csv(
        raw_data_dir=os.path.join(args.target_dir),
        csv_data_dir=os.path.join(args.target_dir.replace("raw", "csv")),
    )


if __name__ == "__main__":
    main()
