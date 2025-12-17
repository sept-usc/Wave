import time
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from tabulate import tabulate
from gpu_pmc_verifier import (
    WaveVerifier,
    AlgorithmSpec,
    PMCData,
    OperationType,
)
from utils import logger
import pandas as pd
from dataclasses import dataclass


@dataclass
class PMCKernelData:
    """Data for a single kernel from PMC measurements."""

    flops_half: int
    flops_float: int
    flops_double: int
    gl_load: int
    gl_write: int
    kernel_name: str
    op_type: OperationType
    stage: Optional[int] = None  # Stage assignment for this kernel


class PMCProcessor:
    """Process PMC data from CSV files."""

    # Constants for PMC data processing
    SECTOR_SIZE = 32  # bytes
    FLOPS_THRESHOLD = 0

    def __init__(self, csv_path: str):
        """Initialize with path to CSV file."""
        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded PMC data from {csv_path}")

    def _get_metric(self, row: pd.Series, metric_name: str, default: int = 0) -> int:
        """Get and convert a PMC metric to integer.

        Args:
            row: DataFrame row containing the metric
            metric_name: Name of the metric column
            default: Default value if metric is not found

        Returns:
            Integer value of the metric
        """
        try:
            return int(float(row.get(metric_name, default)))
        except (ValueError, TypeError):
            logger.warning(
                f"Could not convert metric {metric_name}, using default {default}"
            )
            return default

    def _is_matrix_mul(self, row: pd.Series) -> bool:
        """Check if a kernel is likely a matrix multiplication based on PMC data."""
        # Filter based on shared memory load instructions
        shared_ld = self._get_metric(row, "sm__sass_inst_executed_op_shared_ld.sum")
        return shared_ld > self.FLOPS_THRESHOLD

    def _calculate_flops(self, row: pd.Series) -> Tuple[int, int, int]:
        """Calculate FLOPS for different precisions."""
        # Get instruction counts for half precision (FP16)
        h_add = self._get_metric(
            row, "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum"
        )
        h_mul = self._get_metric(
            row, "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum"
        )
        h_fma = self._get_metric(
            row, "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum"
        )
        flops_half = h_add + h_mul + (h_fma * 2)

        # Get instruction counts for single precision (FP32)
        f_add = self._get_metric(
            row, "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"
        )
        f_mul = self._get_metric(
            row, "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"
        )
        f_fma = self._get_metric(
            row, "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"
        )
        flops_float = f_add + f_mul + (f_fma * 2)

        # Get instruction counts for double precision (FP64)
        d_add = self._get_metric(
            row, "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum"
        )
        d_mul = self._get_metric(
            row, "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum"
        )
        d_fma = self._get_metric(
            row, "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum"
        )
        flops_double = d_add + d_mul + (d_fma * 2)

        return flops_half, flops_float, flops_double

    def _calculate_global_memory(self, row: pd.Series) -> Tuple[int, int]:
        """Calculate global memory operations."""
        # Calculate global load
        gl_load_miss = self._get_metric(
            row, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum"
        )
        gl_load = gl_load_miss * self.SECTOR_SIZE

        # Calculate global store
        gl_store_miss = self._get_metric(
            row, "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss.sum"
        )
        gl_write = gl_store_miss * self.SECTOR_SIZE

        return gl_load, gl_write

    def _get_kernel_name(self, row: pd.Series) -> str:
        """Extract kernel name from the row data."""
        return str(row["Kernel Name"])

    def _get_role(self, row: pd.Series) -> str:
        """Extract role from the row data."""
        return str(row["role"])

    def process_kernels(self) -> List[PMCKernelData]:
        """Process all kernels in the CSV file, including matmul and add kernels."""
        kernel_data = []
        labeled_set = ["matmul", "add"]
        add_index = 0
        for idx in range(1, len(self.df)):
            row = self.df.iloc[idx]
            kernel_name = self._get_kernel_name(row)
            role = self._get_role(row)
            if role in labeled_set:
                flops_half, flops_float, flops_double = self._calculate_flops(row)
                gl_load, gl_write = self._calculate_global_memory(row)
                stage = add_index
                if idx + 1 < len(self.df):
                    next_row = self.df.iloc[idx + 1]
                    next_kernel_role = self._get_role(next_row)
                    if "splitKreduce" == next_kernel_role:
                        _, gl_write = self._calculate_global_memory(next_row)
                        logger.info(
                            f"Using splitKreduce global store for kernel idx {idx}"
                        )
                if role == "add":
                    # NOTE: It uses FMA for the addition, so we divide by 2
                    flops_half = flops_half // 2
                    flops_float = flops_float // 2
                    flops_double = flops_double // 2
                    add_index += 1
                kernel_data.append(
                    PMCKernelData(
                        flops_half=flops_half,
                        flops_float=flops_float,
                        flops_double=flops_double,
                        gl_load=gl_load,
                        gl_write=gl_write,
                        kernel_name=kernel_name,
                        op_type=OperationType.MATMUL
                        if role != "add"
                        else OperationType.ADD,
                        stage=stage,
                    )
                )

        return kernel_data

    def get_pmc_data(
        self, detect_stages: bool = False
    ) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[str]]:
        """Get processed PMC data in the format expected by the verifier."""
        kernel_data = self.process_kernels()

        return (
            [k.flops_half for k in kernel_data],
            [k.flops_float for k in kernel_data],
            [k.flops_double for k in kernel_data],
            [k.gl_load for k in kernel_data],
            [k.gl_write for k in kernel_data],
            [k.kernel_name for k in kernel_data],
            [k.op_type for k in kernel_data],
            [k.stage for k in kernel_data] if detect_stages else None,
        )

    def print_kernel_info(self):
        """Print information about identified matrix multiplication kernels."""
        kernel_data = self.process_kernels()
        logger.info(f"Found {len(kernel_data)} matrix multiplication kernels:")

        # Group kernels by stage for better visualization
        stage_kernels = {}
        for kernel in kernel_data:
            stage = kernel.stage if kernel.stage is not None else 0
            if stage not in stage_kernels:
                stage_kernels[stage] = []
            stage_kernels[stage].append(kernel)

        for stage in sorted(stage_kernels.keys()):
            logger.info(f"\nStage {stage} ({len(stage_kernels[stage])} kernels):")
            for i, kernel in enumerate(stage_kernels[stage]):
                stage_info = (
                    f" (stage {kernel.stage})" if kernel.stage is not None else ""
                )
                logger.info(f"  Kernel {i + 1}: {kernel.kernel_name}{stage_info}")
                logger.info(
                    f"    FLOPS (half/float/double): {kernel.flops_half}/{kernel.flops_float}/{kernel.flops_double}"
                )
                logger.info(
                    f"    Global Memory (load/write): {kernel.gl_load}/{kernel.gl_write}"
                )


def process_csv_file(
    csv_path: str, algorithm_spec: AlgorithmSpec, expected_result: bool
) -> Tuple[bool, float]:
    """Process a single CSV file and verify its PMC data.

    Args:
        csv_path: Path to the CSV file
        algorithm_spec: Algorithm specification for verification

    Returns:
        Tuple[bool, float]: (verification success, time taken in seconds)
    """
    start_time = time.perf_counter()
    logger.info(f"Processing {csv_path}")

    # Process PMC data
    processor = PMCProcessor(csv_path)
    processor.print_kernel_info()

    # Get processed data
    (
        flops_half,
        flops_float,
        flops_double,
        gl_load,
        gl_write,
        kernel_names,
        op_types,
        stages,
    ) = processor.get_pmc_data(detect_stages=True)

    if not flops_half:  # No kernels found
        logger.warning(f"No valid kernels found in {csv_path}")
        return False, time.perf_counter() - start_time

    # Create PMC data
    pmc_data = PMCData(
        flops_half=flops_half,
        flops_float=flops_float,
        flops_double=flops_double,
        gl_load=gl_load,
        gl_write=gl_write,
        kernel_op_types=op_types,
        kernel_to_stage=stages,
    )

    # Create and run verifier
    verifier = WaveVerifier(
        algorithm_spec,
        pmc_data,
        matmul_pmc_upperbound_num=100,
        matmul_pmc_upperbound_den=1,
    )
    solution = verifier.verify()

    time_taken = time.perf_counter() - start_time

    if solution:
        logger.info(f"Found solution for {csv_path}:")
        logger.info(
            f"Parameters: b={solution['parameters']['b']}, "
            f"d={solution['parameters']['d']}, "
            f"d_ffn={solution['parameters']['d_ffn']}"
        )

        for i, config in enumerate(solution["kernel_configs"]):
            logger.info(
                f"Kernel {i}: op={config['op']}, "
                f"shape=({config['shape'][0]}, {config['shape'][1]}, {config['shape'][2]}) x "
                f"({config['shape'][2]}, {config['shape'][3]})"
            )

        for i in range(algorithm_spec.num_ops):
            logger.info(
                f"Operation {i}: kernels [{solution['op_start'][i]}, {solution['op_end'][i]}]"
            )
        logger.info(f"Verification completed in {time_taken:.2f} seconds")
        if expected_result:
            logger.success(
                f"Match expected: {'✓'} (expected {'Success' if expected_result else 'No solution'})"
            )
        else:
            logger.error(
                f"Match expected: {'✗'} (expected {'Success' if expected_result else 'No solution'})"
            )
        return True, time_taken
    else:
        logger.info(f"No solution found for {csv_path}")
        logger.info(f"Verification completed in {time_taken:.2f} seconds")
        if expected_result:
            logger.error(
                f"Match expected: {'✗'} (expected {'Success' if expected_result else 'No solution'})"
            )
        else:
            logger.success(
                f"Match expected: {'✓'} (expected {'Success' if expected_result else 'No solution'})"
            )
        return False, time_taken


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Verify PMC data with different model sizes"
    )
    parser.add_argument(
        "--model-size",
        choices=["tight", "loose"],
        default="tight",
        help="Model size constraint: 'tight' for -1, 'loose' for +1",
    )
    args = parser.parse_args()

    # Calculate max model size based on argument
    base_size = 4 * 1024 * 1024 + 2 * 1024 * 4096
    max_model_size = (
        int(base_size * 0.9) if args.model_size == "tight" else int(base_size * 1.1)
    )
    expected_result = True if args.model_size == "tight" else False

    logger.info(
        f"Testing with {'tight' if args.model_size == 'tight' else 'loose'} model size constraint"
    )
    logger.info(f"Max model size: {max_model_size}")
    logger.info(
        f"Expected verification result: {'Success' if expected_result else 'No solution'}"
    )

    # Define algorithm specification
    algorithm_spec = AlgorithmSpec(
        shapes=[
            (("b", "1", "d"), ("d", "d")),
            (("b", "1", "d"), ("d", "d")),
            (("b", "1", "d"), ("d", "d")),
            (("b", "1", "d"), ("d", "d")),
            (("b", "1", "d")),
            (("b", "1", "d"), ("d", "d_ffn")),
            (("b", "1", "d_ffn"), ("d_ffn", "d")),
            (("b", "1", "d")),
        ],
        num_ops=8,
        max_model_size=max_model_size,
        symbols=["b", "d", "d_ffn"],
        granularity=256,
        op_types=[
            OperationType.MATMUL,
            OperationType.MATMUL,
            OperationType.MATMUL,
            OperationType.MATMUL,
            OperationType.ADD,
            OperationType.MATMUL,
            OperationType.MATMUL,
            OperationType.ADD,
        ],
        op_to_stage=[0, 0, 0, 0, 0, 1, 1, 1],
    )

    # Get all CSV files in the data/csv directory
    csv_dir = Path(__file__).parent.parent / "data" / "upper_bound" / "processed"
    if not csv_dir.exists():
        logger.error(f"CSV directory not found: {csv_dir}")
        return

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {csv_dir}")
        return

    # Process each CSV file
    total_files = len(csv_files)
    successful_verifications = 0
    total_time = 0.0
    results: List[Dict] = []
    matched_results = 0

    for i, csv_file in enumerate(csv_files, 1):
        logger.info(f"Processing file {i}/{total_files}")
        success, time_taken = process_csv_file(
            str(csv_file), algorithm_spec, expected_result
        )
        if success:
            successful_verifications += 1
        total_time += time_taken

        # Check if result matches expected outcome
        matches_expected = success == expected_result
        if matches_expected:
            matched_results += 1

        # Record results for table
        results.append(
            {
                "File": csv_file.name,
                "Time (s)": f"{time_taken:.2f}",
                "Result": "Success" if success else "No solution",
                "Expected": "Success" if expected_result else "No solution",
                "Match": "✓" if matches_expected else "✗",
            }
        )

    # Print summary table
    logger.info("Verification Results Summary:")
    print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))

    # Print statistics
    logger.info("Statistics:")
    logger.info(f"Total files processed: {total_files}")
    logger.info(f"Results matching expectations: {matched_results}/{total_files}")
    logger.info(f"Total verification time: {total_time:.2f} seconds")
    # Per-file times are in results; compute mean/min/max
    per_times = [float(r["Time (s)"]) for r in results]
    if per_times:
        mean_time = sum(per_times) / len(per_times)
        min_time = min(per_times)
        max_time = max(per_times)
        logger.info(
            f"Verification time (s) — mean: {mean_time:.2f}, min: {min_time:.2f}, max: {max_time:.2f}"
        )


if __name__ == "__main__":
    main()
