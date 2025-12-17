import argparse
import time
import re
from pathlib import Path
from typing import List, Tuple, Union
import pandas as pd
from z3 import Int

from gpu_pmc_verifier import (
    WaveVerifier,
    AlgorithmSpec,
    PMCData,
    OperationType,
)
from utils import logger


def parse_model_params_from_filename(filename: str):
    # Example: gpt2_layer6_embd1024_ffn4096_head16_pos512_max2_batch4_prompt1_4090_analysis_output.csv
    layer = embd = ffn = None
    m = re.search(r"layer(\d+)", filename)
    if m:
        layer = int(m.group(1))
    m = re.search(r"embd(\d+)", filename)
    if m:
        embd = int(m.group(1))
    m = re.search(r"ffn(\d+)", filename)
    if m:
        ffn = int(m.group(1))
    return layer, embd, ffn


def parse_inference_params_from_filename(filename: str):
    # Example: gpt2_layer6_embd1024_ffn4096_head16_pos512_max2_batch4_prompt1_4090_analysis_output.csv
    max_tokens = batch_size = prompt_len = None
    m = re.search(r"max(\d+)", filename)
    if m:
        max_tokens = int(m.group(1))
    m = re.search(r"batch(\d+)", filename)
    if m:
        batch_size = int(m.group(1))
    m = re.search(r"prompt(\d+)", filename)
    if m:
        prompt_len = int(m.group(1))
    return max_tokens, batch_size, prompt_len


def reconstruct_algorithm_spec_and_pmcdata_from_csv(
    csv_path: str, model_size_mode: str = "tight"
) -> Tuple[AlgorithmSpec, PMCData]:
    df = pd.read_csv(csv_path)
    roles = df["role"].tolist()
    shapes: List[
        Union[Tuple[Tuple[str, str, str], Tuple[str, str]], Tuple[str, str, str]]
    ] = []
    op_types: List[OperationType] = []
    flops_half = []
    flops_float = []
    flops_double = []
    gl_load = []
    gl_write = []
    symbols = ["b", "d", "d_ffn"]
    symbol_constraints = []
    granularity = 256
    gates_num_in_mlp = 2
    ffn_count = roles.count("FFN1") + roles.count("FFN2") + roles.count("FFN3")

    layer, embd, ffn = parse_model_params_from_filename(Path(csv_path).name)
    if layer is None or embd is None or ffn is None:
        raise ValueError(f"Could not parse model params from filename: {csv_path}")
    _, batch_size, _ = parse_inference_params_from_filename(Path(csv_path).name)
    if batch_size is None:
        raise ValueError(f"Could not parse inference params from filename: {csv_path}")

    for i, role in enumerate(roles):
        if role == "QKV":
            shapes.append((("b", "1", "d"), ("d", "3d")))
            symbols.append("3d")
            symbol_constraints.append(Int("3d") == 3 * Int("d"))
            op_types.append(OperationType.MATMUL)
            flops_half.append(df["half_flops"][i])
            flops_float.append(df["float_flops"][i])
            flops_double.append(df["double_flops"][i])
            gl_load.append(df["global_op_ld_lookup_miss_bytes"][i])
            if i + 1 < len(df) and "splitKreduce" == df["role"][i + 1]:
                gl_write.append(df["global_op_st_lookup_miss_bytes"][i + 1])
            else:
                gl_write.append(df["global_op_st_lookup_miss_bytes"][i])
            theory_flops = 2 * batch_size * embd * 3 * embd
            theory_gl_load = 4 * batch_size * embd + 4 * embd * 3 * embd
            theory_gl_write = 4 * batch_size * 3 * embd
            print(f"role: {role}")
            print(
                f"Theory flops: {theory_flops}, theory gl_load: {theory_gl_load}, theory gl_write: {theory_gl_write}"
            )
            print(
                f"Actual flops: {flops_float[-1]}, actual gl_load: {gl_load[-1]}, actual gl_write: {gl_write[-1]}"
            )
        elif role in ["Q", "K", "V"]:
            shapes.append((("b", "1", "d"), ("d", "d")))
            op_types.append(OperationType.MATMUL)
            flops_half.append(df["half_flops"][i])
            flops_float.append(df["float_flops"][i])
            flops_double.append(df["double_flops"][i])
            gl_load.append(df["global_op_ld_lookup_miss_bytes"][i])
            if i + 1 < len(df) and "splitKreduce" == df["role"][i + 1]:
                gl_write.append(df["global_op_st_lookup_miss_bytes"][i + 1])
            else:
                gl_write.append(df["global_op_st_lookup_miss_bytes"][i])
            theory_flops = 2 * batch_size * embd * embd
            theory_gl_load = 4 * batch_size * embd + 4 * embd * embd
            theory_gl_write = 4 * batch_size * embd
            print(f"role: {role}")
            print(
                f"Theory flops: {theory_flops}, theory gl_load: {theory_gl_load}, theory gl_write: {theory_gl_write}"
            )
            print(
                f"Actual flops: {flops_float[-1]}, actual gl_load: {gl_load[-1]}, actual gl_write: {gl_write[-1]}"
            )
        elif role == "Wo":
            shapes.append((("b", "1", "d"), ("d", "d")))
            op_types.append(OperationType.MATMUL)
            flops_half.append(df["half_flops"][i])
            flops_float.append(df["float_flops"][i])
            flops_double.append(df["double_flops"][i])
            gl_load.append(df["global_op_ld_lookup_miss_bytes"][i])
            if i + 1 < len(df) and "splitKreduce" == df["role"][i + 1]:
                gl_write.append(df["global_op_st_lookup_miss_bytes"][i + 1])
            else:
                gl_write.append(df["global_op_st_lookup_miss_bytes"][i])
            theory_flops = 2 * batch_size * embd * embd
            theory_gl_load = 4 * batch_size * embd + 4 * embd * embd
            theory_gl_write = 4 * batch_size * embd
            print(f"role: {role}")
            print(
                f"Theory flops: {theory_flops}, theory gl_load: {theory_gl_load}, theory gl_write: {theory_gl_write}"
            )
            print(
                f"Actual flops: {flops_float[-1]}, actual gl_load: {gl_load[-1]}, actual gl_write: {gl_write[-1]}"
            )
        elif role == "elementwise_add":
            shapes.append(("b", "1", "d"))
            op_types.append(OperationType.ADD)
            flops_half.append(
                df["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"][i]
            )
            flops_float.append(
                df["smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"][i]
            )
            flops_double.append(
                df["smsp__sass_thread_inst_executed_op_dfma_pred_on.sum"][i]
            )
            gl_load.append(df["global_op_ld_lookup_miss_bytes"][i])
            gl_write.append(df["global_op_st_lookup_miss_bytes"][i])
            theory_flops = batch_size * embd
            theory_gl_load = 2 * 4 * batch_size * embd
            theory_gl_write = 4 * batch_size * embd
            print(f"role: {role}")
            print(
                f"Theory flops: {theory_flops}, theory gl_load: {theory_gl_load}, theory gl_write: {theory_gl_write}"
            )
            print(
                f"Actual flops: {flops_float[-1]}, actual gl_load: {gl_load[-1]}, actual gl_write: {gl_write[-1]}"
            )
        elif role == "FFN1":
            shapes.append((("b", "1", "d"), ("d", "d_ffn")))
            op_types.append(OperationType.MATMUL)
            flops_half.append(df["half_flops"][i])
            flops_float.append(df["float_flops"][i])
            flops_double.append(df["double_flops"][i])
            gl_load.append(df["global_op_ld_lookup_miss_bytes"][i])
            if i + 1 < len(df) and "splitKreduce" == df["role"][i + 1]:
                gl_write.append(df["global_op_st_lookup_miss_bytes"][i + 1])
            else:
                gl_write.append(df["global_op_st_lookup_miss_bytes"][i])
            theory_flops = 2 * batch_size * embd * ffn
            theory_gl_load = 4 * batch_size * embd + 4 * embd * ffn
            theory_gl_write = 4 * batch_size * ffn
            print(f"role: {role}")
            print(
                f"Theory flops: {theory_flops}, theory gl_load: {theory_gl_load}, theory gl_write: {theory_gl_write}"
            )
            print(
                f"Actual flops: {flops_float[-1]}, actual gl_load: {gl_load[-1]}, actual gl_write: {gl_write[-1]}"
            )
        elif role == "FFN2":
            if ffn_count == 2:
                shapes.append((("b", "1", "d_ffn"), ("d_ffn", "d")))
                theory_flops = 2 * batch_size * embd * ffn
                theory_gl_load = 4 * batch_size * ffn + 4 * embd * ffn
                theory_gl_write = 4 * batch_size * embd
                print(f"role: {role}")
                print(
                    f"Theory flops: {theory_flops}, theory gl_load: {theory_gl_load}, theory gl_write: {theory_gl_write}"
                )
                print(
                    f"Actual flops: {flops_float[-1]}, actual gl_load: {gl_load[-1]}, actual gl_write: {gl_write[-1]}"
                )
            else:
                shapes.append((("b", "1", "d"), ("d", "d_ffn")))
                theory_flops = 2 * batch_size * embd * ffn
                theory_gl_load = 4 * batch_size * embd + 4 * embd * ffn
                theory_gl_write = 4 * batch_size * ffn
                print(f"role: {role}")
                print(
                    f"Theory flops: {theory_flops}, theory gl_load: {theory_gl_load}, theory gl_write: {theory_gl_write}"
                )
                print(
                    f"Actual flops: {flops_float[-1]}, actual gl_load: {gl_load[-1]}, actual gl_write: {gl_write[-1]}"
                )
            op_types.append(OperationType.MATMUL)
            flops_half.append(df["half_flops"][i])
            flops_float.append(df["float_flops"][i])
            flops_double.append(df["double_flops"][i])
            gl_load.append(df["global_op_ld_lookup_miss_bytes"][i])
            if i + 1 < len(df) and "splitKreduce" == df["role"][i + 1]:
                gl_write.append(df["global_op_st_lookup_miss_bytes"][i + 1])
            else:
                gl_write.append(df["global_op_st_lookup_miss_bytes"][i])
        elif role == "FFN3":
            assert ffn_count == 3, "FFN3 should only appear in a model with 3 FFNs"
            shapes.append((("b", "1", "d_ffn"), ("d_ffn", "d")))
            op_types.append(OperationType.MATMUL)
            flops_half.append(df["half_flops"][i])
            flops_float.append(df["float_flops"][i])
            flops_double.append(df["double_flops"][i])
            gl_load.append(df["global_op_ld_lookup_miss_bytes"][i])
            if i + 1 < len(df) and "splitKreduce" == df["role"][i + 1]:
                gl_write.append(df["global_op_st_lookup_miss_bytes"][i + 1])
            else:
                gl_write.append(df["global_op_st_lookup_miss_bytes"][i])
            theory_flops = 2 * batch_size * embd * ffn
            theory_gl_load = 4 * batch_size * ffn + 4 * embd * ffn
            theory_gl_write = 4 * batch_size * embd
            gates_num_in_mlp = 3
            print(f"role: {role}")
            print(
                f"Theory flops: {theory_flops}, theory gl_load: {theory_gl_load}, theory gl_write: {theory_gl_write}"
            )
            print(
                f"Actual flops: {flops_float[-1]}, actual gl_load: {gl_load[-1]}, actual gl_write: {gl_write[-1]}"
            )

    # NOTE: it should be the given max model size divided by the number of layers
    base_model_size = embd * embd * 4 + embd * ffn * gates_num_in_mlp
    if model_size_mode == "tight":
        min_model_size = int(base_model_size * 0.75)
    else:
        min_model_size = int(base_model_size * 1.25)
    algorithm_spec = AlgorithmSpec(
        shapes=shapes,
        num_ops=len(shapes),
        min_model_size=min_model_size,
        symbols=symbols,
        granularity=granularity,
        op_types=op_types,
        symbol_constraints=symbol_constraints if symbol_constraints else None,
        gates_num_in_mlp=gates_num_in_mlp,
    )
    flops_half = [int(i) for i in flops_half]
    flops_float = [int(i) for i in flops_float]
    flops_double = [int(i) for i in flops_double]
    gl_load = [int(i) for i in gl_load]
    gl_write = [int(i) for i in gl_write]
    pmc_data = PMCData(
        flops_half=flops_half,
        flops_float=flops_float,
        flops_double=flops_double,
        gl_load=gl_load,
        gl_write=gl_write,
        kernel_op_types=op_types,
    )
    return algorithm_spec, pmc_data


def process_csv(
    csv_path: str, model_size_mode: str, expected_result: bool
) -> Tuple[str, bool, bool, float]:
    logger.info(f"Processing {csv_path}")
    start_time = 0.0  # Initialize start_time to avoid UnboundLocalError
    try:
        algorithm_spec, pmc_data = reconstruct_algorithm_spec_and_pmcdata_from_csv(
            csv_path, model_size_mode
        )
        verifier = WaveVerifier(algorithm_spec, pmc_data)
        start_time = time.perf_counter()
        solution = verifier.verify()
        elapsed = time.perf_counter() - start_time
        success = solution is not None
        match = success == expected_result
        if success:
            logger.info(f"Verification succeeded for {csv_path}")
            logger.info(f"Parameters: {solution['parameters']}")
        else:
            logger.info(f"Verification failed for {csv_path}")
        if match:
            logger.success(
                f"Match expected: {'✓'} (expected {'Success' if expected_result else 'No solution'})"
            )
        else:
            logger.error(
                f"Match expected: {'✗'} (expected {'Success' if expected_result else 'No solution'})"
            )
        logger.info(f"Verification time: {elapsed:.2f} seconds")
        return csv_path, success, match, elapsed
    except Exception as e:
        logger.error(f"Error processing {csv_path}: {e}")
        elapsed = time.perf_counter() - start_time
        logger.info(f"Verification time (failed): {elapsed:.2f} seconds")
        return csv_path, False, False, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Verify lower bound PMC data by reconstructing algorithm spec."
    )
    parser.add_argument(
        "--data-folder",
        type=str,
        default="data/lower_bound/4090/processed",
        help="Path to folder containing processed PMC CSV files",
    )
    parser.add_argument(
        "--model-size",
        choices=["tight", "loose"],
        default="tight",
        help="Model size constraint: 'tight' for base, 'loose' for +10%",
    )
    args = parser.parse_args()
    data_folder = Path(args.data_folder)
    model_size_mode = args.model_size
    expected_result = False if model_size_mode == "tight" else True
    if not data_folder.exists() or not data_folder.is_dir():
        logger.error(f"Data folder does not exist or is not a directory: {data_folder}")
        return
    csv_files = list(data_folder.rglob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {data_folder}")
        return
    total = len(csv_files)
    succeeded = 0
    failed = 0
    matched = 0
    times: List[float] = []
    for csv_path in csv_files:
        _, success, match, elapsed = process_csv(
            str(csv_path), model_size_mode, expected_result
        )
        if success:
            succeeded += 1
        else:
            failed += 1
        if match:
            matched += 1
        times.append(elapsed)
    logger.info("Summary:")
    logger.info(f"Total files processed: {total}")
    logger.info(f"Succeeded: {succeeded}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Results matching expectations: {matched}/{total}")
    if times:
        mean_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        logger.info(
            f"Verification time (s) — mean: {mean_time:.2f}, min: {min_time:.2f}, max: {max_time:.2f}"
        )


default_main = main

if __name__ == "__main__":
    main()
