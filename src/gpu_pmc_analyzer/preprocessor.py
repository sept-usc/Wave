import pandas as pd
import os
import re
import argparse
from dataclasses import dataclass
from src.utils import logger


def col(df, name):
    if name in df.columns:
        return df[name]
    else:
        logger.warning(f"{name} is not in the column names, returning 0")
        return 0


def compute_float_flops(df):
    fma = col(df, "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum")
    mul = col(df, "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum")
    add = col(df, "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum")
    return 2 * fma + mul + add


def compute_half_flops(df):
    hfma = col(df, "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum")
    hmul = col(df, "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum")
    hadd = col(df, "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum")
    return 2 * hfma + hmul + hadd


def compute_double_flops(df):
    dfma = col(df, "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum")
    dmul = col(df, "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum")
    dadd = col(df, "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum")
    return 2 * dfma + dmul + dadd


def compute_tensor_flops(df):
    """
    Approximate Tensor Core FLOPs from Nsight tensor path counters.
    Each tensor core instruction group corresponds to 256 FMA.
    """
    tensor_inst = (
        col(df, "sm__ops_path_tensor_src_fp16_dst_fp16.sum")
        + col(df, "sm__ops_path_tensor_src_fp16_dst_fp32.sum")
        + col(df, "sm__ops_path_tensor_src_bf16_dst_fp32.sum")
        + col(df, "sm__ops_path_tensor_src_fp8.sum")
        + col(df, "sm__ops_path_tensor_src_tf32_dst_fp32.sum")
    )

    # one tensor tile -> 256 FMAs
    tensor_flops = tensor_inst * 256.0

    return tensor_flops


def compute_shared_ops(df):
    return col(df, "smsp__sass_inst_executed_op_shared_ld.sum") + col(
        df, "smsp__sass_inst_executed_op_shared_st.sum"
    )


def compute_global_ops(df):
    return col(df, "smsp__sass_inst_executed_op_global_ld.sum") + col(
        df, "smsp__sass_inst_executed_op_global_st.sum"
    )


def safe_ratio(numer, denom):
    denom = denom.replace(0, 1)
    return numer / denom


def compute_global_bytes(df):
    ld = col(df, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum")
    st = col(df, "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss.sum")
    return 32 * ld, 32 * st  # each sector = 32 bytes


@dataclass
class PMCConfig:
    batch_size: int
    hidden_dim: int
    ffn_dim: int
    timestamp: str


class PMCProcessor:
    def __init__(self, raw_dir):
        self.raw_dir = raw_dir

    def parse_filename(self, filename) -> PMCConfig:
        p = r".*embd(\d+)_ffn(\d+)_.*batch(\d+)_prompt(\d+)"
        m = re.match(p, filename)
        if not m:
            return PMCConfig(1, 0, 0, filename)

        hidden = int(m.group(1))
        ffn = int(m.group(2))
        batch = int(m.group(3))
        timestamp = filename.replace(".csv", "")
        return PMCConfig(batch, hidden, ffn, timestamp)

    def process_file(self, filename):
        file_path = os.path.join(self.raw_dir, filename)
        logger.info(f"Loading {file_path}")

        df = pd.read_csv(file_path, low_memory=False).iloc[1:].copy()

        for c in df.columns:
            if c != "Kernel Name":
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        # FLOPs
        float_flops = compute_float_flops(df)
        half_flops = compute_half_flops(df)
        double_flops = compute_double_flops(df)
        tensor_flops = compute_tensor_flops(df)

        total_flops = float_flops + half_flops + double_flops
        total_flops_with_tensor = total_flops + tensor_flops

        total_fma_ops = (
            col(df, "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum")
            + col(df, "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum")
            + col(df, "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum")
        )

        # Memory operations
        shared_ops = compute_shared_ops(df)
        global_ops = compute_global_ops(df)
        shared_ratio = safe_ratio(shared_ops, shared_ops + global_ops)

        # Global bytes
        ld_bytes, st_bytes = compute_global_bytes(df)

        # Kernel time
        kernel_time = col(df, "gpu__time_duration.sum")
        accumulated_time = kernel_time.cumsum()

        # Build output df
        out = pd.DataFrame(
            {
                "Kernel Name": df["Kernel Name"],
                "index": df.index.values,
                "total_fma_ops": total_fma_ops,
                "float_flops": float_flops,
                "half_flops": half_flops,
                "double_flops": double_flops,
                "tensor_flops": tensor_flops,
                "total_flops": total_flops,
                "total_flops_with_tensor": total_flops_with_tensor,
                "shared_ops": shared_ops,
                "external_memory_ops": global_ops,
                "shared_ratio": shared_ratio,
                "global_op_ld_lookup_miss_bytes": ld_bytes,
                "global_op_st_lookup_miss_bytes": st_bytes,
                "kernel_duration": kernel_time,
                "accumulated_time": accumulated_time,
                "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum": col(
                    df, "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"
                ),
                "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum": col(
                    df, "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"
                ),
                "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": col(
                    df, "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"
                ),
                "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum": col(
                    df, "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum"
                ),
                "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum": col(
                    df, "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum"
                ),
                "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum": col(
                    df, "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum"
                ),
                "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum": col(
                    df, "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum"
                ),
                "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum": col(
                    df, "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum"
                ),
                "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum": col(
                    df, "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum"
                ),
                "sm__ops_path_tensor_src_fp16_dst_fp16.sum": col(
                    df, "sm__ops_path_tensor_src_fp16_dst_fp16.sum"
                ),
                "sm__ops_path_tensor_src_fp16_dst_fp32.sum": col(
                    df, "sm__ops_path_tensor_src_fp16_dst_fp32.sum"
                ),
                "sm__ops_path_tensor_src_bf16_dst_fp32.sum": col(
                    df, "sm__ops_path_tensor_src_bf16_dst_fp32.sum"
                ),
                "sm__ops_path_tensor_src_fp8.sum": col(
                    df, "sm__ops_path_tensor_src_fp8.sum"
                ),
                "sm__ops_path_tensor_src_tf32_dst_fp32.sum": col(
                    df, "sm__ops_path_tensor_src_tf32_dst_fp32.sum"
                ),
                "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum": col(
                    df, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum"
                ),
                "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss.sum": col(
                    df, "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss.sum"
                ),
            }
        )

        return out

    def process_all(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)

        for fname in os.listdir(self.raw_dir):
            if fname.endswith(".csv"):
                df = self.process_file(fname)
                out_path = os.path.join(
                    out_dir, fname.replace(".csv", "_processed.csv")
                )
                df.to_csv(out_path, index=False)
                logger.success(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Process PMC CSV files")
    parser.add_argument("-d", "--directory", help="Directory of raw Nsight CSVs")
    parser.add_argument("-f", "--file", help="Single CSV file to process")
    parser.add_argument("-o", "--output", default="processed", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.directory:
        p = PMCProcessor(args.directory)
        p.process_all(args.output)

    elif args.file:
        folder = os.path.dirname(args.file)
        fname = os.path.basename(args.file)
        p = PMCProcessor(folder)
        df = p.process_file(fname)
        out_path = os.path.join(args.output, fname.replace(".csv", "_processed.csv"))
        df.to_csv(out_path, index=False)
        logger.success(f"Saved → {out_path}")

    else:
        logger.error("Must specify -d or -f")
        exit(1)


if __name__ == "__main__":
    main()
