#!/usr/bin/env python3
"""
Model Size Calculator for GPU-PMC-Verifier

This script calculates the total number of parameters for different model architectures
(GPT2, Llama, Qwen) based on their configuration parameters.

Key differences:
- GPT2: MLP has two linear layers (c_fc and c_proj) with intermediate activation
- Llama: MLP has three linear layers (gate, up, down) with SwiGLU activation
- Qwen: Similar to Llama with three linear layers
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional


def calculate_gpt2_parameters(
    n_layer: int,
    n_embd: int,
    n_inner: int,
    n_head: int,
    n_positions: int,
    vocab_size: int = 50257,
) -> int:
    """
    Calculate GPT2 model parameters.

    GPT2 MLP structure:
    - c_fc: n_embd -> n_inner
    - c_proj: n_inner -> n_embd

    Args:
        n_layer: Number of transformer layers
        n_embd: Hidden dimension (embedding size)
        n_inner: Feed-forward network dimension
        n_head: Number of attention heads
        n_positions: Maximum sequence length
        vocab_size: Vocabulary size (default: GPT2 default)

    Returns:
        Total number of parameters
    """
    # Token embedding
    token_embedding = vocab_size * n_embd

    # Position embedding
    position_embedding = n_positions * n_embd

    # Each transformer layer
    layer_params = 0
    for _ in range(n_layer):
        # Multi-head attention
        # q_proj, k_proj, v_proj: n_embd -> n_embd
        qkv_proj = 3 * n_embd * n_embd
        # out_proj: n_embd -> n_embd
        out_proj = n_embd * n_embd
        # Layer norms
        ln1 = 2 * n_embd  # weight + bias

        # MLP
        # c_fc: n_embd -> n_inner
        c_fc = n_embd * n_inner + n_inner  # weight + bias
        # c_proj: n_inner -> n_embd
        c_proj = n_inner * n_embd + n_embd  # weight + bias
        # Layer norm
        ln2 = 2 * n_embd  # weight + bias

        layer_params += qkv_proj + out_proj + ln1 + c_fc + c_proj + ln2

    # Final layer norm
    final_ln = 2 * n_embd  # weight + bias

    total_params = token_embedding + position_embedding + layer_params + final_ln
    return total_params


def calculate_llama_parameters(
    n_layer: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    max_position_embeddings: int,
    vocab_size: int = 32000,
) -> int:
    """
    Calculate Llama model parameters.

    Llama MLP structure:
    - gate_proj: hidden_size -> intermediate_size
    - up_proj: hidden_size -> intermediate_size
    - down_proj: intermediate_size -> hidden_size

    Args:
        n_layer: Number of transformer layers
        hidden_size: Hidden dimension
        intermediate_size: Feed-forward network dimension
        num_attention_heads: Number of attention heads
        max_position_embeddings: Maximum sequence length
        vocab_size: Vocabulary size (default: Llama default)

    Returns:
        Total number of parameters
    """
    # Token embedding
    token_embedding = vocab_size * hidden_size

    # Each transformer layer
    layer_params = 0
    for _ in range(n_layer):
        # Multi-head attention
        # q_proj, k_proj, v_proj: hidden_size -> hidden_size
        qkv_proj = 3 * hidden_size * hidden_size
        # o_proj: hidden_size -> hidden_size
        o_proj = hidden_size * hidden_size
        # Layer norms
        input_layernorm = 2 * hidden_size  # weight + bias
        post_attention_layernorm = 2 * hidden_size  # weight + bias

        # MLP
        # gate_proj: hidden_size -> intermediate_size
        gate_proj = hidden_size * intermediate_size + intermediate_size  # weight + bias
        # up_proj: hidden_size -> intermediate_size
        up_proj = hidden_size * intermediate_size + intermediate_size  # weight + bias
        # down_proj: intermediate_size -> hidden_size
        down_proj = intermediate_size * hidden_size + hidden_size  # weight + bias

        layer_params += (
            qkv_proj
            + o_proj
            + input_layernorm
            + post_attention_layernorm
            + gate_proj
            + up_proj
            + down_proj
        )

    # Final layer norm
    norm = 2 * hidden_size  # weight + bias

    total_params = token_embedding + layer_params + norm
    return total_params


def calculate_qwen_parameters(
    n_layer: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    max_position_embeddings: int,
    vocab_size: int = 151936,
) -> int:
    """
    Calculate Qwen model parameters.

    Qwen MLP structure is similar to Llama:
    - gate_proj: hidden_size -> intermediate_size
    - up_proj: hidden_size -> intermediate_size
    - down_proj: intermediate_size -> hidden_size

    Args:
        n_layer: Number of transformer layers
        hidden_size: Hidden dimension
        intermediate_size: Feed-forward network dimension
        num_attention_heads: Number of attention heads
        max_position_embeddings: Maximum sequence length
        vocab_size: Vocabulary size (default: Qwen default)

    Returns:
        Total number of parameters
    """
    # Qwen uses the same MLP structure as Llama
    return calculate_llama_parameters(
        n_layer,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        max_position_embeddings,
        vocab_size,
    )


def calculate_estimated_gpt2_parameters(
    n_layer: int,
    n_embd: int,
    n_inner: int,
    n_head: int,
    n_positions: int,
    vocab_size: int = 50257,
) -> int:
    """
    Calculate estimated GPT2 model parameters (excluding layernorm, position embedding, token embedding, and FFN bias).

    This gives a more focused view of core transformer parameters by excluding:
    - Layer normalization weights and biases
    - Position embeddings
    - Token embeddings
    - Bias terms in feed-forward networks

    Args:
        n_layer: Number of transformer layers
        n_embd: Hidden dimension (embedding size)
        n_inner: Feed-forward network dimension
        n_head: Number of attention heads
        n_positions: Maximum sequence length
        vocab_size: Vocabulary size (default: GPT2 default)

    Returns:
        Estimated number of parameters (transformer weights only)
    """
    # Each transformer layer
    layer_params = 0
    for _ in range(n_layer):
        # Multi-head attention
        # q_proj, k_proj, v_proj: n_embd -> n_embd (weights only, no bias)
        qkv_proj = 3 * n_embd * n_embd
        # out_proj: n_embd -> n_embd (weights only, no bias)
        out_proj = n_embd * n_embd

        # MLP (weights only, no bias)
        # c_fc: n_embd -> n_inner (weights only)
        c_fc = n_embd * n_inner
        # c_proj: n_inner -> n_embd (weights only)
        c_proj = n_inner * n_embd

        layer_params += qkv_proj + out_proj + c_fc + c_proj

    total_params = layer_params
    return total_params


def calculate_estimated_llama_parameters(
    n_layer: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    max_position_embeddings: int,
    vocab_size: int = 32000,
) -> int:
    """
    Calculate estimated Llama model parameters (excluding layernorm, position embedding, token embedding, and FFN bias).

    This gives a more focused view of core transformer parameters by excluding:
    - Layer normalization weights and biases
    - Position embeddings
    - Token embeddings
    - Bias terms in feed-forward networks

    Args:
        n_layer: Number of transformer layers
        hidden_size: Hidden dimension
        intermediate_size: Feed-forward network dimension
        num_attention_heads: Number of attention heads
        max_position_embeddings: Maximum sequence length
        vocab_size: Vocabulary size (default: Llama default)

    Returns:
        Estimated number of parameters (transformer weights only)
    """
    # Each transformer layer
    layer_params = 0
    for _ in range(n_layer):
        # Multi-head attention
        # q_proj, k_proj, v_proj: hidden_size -> hidden_size (weights only, no bias)
        qkv_proj = 3 * hidden_size * hidden_size
        # o_proj: hidden_size -> hidden_size (weights only, no bias)
        o_proj = hidden_size * hidden_size

        # MLP (weights only, no bias)
        # gate_proj: hidden_size -> intermediate_size (weights only)
        gate_proj = hidden_size * intermediate_size
        # up_proj: hidden_size -> intermediate_size (weights only)
        up_proj = hidden_size * intermediate_size
        # down_proj: intermediate_size -> hidden_size (weights only)
        down_proj = intermediate_size * hidden_size

        layer_params += qkv_proj + o_proj + gate_proj + up_proj + down_proj

    total_params = layer_params
    return total_params


def calculate_estimated_qwen_parameters(
    n_layer: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    max_position_embeddings: int,
    vocab_size: int = 151936,
) -> int:
    """
    Calculate estimated Qwen model parameters (excluding layernorm, position embedding, token embedding, and FFN bias).

    This gives a more focused view of core transformer parameters by excluding:
    - Layer normalization weights and biases
    - Position embeddings
    - Token embeddings
    - Bias terms in feed-forward networks

    Args:
        n_layer: Number of transformer layers
        hidden_size: Hidden dimension
        intermediate_size: Feed-forward network dimension
        num_attention_heads: Number of attention heads
        max_position_embeddings: Maximum sequence length
        vocab_size: Vocabulary size (default: Qwen default)

    Returns:
        Estimated number of parameters (transformer weights only)
    """
    # Qwen uses the same MLP structure as Llama
    return calculate_estimated_llama_parameters(
        n_layer,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        max_position_embeddings,
        vocab_size,
    )


def calculate_estimated_model_size(config: Dict[str, any]) -> Tuple[int, str]:
    """
    Calculate estimated model size based on configuration (excluding layernorm, position embedding, and FFN bias).

    Args:
        config: Model configuration dictionary

    Returns:
        Tuple of (estimated_parameter_count, human_readable_size)
    """
    model_type = config["model_type"].lower()
    n_layer = config["n_layer"]
    hidden_dim = config["hidden_dim"]
    ffn_dim = config["ffn_dim"]
    n_head = config["n_head"]
    n_positions = config["n_positions"]

    if model_type == "gpt2":
        param_count = calculate_estimated_gpt2_parameters(
            n_layer, hidden_dim, ffn_dim, n_head, n_positions
        )
    elif model_type == "llama":
        param_count = calculate_estimated_llama_parameters(
            n_layer, hidden_dim, ffn_dim, n_head, n_positions
        )
    elif model_type == "qwen":
        param_count = calculate_estimated_qwen_parameters(
            n_layer, hidden_dim, ffn_dim, n_head, n_positions
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Convert to human readable format
    if param_count >= 1e9:
        human_readable = f"{param_count / 1e9:.2f}B"
    elif param_count >= 1e6:
        human_readable = f"{param_count / 1e6:.2f}M"
    elif param_count >= 1e3:
        human_readable = f"{param_count / 1e3:.2f}K"
    else:
        human_readable = str(param_count)

    return param_count, human_readable


def parse_model_config_from_filename(filename: str) -> Optional[Dict[str, any]]:
    """
    Parse model configuration from filename.

    Expected format: {model_type}_layer{n_layer}_embd{hidden_dim}_ffn{ffn_dim}_head{n_head}_pos{n_positions}_max{max_tokens}_batch{batch_size}_prompt{prompt_len}

    Args:
        filename: Filename to parse

    Returns:
        Dictionary with parsed parameters or None if parsing fails
    """
    # Remove file extension
    name = Path(filename).stem

    # Pattern for the expected format
    pattern = r"^(\w+)_layer(\d+)_embd(\d+)_ffn(\d+)_head(\d+)_pos(\d+)_max(\d+)_batch(\d+)_prompt(\d+)$"

    match = re.match(pattern, name)
    if not match:
        return None

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


def calculate_model_size(config: Dict[str, any]) -> Tuple[int, str]:
    """
    Calculate model size based on configuration.

    Args:
        config: Model configuration dictionary

    Returns:
        Tuple of (parameter_count, human_readable_size)
    """
    model_type = config["model_type"].lower()
    n_layer = config["n_layer"]
    hidden_dim = config["hidden_dim"]
    ffn_dim = config["ffn_dim"]
    n_head = config["n_head"]
    n_positions = config["n_positions"]

    if model_type == "gpt2":
        param_count = calculate_gpt2_parameters(
            n_layer, hidden_dim, ffn_dim, n_head, n_positions
        )
    elif model_type == "llama":
        param_count = calculate_llama_parameters(
            n_layer, hidden_dim, ffn_dim, n_head, n_positions
        )
    elif model_type == "qwen":
        param_count = calculate_qwen_parameters(
            n_layer, hidden_dim, ffn_dim, n_head, n_positions
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Convert to human readable format
    if param_count >= 1e9:
        human_readable = f"{param_count / 1e9:.2f}B"
    elif param_count >= 1e6:
        human_readable = f"{param_count / 1e6:.2f}M"
    elif param_count >= 1e3:
        human_readable = f"{param_count / 1e3:.2f}K"
    else:
        human_readable = str(param_count)

    return param_count, human_readable


def process_data_folder(data_folder: str) -> None:
    """
    Process all model configurations in the data folder and calculate their sizes.

    Args:
        data_folder: Path to the data folder containing model configurations
    """
    data_path = Path(data_folder)

    if not data_path.exists():
        print(f"Error: Data folder '{data_folder}' does not exist")
        return

    if not data_path.is_dir():
        print(f"Error: '{data_folder}' is not a directory")
        return

    print(f"Processing model configurations in: {data_folder}")
    print("=" * 80)

    # Find all CSV files and other relevant files
    csv_files = list(data_path.glob("*.csv"))
    other_files = []

    # Look for files with model configuration patterns
    for file_path in data_path.iterdir():
        if file_path.is_file() and not file_path.suffix == ".csv":
            config = parse_model_config_from_filename(file_path.name)
            if config:
                other_files.append((file_path, config))

    results = []

    # Process CSV files first
    for csv_file in csv_files:
        print(f"Processing CSV file: {csv_file.name}")

        # Try to parse model config from CSV filename
        config = parse_model_config_from_filename(csv_file.name)
        if config:
            try:
                param_count, human_readable = calculate_model_size(config)
                estimated_param_count, estimated_human_readable = (
                    calculate_estimated_model_size(config)
                )
                results.append(
                    {
                        "source": csv_file.name,
                        "config": config,
                        "param_count": param_count,
                        "human_readable": human_readable,
                        "estimated_param_count": estimated_param_count,
                        "estimated_human_readable": estimated_human_readable,
                    }
                )
                print(
                    f"  ✓ {config['model_type'].upper()} - {human_readable} parameters (estimated: {estimated_human_readable})"
                )
            except Exception as e:
                print(f"  ✗ Error calculating size: {e}")
        else:
            print("  ⚠ Could not parse model config from filename")

    # Process other files with model configurations
    for file_path, config in other_files:
        try:
            param_count, human_readable = calculate_model_size(config)
            estimated_param_count, estimated_human_readable = (
                calculate_estimated_model_size(config)
            )
            results.append(
                {
                    "source": file_path.name,
                    "config": config,
                    "param_count": param_count,
                    "human_readable": human_readable,
                    "estimated_param_count": estimated_param_count,
                    "estimated_human_readable": estimated_human_readable,
                }
            )
            print(
                f"✓ {file_path.name}: {config['model_type'].upper()} - {human_readable} parameters (estimated: {estimated_human_readable})"
            )
        except Exception as e:
            print(f"✗ {file_path.name}: Error calculating size - {e}")

    if not results:
        print("No model configurations found in the data folder.")
        return

    # Sort results by parameter count
    results.sort(key=lambda x: x["param_count"])

    # Print summary
    print("\n" + "=" * 80)
    print("MODEL SIZE SUMMARY")
    print("=" * 80)

    for result in results:
        config = result["config"]
        print(
            f"{config['model_type'].upper():<6} | "
            f"Layers: {config['n_layer']:<2} | "
            f"Hidden: {config['hidden_dim']:<4} | "
            f"FFN: {config['ffn_dim']:<5} | "
            f"Heads: {config['n_head']:<2} | "
            f"Full: {result['human_readable']:<8} | "
            f"Est: {result['estimated_human_readable']:<8} | "
            f"Source: {result['source']}"
        )

    # Save results to CSV
    output_csv = data_path / "model_sizes.csv"
    with open(output_csv, "w") as f:
        f.write(
            "model_type,n_layer,hidden_dim,ffn_dim,n_head,n_positions,param_count,human_readable,estimated_param_count,estimated_human_readable,source\n"
        )
        for result in results:
            config = result["config"]
            f.write(
                f"{config['model_type']},{config['n_layer']},{config['hidden_dim']},"
                f"{config['ffn_dim']},{config['n_head']},{config['n_positions']},"
                f"{result['param_count']},{result['human_readable']},"
                f"{result['estimated_param_count']},{result['estimated_human_readable']},"
                f"{result['source']}\n"
            )

    print(f"\nDetailed results saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate model sizes for different architectures (GPT2, Llama, Qwen)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/calculate_model_size.py data/lower_bound/4090/csv/
  python scripts/calculate_model_size.py data/overhead/4090/
        """,
    )

    parser.add_argument(
        "data_folder", help="Path to the data folder containing model configurations"
    )

    args = parser.parse_args()

    try:
        process_data_folder(args.data_folder)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
