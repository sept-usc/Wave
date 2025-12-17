import argparse
import os
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    LlamaConfig,
    AutoModelForCausalLM,
    LlamaTokenizer,
    Qwen2Config,
    Qwen2Tokenizer,
)


def get_model_dir(model_type, n_layer, n_embd, ffn_dim, n_head, n_positions):
    """Generate model directory name based on configuration parameters."""
    name = f"{model_type}_layer{n_layer}_embd{n_embd}_ffn{ffn_dim}_head{n_head}_pos{n_positions}"
    return os.path.join("models", name)


def create_model(
    model_type,
    n_layer,
    n_embd,
    ffn_dim,
    n_head,
    n_positions,
    save_dir=None,
):
    """Create and save a model configuration with the specified parameters."""
    if save_dir is None:
        raise ValueError("save_dir must be provided")

    if model_type == "gpt2":
        config = GPT2Config(
            n_layer=n_layer,
            n_embd=n_embd,
            n_inner=ffn_dim,
            n_head=n_head,
            n_positions=n_positions,
            vocab_size=50257,  # default GPT-2 vocab
        )
        model = GPT2LMHeadModel(config)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    elif model_type == "llama":
        config = LlamaConfig(
            max_position_embeddings=n_positions,
            hidden_size=n_embd,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            vocab_size=32000,  # default Llama vocab size
            intermediate_size=ffn_dim if ffn_dim is not None else n_embd * 4,
            rms_norm_eps=1e-6,
        )
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.float32,
        )
        tokenizer = LlamaTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer"
        )
    elif model_type == "qwen":
        config = Qwen2Config(
            vocab_size=151936,  # default Qwen2 vocab size
            hidden_size=n_embd,
            intermediate_size=ffn_dim if ffn_dim is not None else n_embd * 4,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            num_key_value_heads=n_head,
            max_position_embeddings=n_positions,
        )
        model = AutoModelForCausalLM.from_config(config).to("cuda")
        tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2-1.5B")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return save_dir


def main():
    parser = argparse.ArgumentParser(
        description="Create custom models for vLLM evaluation"
    )

    # Model parameters
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["gpt2", "llama", "qwen"],
        default="gpt2",
        help="Type of model to create (gpt2, llama, or qwen)",
    )
    parser.add_argument(
        "--n_layer", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=1024, help="Hidden dimension (embedding size)"
    )
    parser.add_argument(
        "--ffn-dim",
        type=int,
        default=4096,
        help="Feed-forward network dimension (n_inner for GPT2, intermediate_size for Llama)",
    )
    parser.add_argument(
        "--n_head", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--n_positions", type=int, default=1024, help="Maximum sequence length"
    )
    parser.add_argument(
        "--output-dir", type=str, help="Custom output directory (optional)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force recreation even if model exists"
    )

    args = parser.parse_args()

    n_embd = args.hidden_dim
    ffn_dim = args.ffn_dim

    # Determine model path
    if args.output_dir:
        model_path = args.output_dir
    else:
        model_path = get_model_dir(
            args.model_type,
            args.n_layer,
            n_embd,
            ffn_dim,
            args.n_head,
            args.n_positions,
        )

    # Check if model already exists
    if (
        os.path.exists(model_path)
        and os.path.exists(os.path.join(model_path, "config.json"))
        and not args.force
    ):
        print(f"Model already exists at: {model_path}")
        print("Use --force to recreate the model")
        return

    # Create the model
    print(f"Creating {args.model_type} model with config:")
    print(f"  n_layer: {args.n_layer}")
    print(f"  n_embd: {n_embd}")
    print(f"  ffn_dim: {ffn_dim}")
    print(f"  n_head: {args.n_head}")
    print(f"  n_positions: {args.n_positions}")
    try:
        create_model(
            model_type=args.model_type,
            n_layer=args.n_layer,
            n_embd=n_embd,
            n_head=args.n_head,
            n_positions=args.n_positions,
            save_dir=model_path,
            ffn_dim=ffn_dim,
        )
        print(f"Model successfully created at: {model_path}")
    except Exception as e:
        print(f"Error creating model: {e}")
        return


if __name__ == "__main__":
    main()
