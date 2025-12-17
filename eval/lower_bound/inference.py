import argparse
import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from create_model import get_model_dir


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


# Set random seeds for reproducibility
seed = set_random_seeds(42)


# Sample prompts.
def get_prompts(batch_size, prompt_len, tokenizer, device):
    hello_token = "Hello"
    hello_id = tokenizer.encode(hello_token, add_special_tokens=False)[0]
    input_ids = torch.full(
        (batch_size, prompt_len), hello_id, dtype=torch.long, device=device
    )
    attention_mask = torch.ones(
        (batch_size, prompt_len), dtype=torch.long, device=device
    )
    return input_ids, attention_mask


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model using vLLM with configurable model parameters"
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
        "--n_head", type=int, default=16, help="Number of attention heads"
    )
    parser.add_argument(
        "--n_positions", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument("--granularity", type=int, default=256, help="Granularity")

    # Sampling parameters
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2,
        help="Maximum number of tokens to generate",
    )

    # Prompt parameters
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size (number of prompts)"
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=1,
        help="Prompt sequence length (truncate prompts)",
    )
    args = parser.parse_args()

    n_embd = args.hidden_dim
    ffn_dim = args.ffn_dim
    batch_size = args.batch_size
    prompt_len = args.prompt_len

    model_path = get_model_dir(
        args.model_type,
        args.n_layer,
        n_embd,
        ffn_dim,
        args.n_head,
        args.n_positions,
    )

    # Check if model exists
    if not os.path.exists(model_path) or not os.path.exists(
        os.path.join(model_path, "config.json")
    ):
        print(f"Error: Model not found at {model_path}")
        print("Please create the model first using create_model.py")
        print(
            f"Example: python create_model.py --model-type {args.model_type} --n_layer {args.n_layer} --hidden-dim {n_embd} --ffn-dim {ffn_dim} --n_head {args.n_head} --n_positions {args.n_positions}"
        )
        return

    print(f"Using existing model at: {model_path}")

    # Additional validation
    if not os.path.exists(os.path.join(model_path, "config.json")):
        print(f"Error: {model_path} does not contain a valid model configuration")
        return
    if not os.path.exists(
        os.path.join(model_path, "tokenizer.json")
    ) and not os.path.exists(os.path.join(model_path, "vocab.json")):
        print(
            f"Warning: {model_path} may not contain a tokenizer. This could cause issues."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_ids, attention_mask = get_prompts(batch_size, prompt_len, tokenizer, device)

    # Load model and tokenizer using HuggingFace Transformers
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.eval()
        model = model.to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        print(f"Generating with parameters: max_tokens={args.max_new_tokens}")
        print(f"Batch size: {batch_size}, Prompt seq len: {prompt_len}")
        print("\nGenerated Outputs:\n" + "-" * 60)
        for i in range(batch_size):
            prompt_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
            generated_text = tokenizer.decode(output_ids[i], skip_special_tokens=True)
            print(f"Prompt:    {prompt_text!r}")
            print(f"Output:    {generated_text!r}")
            print("-" * 60)
    except Exception as e:
        print(f"Error loading or running model: {e}")
        print("\nTroubleshooting tips:")
        print(
            "1. Make sure the model directory contains both model files and tokenizer files"
        )
        print("2. Check that the model path is correct")
        print("3. For custom models, ensure the configuration parameters are valid")
        return


if __name__ == "__main__":
    main()
