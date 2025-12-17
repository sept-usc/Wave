import argparse

from gpu_pmc_verifier import SplitAttacker


def main():
    """Attempt matrix splitting attacks to test upper bound check."""
    parser = argparse.ArgumentParser(
        description="Attempt matrix splitting attacks to evade upper bound check"
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=4, help="Batch size (default: 4)"
    )
    parser.add_argument(
        "--seq-len", "-s", type=int, default=1, help="Sequence length (default: 1)"
    )
    parser.add_argument(
        "--hidden-dim",
        "-d",
        type=int,
        default=1024,
        help="Hidden dimension (default: 1024)",
    )
    parser.add_argument(
        "--ffn-dim",
        "-f",
        type=int,
        default=4096,
        help="Feed-forward network dimension (default: 4096)",
    )
    parser.add_argument(
        "--granularity",
        "-g",
        type=int,
        default=256,
        help="Granularity for matrix operations (default: 256)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Enable test mode to verify correctness against reference results",
    )

    args = parser.parse_args()
    b = args.batch_size
    s = args.seq_len
    d = args.hidden_dim
    d_ffn = args.ffn_dim

    attacker = SplitAttacker(seed=args.seed, granularity=args.granularity)
    kernels, results = attacker.attempt_attack(b, s, d, d_ffn, test_mode=args.test)


if __name__ == "__main__":
    main()
