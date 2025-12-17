# Wave: Leveraging Architecture Observation for Privacy-Preserving Model Oversight

This repository contains the artifact for the paper "Wave: Leveraging Architecture Observation for Privacy-Preserving Model Oversight", accepted by ASPLOS 2026.

## Requirements

- Nvidia GPU (Tested on 4090, 5080, and H100)
- CUDA 12.8
- Nsight Compute 2025.1.1.0 or 2025.2.1.0
- PyTorch 2.7.0
- [uv](https://docs.astral.sh/uv/) for Python package management
- [hyperfine](https://github.com/sharkdp/hyperfine) for evaluating overhead

## Setup Environment

```bash
bash scripts/setup_environment.sh
```

## Motivating Example (Section 3.2)

We collect FLOP-relevant metrics on an Nvidia 5080 GPU to illustrate how FLOPs for matrix-multiplication kernels vary with the hidden dimension and number of layers of a LLaMA model.

### 1. Collect data:

```bash
bash scripts/collect_motivating_example_data.sh 5080
```

Outputs are written to `data/motivating_example/5080/` (raw reports and CSVs). Our collected CSVs are included under `data/motivating_example/5080/csv/`.

### 2. Data preprocessing:

```bash
uv run scripts/run_preprocessing_pipeline.py data/motivating_example/5080/csv/
```

Outputs: processed PMC CSV files under `data/motivating_example/5080/processed/`.

### 3. Generate plots:

```bash
uv run scripts/analyze_motivating_example.py data/motivating_example/5080
```

Plots are saved to `figs/motivating_example/`; the repository already contains the generated figures for reference.

## QKV load-byte Observation vs. Theoretical Values (Section 6.1)

```bash
uv run scripts/plot_load_bytes.py --csv_dir data/lower_bound/<gpu_name>/processed
```

Plots are saved to `figs/load_bytes_observation/`; the repository already contains the generated figures for reference.

## Model Lower Bound (Section 6.2.1)

We evaluate Wave's ability to enforce a minimum (lower-bound) model size, corresponding to the cloud inference scenario.

### 1. Collect data (multiple GPT2/Llama/Qwen configs):

```bash
bash scripts/collect_lower_bound_data.sh <gpu_name>
```

Outputs: raw NCU reports under `data/lower_bound/<gpu_name>/raw/` and CSV exports under `data/lower_bound/<gpu_name>/csv/`. 

Collected data for 4090, 5080, and H100 GPUs follow the same structure (see configs in `scripts/collect_lower_bound_data.sh`). The data uses 2 generated tokens (1 for prefill and 1 for decoding that is used in verification).

### 2. Data preprocessing:

```bash
uv run scripts/run_preprocessing_pipeline.py data/lower_bound/<gpu_name>/csv/
```

Outputs: processed PMC CSV files under `data/lower_bound/<gpu_name>/processed/`, with intermediate files under `data/lower_bound/<gpu_name>/preprocessed/`.

### 3. Verify:

```bash
uv run scripts/verify_lower_bound.py --model-size tight --data-folder data/lower_bound/<gpu_name>/processed/
```

or

```bash
uv run scripts/verify_lower_bound.py --model-size loose --data-folder data/lower_bound/<gpu_name>/processed/
```

Mode semantics:
- `tight`: claimed minimum size = 0.75× actual. Expect **no** solution; a solution is a false positive.
- `loose`: claimed minimum size = 1.25× actual. Expect a solution; missing one is a false negative.

## Model Upper Bound (Section 6.2.2)

We evaluate Wave's ability to detect model size violations when attackers split linear layers to evade the upper-bound check. The split attack implementation (`src/gpu_pmc_verifier/attacks/split_attacker.py`) randomly splits attention and feed-forward layers into multiple smaller matrix operations.

### 1. Collect data:

```bash
bash scripts/collect_upper_bound_data.sh
```

Outputs: split configurations, raw NCU reports, and CSVs under `data/upper_bound/`. Provided data includes one no-split case, three all-split cases, and ten random cases (1–5 split linear layers) for batch size 4, sequence length 1, hidden dim 1024, and FFN dim 4096.

### 2. Data preprocessing:

```bash
uv run scripts/run_upper_bound_pipeline.py
```

Outputs: processed PMC CSV files under `data/upper_bound/processed/`, with intermediate files under `data/upper_bound/preprocessed/`.

### 3. Verify:

```bash
uv run scripts/verify_upper_bound.py --model-size tight
```

or

```bash
uv run scripts/verify_upper_bound.py --model-size loose
```

Mode semantics:
- `tight`: claimed maximum size = 0.75× actual. Expect a solution; missing one is a false negative.
- `loose`: claimed maximum size = 1.25× actual. Expect **no** solution; finding one is a false positive.

## Overhead Evaluation (Section 7.1)

We measure inference runtime overhead from collecting metrics. The `hw` mode collects a single GPU timing metric, while `all` collects the full set of metrics used by Wave.

### 1. Collect data:

```bash
bash scripts/evaluate_overhead.sh <gpu_name> <hw/all>
```

Results are written to `data/overhead/<gpu_name>/<mode>/`. We provide collected results for 4090, 5080, and H100 in the same directory structure.

### 2. Analyze data:

```bash
uv run scripts/analyze_overhead.py data/overhead/<gpu_name>/<mode>/overhead_summary.txt
```
The script prints statistics of timings and overhead percentages for baseline vs. profiled runs.