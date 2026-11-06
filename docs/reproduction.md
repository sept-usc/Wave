# Reproduce the Wave artifact

Run commands from the repository root. **Replaying saved data** analyzes the committed measurements on a CPU. **Collecting new data** runs models and NVIDIA profiling tools and requires a suitable GPU. Neither route implements the authenticated trace-reporting service assumed by the research design.

## Environment

For solver replay and plots, install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync
```

The project requires Python 3.12. Its dependency set includes PyTorch, pandas, matplotlib, and Z3. Although CSV replay needs no GPU, the package imports its PyTorch-based split-attack implementation during initialization. The standard-library-only [overhead check](#profiling-overhead) is the smallest starting point.

For fresh measurements, the paper's environment uses NVIDIA RTX 4090, RTX 5080, or H100; CUDA 12.8; PyTorch 2.7.0; and Nsight Compute 2025.1.1.0 (4090/H100) or 2025.2.1.0 (5080). The dependency manifest permits newer package versions; consult `uv.lock` when recording the exact environment for a rerun. Overhead collection also uses [hyperfine](https://github.com/sharkdp/hyperfine), `bc`, and `jq`.

[`scripts/setup_environment.sh`](../scripts/setup_environment.sh) is an optional Linux setup helper. Read it before use: it can install tools and change NVIDIA profiling-access settings through `sudo`. It is unnecessary for replaying the saved overhead logs.

In the commands below, replace `<gpu_name>` with `4090`, `5080`, or `h100`. Collection outputs can overwrite matching data paths; use a separate checkout when retaining the original measurements matters.

## FLOPs and model scale

The motivating example varies the hidden dimension and layer count of a LLaMA-style model on RTX 5080. Start with the bundled data:

```bash
uv run scripts/run_preprocessing_pipeline.py data/motivating_example/5080/csv/
uv run scripts/analyze_motivating_example.py data/motivating_example/5080
```

Preprocessing writes intermediate files to `data/motivating_example/5080/preprocessed/` and role-labeled CSVs to `data/motivating_example/5080/processed/`. Plots go to [`figs/motivating_example/`](../figs/motivating_example/).

To collect a new dataset on a configured NVIDIA system:

```bash
bash scripts/collect_motivating_example_data.sh 5080
```

Raw reports and exported CSVs go under `data/motivating_example/5080/`. The supplied processed CSVs select a representative layer; do not interpret the sum of those selected projection rows as whole-request FLOPs.

## QKV load bytes

Compare measured projection load-miss bytes with the model's predicted memory traffic:

```bash
uv run scripts/plot_load_bytes.py --csv_dir data/lower_bound/h100/processed
```

Use `4090` or `5080` in the path for the other devices. Plots go to [`figs/load_bytes_observation/`](../figs/load_bytes_observation/). Existing plots are included for inspection.

## Model lower bound

This experiment asks whether counters admit a model smaller than a claimed minimum. Replay the supplied processed traces:

```bash
uv run scripts/verify_lower_bound.py --model-size tight --data-folder data/lower_bound/h100/processed/
uv run scripts/verify_lower_bound.py --model-size loose --data-folder data/lower_bound/h100/processed/
```

`tight` sets the promised minimum to 0.75 times the experiment's actual per-layer size, so the expected outcome is no violating solution. `loose` sets it to 1.25 times actual, so a violating solution is expected. These labels name experimental settings; they are not general solver assurance levels. Some recorded cases do not match the expected result; see the paper's evaluation and the script's summary.

The committed processed datasets contain 38 files for RTX 4090, 33 for RTX 5080, and 44 for H100. Models that did not fit on a device do not supply a trace. The original experiments request two generated tokens so that one decoding step follows prefill.

To process the bundled raw CSV exports again:

```bash
uv run scripts/run_preprocessing_pipeline.py data/lower_bound/h100/csv/
```

The pipeline writes `preprocessed/` and `processed/` next to `csv/`, then checks the assigned role order. New GPU measurements use:

```bash
bash scripts/collect_lower_bound_data.sh <gpu_name>
```

Raw Nsight reports go to `data/lower_bound/<gpu_name>/raw/`, and CSV exports go to `data/lower_bound/<gpu_name>/csv/`. Model configurations are listed in the collection script.

## Model upper bound

This experiment uses a one-layer GPT-2-like model whose Q/K/V/O and feed-forward projections can be split into smaller matrix operations. The implementation is [`src/gpu_pmc_verifier/split_attacker.py`](../src/gpu_pmc_verifier/split_attacker.py).

The provided 14 cases comprise one unsplit run, three all-split runs, and ten random cases with one to five split projections. The configuration uses batch size 4, decoding sequence length 1, hidden dimension 1024, and feed-forward dimension 4096.

Replay the processed CSVs:

```bash
uv run scripts/verify_upper_bound.py --model-size tight
uv run scripts/verify_upper_bound.py --model-size loose
```

The current script sets the promised maximum to **0.9 times actual** for `tight` and **1.1 times actual** for `loose`. The expected results are a violating solution and no violating solution, respectively. These ratios differ from the lower-bound experiment.

To repeat preprocessing of the saved CSV exports:

```bash
uv run scripts/run_upper_bound_pipeline.py
```

To collect fresh traces before preprocessing:

```bash
bash scripts/collect_upper_bound_data.sh
```

Collection writes split configurations, raw reports, and CSVs under `data/upper_bound/`. Preprocessing writes `preprocessed/` and `processed/` there.

## Interpret the solver results

The [`WaveVerifier`](../src/gpu_pmc_verifier/verifier.py) encodes a counterexample search. The constraints require matrix shapes to explain the counters while violating the requested bound:

- Lower bound: reconstructed per-layer size is **less than** the minimum.
- Upper bound: reconstructed per-layer size is **greater than** the maximum.

The scripts call finding a solution “Success” or “Verification succeeded.” That means the solver found a candidate violation. It does **not** certify the provider. “No solution” means a counterexample was not returned for the encoded model; the conclusion depends on the model, allowed transformations, and tolerances.

The current `verify()` API returns a dictionary for a satisfiable instance and `None` otherwise. It does not distinguish `unknown` from `unsat`. A production integration must preserve that distinction and treat unknown results as inconclusive. The CLI is an experiment driver, not a fail-closed remote attestation API.

The size expression is `4*d*d + g*d*d_ffn` per transformer layer, where `g` is the FFN projection count. It omits embeddings, biases, and normalization parameters. Comparisons to advertised full-model parameter counts need that distinction.

## Profiling overhead

Analyze saved logs with Python 3 and no third-party dependencies:

```bash
python3 scripts/analyze_overhead.py data/overhead/h100/all/overhead_summary.txt
```

Change the GPU path to `4090` or `5080` for the other logs. `all` measures the profiling configuration used by Wave; `hw` collects only `gpu__time_duration.sum`.

| GPU | Average full-metric overhead from bundled logs |
|---|---:|
| RTX 4090 | 4688.50% |
| RTX 5080 | 3006.67% |
| H100 | 1340.00% |

Overhead is `(profiled runtime / baseline runtime − 1) × 100%`. H100's average therefore corresponds to approximately 14.4 times baseline runtime, not 13.4 times. These measurements characterize this profiling setup; they do not measure a future hardware-integrated monitor.

Fresh timing collection requires the configured GPU and timing tools:

```bash
bash scripts/evaluate_overhead.sh <gpu_name> <hw/all>
```

Replace `<hw/all>` with either `hw` or `all`. Results are written under `data/overhead/<gpu_name>/<mode>/`.

## Documentation verification

The 2026-11-06 documentation update reanalyzed the three bundled full-metric overhead summaries and checked dataset counts, documented paths, and threshold constants against source. It did not collect new GPU traces or rerun the complete solver suite. Evidence and retained results are in [`editorial-evidence/`](editorial-evidence/).
