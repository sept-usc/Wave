# GPU PMC analyzer

This package detects repeated token and layer patterns in kernel metrics exported from NVIDIA Nsight Compute. It assigns operation roles and writes CSVs for the downstream model-size checks.

From the repository root, install the project dependencies with `uv sync`, then process the bundled RTX 5080 data:

```bash
uv run scripts/run_preprocessing_pipeline.py data/lower_bound/5080/csv/
```

The pipeline runs the preprocessor, batch analyzer, and role-order check in sequence. Intermediate files go to `data/lower_bound/5080/preprocessed/`; role-labeled output goes to `data/lower_bound/5080/processed/`. Rerunning it can replace existing outputs.

The family-specific patterns for `gpt2`, `llama`, and `qwen` are defined in [`AnalyzerConfig.for_family`](analyzer.py). Those patterns and the periodicity assumptions determine which traces the analyzer can interpret; a matching role sequence alone is not evidence of exact model weights or output correctness.

See the [reproduction guide](../../docs/reproduction.md) for the full experiment workflow, solver-result semantics, and the distinction between saved-data replay and new GPU collection.
