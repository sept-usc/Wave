## gpu_pmc_analyzer

This folder contains the analyzer used to detect periodic structure in kernel-level metrics exported from Nsight Compute (e.g., layer/token cycles), and to export a CSV for downstream verification.

## Quick test (recommended)

From the project root:

```bash
uv run scripts/run_preprocessing_pipeline.py data/cloud_inference/5080/csv/
```

- `scripts/run_preprocessing_pipeline.py` reads input CSVs under the provided directory and invokes `src/gpu_pmc_analyzer/analyzer.py` to run cycle detection and export results.
- If you need a different output location or model family (`gpt2`/`llama`/`qwen`), adjust the corresponding options in `scripts/run_preprocessing_pipeline.py`.
