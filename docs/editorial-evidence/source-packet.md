# Wave editorial source packet

Prepared 2026-11-06. This packet records source provenance and checks for the README and verifiable.pro explainer. It does not claim a new GPU reproduction.

## Primary sources

- Artifact: <https://github.com/sept-usc/Wave>, inspected at commit `29773cdce6bf06313dc340bce5df86d920e2ed21` (2025-12-24). The artifact is MIT-licensed; see [LICENSE](../../LICENSE).
- Author-hosted paper: <https://terryxu.site/assets/pdf/wave-asplos26.pdf>.
- Publication record: <https://doi.org/10.1145/3779212.3790247>, ASPLOS 2026, volume 2, pages 2212–2231.
- Citation cross-check: the publisher-deposited [Crossref record](https://api.crossref.org/works/10.1145/3779212.3790247) gives the proceedings title as the **31st** ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, with publication date 2026-03-22. Its ordinal agrees with the conference and author publication pages; the PDF's first-page reference text says 30th. The README follows the Crossref record.
- Author publication page: <https://terryxu.site/publications/>.
- Authors: Haoxuan Xu, Chen Gong, Beijie Liu, Haizhong Zheng, Beidi Chen, Mengyuan Li. The first three authors share first authorship. Mengyuan Li is the corresponding author.
- The paper and artifact are affiliated with SEPT Lab at USC. verifiable.pro is maintained by SEPT Lab; disclose this relationship in an article covering this work. Do not describe it as independent validation.

The paper PDF and extracted text were retained outside the Git checkout as editorial research inputs; they are not repository assets. The paper states a CC BY-NC-ND 4.0 license. The new explanatory figures use facts and the MIT artifact rather than republishing paper pages or modifying its figures.

## Article angle and claim boundaries

Wave asks whether a GPU's observed computation is consistent with an advertised dense model's structure and size. Matrix operations leave counter signatures; its analyzer finds repeated token/layer patterns, then its SMT solver searches for an explanation that violates a size promise. This is architecture oversight under a model and measurement assumptions. It does not establish exact weight identity, output correctness, model quality, or a deployed service's honesty.

Explain a concrete promise first: a service says it runs at least a specified model size. A smaller model can still produce plausible answers. Wave examines GPU performance monitoring counters (PMCs), rather than reading weights or prompts, for structural evidence.

The implementation maps each kernel to FLOPs, shared-memory instruction ratio, and aggregate load/store bytes (`src/gpu_pmc_analyzer/analyzer.py`, `build_feature_vectors`). It uses periodicity and a canonical role sequence. GPT-2's default signature is QKV, output projection, FFN1, FFN2 with FLOP ratios 3:1:4:4. LLaMA/Qwen defaults use separate Q/K/V and three FFN projections. Framework changes can change these patterns.

In the solver, **a found solution is a possible counterexample** to the size promise. The CLI's phrase “Verification succeeded” means the solver found that solution; it does not mean the provider's promise passed. “No solution” means no violating explanation was found under the encoded constraints. It is not unconditional proof of the real-world execution. The current `verify()` implementation also collapses non-SAT solver outcomes to `None`; do not advertise a production fail-closed verifier.

## Source-grounded mathematics for original figures

For matrix multiplication `(B,M,K) @ (K,N)`, the core solver uses a FLOP lower bound `2*B*M*K*N`. At element width `w` bytes, the modeled minimum load is `w*(B*M*K + K*N)`, and minimum store is `w*B*M*N`. Actual counters include effects from implementations; tolerances matter.

For one decoding token per batch item, a dense hidden-to-hidden projection uses approximately `2*b*d^2` FLOPs. Doubling hidden dimension `d` quadruples this projection's modeled work. With fixed dimensions, repeating layers scales work linearly.

The artifact's per-layer size expression is `4*d*d + g*d*d_ffn`, with `g=2` or `3` FFN projections. Multiplying by layer count approximates transformer-block parameters. It excludes embeddings and low-order terms; never present it as an exact full-checkpoint count.

The counterexample constraints are `< minimum` for a lower-bound promise and `> maximum` for an upper-bound promise. Two smaller matrix kernels may be explained as one split operation: splitting the kernel does not itself shrink the reconstructed model.

## Bundled measurements suitable for a figure

These values were read directly with Python's `csv` module from `data/motivating_example/5080/processed/`, taking rows labeled `Q` and `FFN1`. Configuration: LLaMA-style template, 6 layers, batch 1, FFN dimension `4*d`. Values below are selected per-layer projection FLOPs, not whole-request totals.

| Hidden dimension | Q measured FLOPs | Q theory (`2*d²`) | FFN-up measured FLOPs | FFN-up theory (`8*d²`) |
|---:|---:|---:|---:|---:|
| 512 | 613888 | 524288 | 2154496 | 2097152 |
| 768 | 1305600 | 1179648 | 5354496 | 4718592 |
| 1024 | 2265088 | 2097152 | 8503296 | 8388608 |
| 2048 | 8560640 | 8388608 | 37347328 | 33554432 |
| 4096 | 33767424 | 33554432 | 135069696 | 134217728 |

The committed lower-bound processed CSV counts are 38 for RTX 4090, 33 for RTX 5080, and 44 for H100. Avoid implying all 44 configurations were available on every GPU. Upper-bound data contains 14 cases.

## Read the evaluation carefully

- Paper §6 uses customized GPT-2, LLaMA and Qwen architecture templates, rather than pretrained checkpoints, with FP32 by default. Its abstract's 6.8% headline concerns recovered architectural parameters; §6.1 reports about 11% average model-size error. Do not call 6.8% a universal model-identification error or a detection success rate.
- Paper §6.2 lower-bound results include missed violations. Report the tested setup, not a universal guarantee.
- `scripts/verify_lower_bound.py` uses minimum thresholds 0.75× (tight) and 1.25× (loose) the actual per-layer size.
- `scripts/verify_upper_bound.py` uses maximum thresholds 0.9× (tight) and 1.1× (loose). The original README's 0.75×/1.25× upper-bound wording is incorrect.
- Paper §7 explicitly calls Wave a proof of concept. Secure signed per-run counters and binding to a particular query are assumptions and future integration work, not implemented end-to-end evidence in this repository.
- Scope: dense decoder models, single GPU, primarily decoding, static batching. Exact weights, fine-tuned identity, quantization, vLLM/SGLang-style optimizations, distributed inference, continuous batching, and MoE need separate support or are outside the evaluated scope.

## Checks actually run locally

Ran `python3 scripts/analyze_overhead.py data/overhead/<gpu>/all/overhead_summary.txt` for all three bundled logs. No new inference or profiling was performed.

| GPU | Minimum overhead | Maximum overhead | Average overhead |
|---|---:|---:|---:|
| RTX 4090 | 4094% | 5288% | 4688.50% |
| RTX 5080 | 2500% | 3443% | 3006.67% |
| H100 | 1196% | 1585% | 1340.00% |

These are overhead percentages `(profiled/baseline - 1)*100`, not percent of baseline. For example 1340% overhead means approximately 14.4× baseline runtime. Do not claim low overhead from the architectural potential discussed in the paper.

Inspected the preprocessing, verifier, collector, and setup entry points. The setup script can install tools and alter Linux profiling settings; do not run it merely to read the artifact. CPU-only overhead analysis requires only the Python standard library. The released verifier package eagerly imports the attack implementation, so its standard import path also needs PyTorch even for CSV replay.

## Figure plan and provenance

1. `wave-observation.gif` with `wave-observation.png` static fallback: original schematic of a model inference yielding counters, recovered layer pattern, and solver question. Mark the authenticated GPU-to-verifier evidence path as an assumption / deployment requirement. Do not draw a ready-made signed evidence service.
2. `wave-bounds.gif` with `wave-bounds.png` static fallback: original schematic of minimum/maximum promises and the counterexample search. Label a satisfying witness “possible violation”; make clear the verdict is conditional on the trace model and tolerances.

Both figures should have human-readable captions, static alternatives, and a generation script. The data table above can support an additional measured-versus-theory chart. Cite this commit's source files and the author paper in the figure provenance notes. Schematics must be labeled as explanatory, not measured runtime recordings.
