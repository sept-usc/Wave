# Wave figures

These original explanatory diagrams accompany the repository README and the bilingual verifiable.pro article. They distinguish the paper's proposed measurement trust assumptions from the released profiling prototype. The animations illustrate reasoning; the separate observation plot uses the repository's committed data.

| Figure | Purpose | Files |
| --- | --- | --- |
| Observation flow | Follow private GPU execution through counter observations and architecture constraints to a model-size bound check. The authenticated measurement path remains an assumption. | [GIF](wave-observation.gif), [MP4](wave-observation.mp4), [PNG](wave-observation.png), [SVG](wave-observation.svg), [Excalidraw](wave-observation.excalidraw) |
| Bound search | Show why finding a feasible violating candidate leaves a claimed minimum or maximum uncertified. The example range of 80–120 uses arbitrary units. | [GIF](wave-bounds.gif), [MP4](wave-bounds.mp4), [PNG](wave-bounds.png), [SVG](wave-bounds.svg), [Excalidraw](wave-bounds.excalidraw) |
| Observed projection FLOPs | Compare the bundled RTX 5080 Q and FFN-up kernel observations with the template calculations `2d²` and `8d²`. | [PNG](wave-observed-flops.png), [SVG](wave-observed-flops.svg), [data and source hashes](wave-observed-flops.json) |

## Read the figures

The observation animation runs for 12 seconds. Each stage explains what the next stage receives. The counter bars are schematic and do not represent experiment results. Wave does not identify exact weight values or establish that an answer is correct.

The bound-search animation runs for 16 seconds. Its four cases use one illustrative feasible set: parameter size `P` between 80 and 120. A minimum of 75 and a maximum of 125 have no violating candidate in that set. A minimum of 100 admits a smaller candidate at 90; a maximum of 100 admits a larger candidate at 110. These examples are neither paper measurements nor an actual solver trace. **SAT** identifies a violating feasible candidate. **UNSAT** excludes one only within the modeled constraints and assumptions. **UNKNOWN** remains inconclusive; the artifact's current `verify()` return value does not distinguish every non-SAT outcome.

The empirical plot uses the `total_flops_with_tensor` entries for the unique `Q` and `FFN1` rows in the committed six-layer, batch-1 RTX 5080 motivating-example CSVs. Each point represents a selected per-layer projection kernel, not total inference FLOPs. Both axes use logarithmic spacing. The source-file hashes and plotted values are in [wave-observed-flops.json](wave-observed-flops.json). Replotting these files is not a new GPU experiment.

## Edit and render

From the repository root, use a separate environment for the illustration tools:

```bash
python3 -m venv .venv-figures
.venv-figures/bin/python -m pip install -r docs/figures/requirements-animation.txt
.venv-figures/bin/python docs/figures/render-animations.py
```

The renderer produces PNG, SVG, Excalidraw, GIF and MP4 versions. MP4 uses H.264 with `yuv420p` pixels for browser playback. It uses `ffmpeg` on `PATH` or the binary provided by `imageio-ffmpeg`. Each video is 1280 × 800 at 10 frames per second. GIF encoding combines identical held frames while retaining the full 12- or 16-second timeline.

The text and cases are editable in [wave-observation-spec.json](wave-observation-spec.json) and [wave-bounds-spec.json](wave-bounds-spec.json). Their canvas and feasible-range fields document the fixed scene; structural changes belong in [render-animations.py](render-animations.py). SVG and Excalidraw exports can also be edited directly for presentations. Rerendering replaces those exports.

The raster renderer uses Helvetica on macOS, with DejaVu Sans, Liberation Sans and Arial fallbacks for other platforms. Rasterization can differ across fonts and operating systems. It checks text widths to catch overflowing labels.

The committed animations were rendered with Python 3.12, Pillow 13.0.0 and ffmpeg 9.0, supplied by imageio-ffmpeg 0.8.1, on macOS arm64. The validation JSON files record hashes of the exported artwork and media.

To recreate the empirical plot:

```bash
.venv-figures/bin/python -m pip install matplotlib
.venv-figures/bin/python docs/figures/plot-observations.py
```

This reads the committed CSVs, writes the plot and its provenance JSON, and does not run inference or modify the CSVs.

## Sources and validation

The technical source is [Wave's ASPLOS 2026 paper](https://doi.org/10.1145/3779212.3790247), available as an [author-hosted PDF](https://terryxu.site/assets/pdf/wave-asplos26.pdf), checked against [artifact commit `29773cd`](https://github.com/sept-usc/Wave/tree/29773cdce6bf06313dc340bce5df86d920e2ed21). These are original schematic redraws, not copied paper figures. The measurement plot redraws the repository's supplied observations with source hashes recorded alongside it.

[animation-provenance.json](animation-provenance.json) records the creation method, source URLs and bilingual alternative text and captions. The two validation reports check dimensions, decoded GIF duration, nonzero frame differences, unique Excalidraw IDs, text font families and empty embedded-file dictionaries. The PNG posters and four representative frames per animation were visually reviewed. Both MP4s were decoded in full with ffmpeg: 120 frames for the observation flow and 160 frames for the bound search. No GPU experiment was run to create these assets.

The website should start videos paused, expose native playback controls and show the PNG poster. The GIF links provide a separate animated download; the PNG/SVG files provide still alternatives. Repeat each mechanism's explanation in article prose so the video is optional.

## License

The original rendered diagrams, editable artwork and newly drawn empirical plot are dual licensed under the repository's [MIT license](../../LICENSE) or [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), at the recipient's choice. Credit the Wave contributors and verifiable.pro. The original measurement data retains its repository license. The new renderer and plotting code use the repository's MIT license.

The bundled drawing and Excalidraw primitives in `_tools/lanshu_renderer.py` are adapted from the Lanshu animated architecture diagram skill, copyright © 2026 Lanshu, under the [included MIT license](_tools/LICENSE-lanshu.txt). That notice remains with the adapted code. No third-party icon or image library is embedded in the figures.
