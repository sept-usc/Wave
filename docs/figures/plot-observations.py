#!/usr/bin/env python3
"""Plot selected per-layer projections from Wave's committed RTX 5080 data.

Run from any directory: python docs/figures/plot-observations.py
Requires matplotlib; does not run inference or change the source CSVs.
"""

import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATA = ROOT / "data/motivating_example/5080/processed"
records = []
inputs = []
for source in sorted(DATA.glob("llama_layer6_embd*_batch1_prompt1_processed.csv")):
    dimension = int(re.search(r"_embd(\d+)_", source.name).group(1))
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    record = {"hidden_dimension": dimension}
    for role in ("Q", "FFN1"):
        selected = [row for row in rows if row["role"] == role]
        if len(selected) != 1:
            raise ValueError(f"Expected one {role} row in {source.name}")
        record[role] = int(float(selected[0]["total_flops_with_tensor"]))
    record["Q_theory"] = 2 * dimension**2
    record["FFN1_theory"] = 8 * dimension**2
    records.append(record)
    inputs.append({"path": source.relative_to(ROOT).as_posix(),
                   "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
records.sort(key=lambda row: row["hidden_dimension"])
if [row["hidden_dimension"] for row in records] != [512, 768, 1024, 2048, 4096]:
    raise ValueError("Unexpected source dimension grid")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "svg.hashsalt": "wave-observed-flops-v1",
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(11, 7), dpi=160, facecolor="#faf9f5")
ax.set_facecolor("#faf9f5")
fig.subplots_adjust(left=.12, right=.95, top=.79, bottom=.22)
fig.text(.12, .94, "GPU work grows with hidden dimension", fontsize=22, weight="bold", color="#17252c")
fig.text(.12, .885, "Bundled RTX 5080 observations · LLaMA-style template · batch 1", color="#50616a")
dimensions = [row["hidden_dimension"] for row in records]
for role, label, color, marker, formula in [
    ("Q", "Q projection", "#116b81", "o", "2d²"),
    ("FFN1", "FFN-up projection", "#ad4e27", "s", "8d²"),
]:
    ax.plot(dimensions, [row[role] / 1e6 for row in records], color=color,
            marker=marker, linewidth=2.4, markersize=7, label=f"{label}: observed")
    ax.plot(dimensions, [row[role + "_theory"] / 1e6 for row in records],
            color=color, linestyle="--", alpha=.7, linewidth=1.8, label=f"{label}: {formula}")
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xticks(dimensions, [str(value) for value in dimensions])
ax.set_yticks([.5, 2, 8, 32, 128], ["0.5", "2", "8", "32", "128"])
ax.set_xlabel("Hidden dimension d", labelpad=10)
ax.set_ylabel("Selected projection FLOPs (millions)", labelpad=10)
ax.grid(axis="y", color="#ced4d5", linewidth=.7)
ax.set_axisbelow(True)
ax.legend(loc="upper left", fontsize=10.5, frameon=False)
fig.text(.12, .105, "Each point is one selected per-layer kernel, not a whole inference request.", color="#30474f", fontsize=11)
fig.text(.12, .065, "Six-layer configuration; FFN width = 4d. Both axes use logarithmic spacing. No new GPU run.", color="#50616a", fontsize=10)
fig.savefig(OUT / "wave-observed-flops.svg", metadata={"Date": None, "Creator": "Wave observation plot"})
fig.savefig(OUT / "wave-observed-flops.png", dpi=160, metadata={"Software": "Wave observation plot"})
(OUT / "wave-observed-flops.json").write_text(json.dumps({
    "source_commit": "29773cdce6bf06313dc340bce5df86d920e2ed21",
    "measurement": "total_flops_with_tensor for the unique Q and FFN1 rows",
    "scope": "Selected per-layer kernels, batch 1, six-layer LLaMA-style template, RTX 5080",
    "new_gpu_run": False, "inputs": inputs, "records": records,
}, indent=2) + "\n")
print(json.dumps(records, indent=2))
