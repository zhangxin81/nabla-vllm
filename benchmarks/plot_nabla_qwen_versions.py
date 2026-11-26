# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot Qwen serving curves from the checked-in first-round measurements."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

REPORT_DIR = Path(__file__).resolve().parents[1] / "docs/benchmarks/nabla-vllm"
METRICS = (
    ("throughput", "Output throughput (tok/s)", "higher is better"),
    ("ttft", "TTFT p50 (ms)", "lower is better"),
    ("tpot", "TPOT p50 (ms)", "lower is better"),
)


def read_csv(name):
    with (REPORT_DIR / name).open(newline="") as source:
        return list(csv.DictReader(source))


def first_round_data():
    specs = (
        (
            "base",
            read_csv("v02-qwen-serving-comparison.csv"),
            lambda r: r["baseline"] == "upstream_nixl_same_host_round1",
            ("input_tokens", "output_tokens", "concurrency"),
            (
                "baseline_output_tok_s",
                "baseline_ttft_p50_ms",
                "baseline_tpot_p50_ms",
            ),
            "requests_per_round",
        ),
        (
            "v0.1",
            read_csv("all-cases.csv"),
            lambda r: (
                r["model"] == "qwen_fp8_historical"
                and r["arm"] == "nabla_nixl"
                and r["round"] == "1"
            ),
            ("input_tokens", "output_tokens", "concurrency"),
            ("output_tok_s", "ttft_p50_ms", "tpot_p50_ms"),
            "requests",
        ),
        (
            "v0.2",
            read_csv("v02-qwen-serving-cases.csv"),
            lambda r: (
                r["kind"] == "main"
                and r["arm"] == "nabla_v02_nixl"
                and r["round"] == "1"
            ),
            ("isl", "osl", "concurrency"),
            ("output_throughput", "median_ttft_ms", "median_tpot_ms"),
            "requests",
        ),
    )
    data = {}
    for label, rows, select, key_fields, metrics, requests in specs:
        selected = [row for row in rows if select(row)]
        points = {}
        for row in selected:
            key = tuple(int(row[field]) for field in key_fields)
            values = tuple(float(row[field]) for field in metrics)
            assert key not in points, (label, "duplicate shape", key)
            assert int(row[requests]) == max(8, 2 * key[2]), (label, key)
            assert all(math.isfinite(v) and v > 0 for v in values), (label, key)
            points[key] = values
        assert len(points) == 39, (label, len(points))
        data[label] = points
    assert data["base"].keys() == data["v0.1"].keys() == data["v0.2"].keys()
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png-dir", type=Path, help="Also save PNG previews here")
    args = parser.parse_args()
    data = first_round_data()
    shapes = sorted({key[:2] for key in data["base"]})
    assert len(shapes) == 9
    styles = (
        ("base", "#667085", "o", "-"),
        ("v0.1", "#D97706", "s", "--"),
        ("v0.2", "#087EBA", "^", "-"),
    )
    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "nabla-qwen-versions"})
    for index, (slug, title, direction) in enumerate(METRICS):
        fig, axes = plt.subplots(3, 3, figsize=(13.2, 10.2))
        for ax, shape in zip(axes.flat, shapes):
            keys = sorted(key for key in data["base"] if key[:2] == shape)
            concurrency = [key[2] for key in keys]
            for label, color, marker, linestyle in styles:
                ax.plot(
                    concurrency,
                    [data[label][key][index] for key in keys],
                    label=label,
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.8,
                    markersize=5,
                    markerfacecolor="white" if label != "v0.2" else color,
                )
            ax.set_title(f"Input {shape[0]:,} / output {shape[1]:,} tokens")
            ax.set_xscale("log", base=2)
            ax.set_xticks(concurrency)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Concurrency")
            ax.set_ylabel(title)
            ax.grid(alpha=0.22)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"Qwen3.8-Flash-Next-FP8 | {title} | {direction}", y=0.98)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.955)
        )
        fig.text(
            0.5,
            0.018,
            "8 x H100, 4P4D, NIXL | All 39 matched shapes, round 1 | "
            "base/v0.2: same host; v0.1: historical measurements",
            ha="center",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0.045, 1, 0.915))
        stem = f"v02-qwen-versions-{slug}"
        svg_path = REPORT_DIR / f"{stem}.svg"
        fig.savefig(svg_path, metadata={"Date": None})
        svg_path.write_text(
            "\n".join(line.rstrip() for line in svg_path.read_text().splitlines())
            + "\n"
        )
        if args.png_dir:
            args.png_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.png_dir / f"{stem}.png", dpi=180)
        plt.close(fig)
    print("Validated 3 versions x 39 matched shapes; generated 3 comparison charts.")


if __name__ == "__main__":
    main()
