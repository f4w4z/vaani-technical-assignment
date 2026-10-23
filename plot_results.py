"""Plot evaluate.py's report.json, samples.csv, or wer_vs_quality.csv.

Example: python plot_results.py --input outputs/evaluation/report.json
Only matplotlib is needed beyond the standard library. No model imports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


def read_results(path: str | Path) -> tuple[list[dict[str, Any]], str, int | None]:
    """Read successful sample records, the quality label, and known failure count.

    CSV does not contain metadata, so its quality label is intentionally generic.
    Use --quality-label to name a custom scorer when plotting CSV.
    """
    source = Path(path).expanduser()
    metric = "Response quality"
    failed: int | None = None
    if source.suffix.lower() == ".csv":
        with source.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif source.suffix.lower() == ".json":
        report = json.loads(source.read_text(encoding="utf-8-sig"))
        if not isinstance(report, dict):
            raise ValueError("JSON input must be an evaluation report with a samples array")
        rows = report.get("samples")
        metric = str(report.get("metadata", {}).get("quality_metric", metric))
        failed = report.get("summary", {}).get("samples_failed")
    else:
        raise ValueError("Input must be an evaluation CSV or JSON report")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Report has no successful sample records to plot")
    return rows, metric, failed


def _number(row: dict[str, Any], field: str) -> float | None:
    """Parse CSV/JSON numbers; missing values remain undefined, never become zero."""
    value = row.get(field)
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Sample {row.get('id', '?')}: invalid {field}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Sample {row.get('id', '?')}: nonfinite {field}")
    if field in {"asr_response_quality", "clean_response_quality"}:
        if not 0 <= parsed <= 1:
            raise ValueError(f"Sample {row.get('id', '?')}: {field} must be in [0, 1]")
    elif parsed < 0:
        raise ValueError(f"Sample {row.get('id', '?')}: {field} must be nonnegative")
    return parsed


def create_plots(
    input_path: str | Path,
    output_dir: str | Path = "outputs/plots",
    *,
    quality_label: str | None = None,
) -> list[Path]:
    """Save paired-quality scatter and mean-latency PNGs, returning their paths.

    Undefined WER/quality values are omitted from their series, with sample counts
    in the legend. If latency columns are absent, only the scatter is produced.
    Timing columns that are partially present are rejected to avoid reporting
    misleading averages from different populations. All plots use successful
    evaluator samples only. The plotting CSV has no failure-count metadata.
    """
    rows, metric, failed = read_results(input_path)
    quality_label = quality_label or ("Word overlap F1" if metric == "word_overlap_f1" else metric)
    series: dict[str, list[tuple[float, float]]] = {}
    for field in ("asr_response_quality", "clean_response_quality"):
        points = []
        for row in rows:
            wer, quality = _number(row, "wer"), _number(row, field)
            if wer is not None and quality is not None:
                points.append((wer, quality))
        if points:
            series[field] = points
    if not series:
        raise ValueError("No valid WER/response-quality pairs to plot")

    stage_fields = ("asr_seconds", "llm_seconds", "total_seconds")
    has_latency = any(field in row for row in rows for field in stage_fields)
    stages: list[tuple[float, float, float]] = []
    clean_times: list[float] = []
    if has_latency:
        for row in rows:
            asr, llm, total = (_number(row, field) for field in stage_fields)
            if asr is None or llm is None or total is None:
                raise ValueError("Latency plotting requires ASR, LLM, and total times for every sample")
            if total + 1e-6 < asr + llm:
                raise ValueError(f"Sample {row.get('id', '?')}: total latency is less than stage sum")
            stages.append((asr, llm, max(0.0, total - asr - llm)))
            clean = _number(row, "clean_llm_seconds")
            if clean is not None:
                clean_times.append(clean)
        if clean_times and len(clean_times) != len(rows):
            raise ValueError("Clean LLM latency must be provided for all samples or none")

    import matplotlib

    matplotlib.use("Agg")  # Save images without needing a desktop/display server.
    import matplotlib.pyplot as plt

    directory = Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    footer = f"Successful samples: {len(rows)}"
    footer += f" | Failed samples excluded: {failed}" if failed is not None else " | Failure count unavailable in CSV"
    saved: list[Path] = []
    style = {
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.axisbelow": True,
        "savefig.dpi": 180,
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(8, 5.2))
        try:
            for field, color, marker, label in (
                ("asr_response_quality", "#176B87", "o", "ASR transcript → LLM"),
                ("clean_response_quality", "#BD632F", "^", "Human transcript → LLM"),
            ):
                if field in series:
                    points = series[field]
                    ax.scatter(
                        [point[0] for point in points], [point[1] for point in points],
                        s=38, alpha=0.7, color=color, marker=marker,
                        label=f"{label} (n={len(points)})", edgecolors="none",
                    )
            ax.set(title="ASR errors and response quality", xlabel="Word error rate (ratio)",
                   ylabel=f"{quality_label} (0–1)", ylim=(-0.04, 1.04))
            # Small left margin keeps zero-WER markers visible. WER is not capped at 1.
            largest_wer = max(point[0] for points in series.values() for point in points)
            ax.set_xlim(left=-0.025 * max(1.0, largest_wer))
            ax.grid(alpha=0.2)
            ax.legend(loc="best", frameon=False)
            fig.text(0.02, 0.02, footer, fontsize=8, color="#555555")
            fig.tight_layout(rect=(0, 0.05, 1, 1))
            destination = directory / "wer_vs_quality.png"
            fig.savefig(destination)
            saved.append(destination)
        finally:
            plt.close(fig)

        if stages:
            fig, ax = plt.subplots(figsize=(8, 4.6))
            try:
                left = 0.0
                for index, color, label in (
                    (0, "#176B87", "ASR"), (1, "#BD632F", "LLM"),
                    (2, "#ADB5BD", "Pipeline overhead"),
                ):
                    width = mean(stage[index] for stage in stages)
                    ax.barh("Audio → answer", width, left=left, color=color, label=label, height=0.5)
                    left += width
                ax.text(left, 0, f"  {left:.3f}s", va="center", fontsize=10)
                if clean_times:
                    clean_mean = mean(clean_times)
                    ax.barh("Human transcript → LLM", clean_mean, color="#BD632F", height=0.5)
                    ax.text(clean_mean, 1, f"  {clean_mean:.3f}s", va="center", fontsize=10)
                ax.set(title="Mean inference latency", xlabel="Wall-clock seconds")
                ax.set_xlim(0, max(left, mean(clean_times) if clean_times else 0, 0.001) * 1.28)
                ax.grid(axis="x", alpha=0.2)
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=9)
                fig.text(0.02, 0.02, footer + "\nModel loading excluded; clean LLM call is separate from pipeline latency.",
                         fontsize=8, color="#555555")
                fig.tight_layout(rect=(0, 0.14, 1, 1))
                destination = directory / "latency_breakdown.png"
                fig.savefig(destination)
                saved.append(destination)
            finally:
                plt.close(fig)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/plots"))
    parser.add_argument("--quality-label", help="Display name of the [0, 1] response scorer")
    args = parser.parse_args()
    try:
        paths = create_plots(args.input, args.output_dir, quality_label=args.quality_label)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(1, f"Plotting failed: {exc}\n")
    for path in paths:
        print(path.resolve())
    if len(paths) == 1:
        print("Latency columns are absent; only the quality scatter plot was generated.")


if __name__ == "__main__":
    main()
