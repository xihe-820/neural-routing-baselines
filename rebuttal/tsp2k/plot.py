"""Export measured fullset curves; include all actual points, including >30s."""
import argparse
import json
from pathlib import Path

from rebuttal.tsp2k.aggregate import aggregate


def checked_report(path):
    report = json.loads(Path(path).read_text())
    if (report.get("schema") != "tsp2k-curves.v1" or
            report.get("evidence_kind") != "SERVER_MEASUREMENT" or
            report.get("status") != "MEASURED_FULLSET"):
        raise ValueError("only verified measured fullset artifacts may be plotted")
    actual = aggregate([p["artifact_path"] for p in report["points"]], require_fullset=True)
    if actual != report:
        raise ValueError("curve report differs from hash-verified measurement artifacts")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    report = checked_report(args.report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    for method in ("GLOP", "UDC", "SIL"):
        points = sorted([p for p in report["points"] if p["method"] == method],
                        key=lambda p: p["mean_time_seconds"])
        axis.plot([p["mean_time_seconds"] for p in points],
                  [p["mean_drop_percent"] for p in points], "o-", label=method)
    axis.set(xlabel="Time (s)", ylabel="Drop (%)", xlim=(0, None))
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        target = args.output_prefix.with_suffix("." + ext)
        if target.exists():
            raise ValueError(f"plot already exists: {target}")
        fig.savefig(target, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
