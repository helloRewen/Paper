from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "main" / "checkpoint_stage_h23_h4_tables" / "h23_checkpoint_config_metrics.csv"
OUTPUT = ROOT / "results" / "paper_tables" / "table_03_feature_ablation_performance.csv"


def prepare(recompute: bool) -> None:
    if recompute or not SOURCE.exists():
        for name in ["main_baseline.py", "main_checkpoints.py", "main_signal_ablation.py", "main_hypothesis_tables.py"]:
            subprocess.run([sys.executable, str(ROOT / "src" / name)], cwd=ROOT, check=True)


def main() -> None:
    prepare("--recompute" in sys.argv)
    data = pd.read_csv(SOURCE)
    grouped = data.groupby(["Check method", "Configuration", "Stage"], as_index=False)[["ap", "f1", "brier"]].mean()
    parts = []
    for metric, label in [("ap", "AP"), ("f1", "F1"), ("brier", "Brier Score")]:
        part = grouped.pivot(index=["Check method", "Configuration"], columns="Stage", values=metric)
        part = part.reindex(columns=["Early", "Middle", "Late"])
        part.columns = [f"{stage} {label}" for stage in part.columns]
        parts.append(part)
    table = pd.concat(parts, axis=1).reset_index()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
