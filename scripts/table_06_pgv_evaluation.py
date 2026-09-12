from pathlib import Path
import math
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREFIX_METRICS = ROOT / "results" / "main" / "tpop_prefix_10_90" / "baseline_models_with_tpop_seed42.csv"
PREFIX_BASE_PREDICTIONS = ROOT / "results" / "main" / "prefix_10_90_baseline" / "prefix_10_90_baseline_predictions.csv"
PREFIX_TPOP_PREDICTIONS = ROOT / "results" / "main" / "tpop_prefix_10_90" / "tpop_predictions.csv"
CHECKPOINT_SUMMARY = ROOT / "results" / "main" / "tpop_operational_checkpoints" / "operational_pgv_summary_with_tpop.csv"
OUTPUT = ROOT / "results" / "paper_tables" / "table_06_pgv_evaluation.csv"


def run(name: str, *arguments: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "src" / name), *arguments], cwd=ROOT, check=True)


def prepare(recompute: bool) -> None:
    prefix_sources = [PREFIX_METRICS, PREFIX_BASE_PREDICTIONS, PREFIX_TPOP_PREDICTIONS]
    if recompute or not all(path.exists() for path in prefix_sources):
        run("main_baseline.py")
        run("tpop_model.py", "--seeds", "42")
    if recompute or not CHECKPOINT_SUMMARY.exists():
        run("main_checkpoints.py")
        run("main_tpop.py")


def top_k_recall(labels: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    count = max(1, int(math.ceil(len(labels) * fraction)))
    selected = np.argsort(-scores, kind="mergesort")[:count]
    positives = int(np.sum(labels == 1))
    return float(np.sum(labels[selected] == 1) / positives) if positives else 0.0


def net_benefit(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    predicted = scores >= threshold
    tp = float(np.sum((labels == 1) & predicted))
    fp = float(np.sum((labels == 0) & predicted))
    return tp / len(labels) - fp / len(labels) * threshold / (1.0 - threshold)


def proportional_row() -> dict[str, object]:
    metrics = pd.read_csv(PREFIX_METRICS)
    metrics = metrics[~metrics["model"].str.contains("N-gram", case=False, na=False)]
    best = metrics.loc[metrics.groupby("prefix_ratio")["ap"].idxmax()].sort_values("prefix_ratio")
    baseline = pd.read_csv(PREFIX_BASE_PREDICTIONS)
    tpop = pd.read_csv(PREFIX_TPOP_PREDICTIONS)
    predictions = pd.concat([baseline, tpop], ignore_index=True, sort=False)
    rows = []
    for row in best.itertuples(index=False):
        selected = predictions[
            np.isclose(predictions["prefix_ratio"].astype(float), float(row.prefix_ratio))
            & predictions["model"].eq(row.model)
        ]
        labels = selected["label"].to_numpy(dtype=int)
        scores = selected["score"].to_numpy(dtype=float)
        rows.append(
            {
                "Top-5% Recall": top_k_recall(labels, scores, 0.05),
                "Top-10% Recall": top_k_recall(labels, scores, 0.10),
                "Top-20% Recall": top_k_recall(labels, scores, 0.20),
                "Net Benefit": net_benefit(labels, scores),
            }
        )
    governance = pd.DataFrame(rows).mean()
    return {
        "Check method": "Proportional prefix",
        "Checkpoint": "Mean across 10%-90%",
        "Models": "/".join(dict.fromkeys(best["model"].tolist())),
        "AP": best["ap"].mean(),
        "Top-5% Recall": governance["Top-5% Recall"],
        "Top-10% Recall": governance["Top-10% Recall"],
        "Top-20% Recall": governance["Top-20% Recall"],
        "Brier Score": best["brier"].mean(),
        "Net Benefit": governance["Net Benefit"],
    }


def main() -> None:
    prepare("--recompute" in sys.argv)
    rows = [proportional_row()]
    checkpoints = pd.read_csv(CHECKPOINT_SUMMARY)
    method_names = {"fixed_event": "Fixed event", "natural_time": "Natural time"}
    for row in checkpoints.itertuples(index=False):
        rows.append(
            {
                "Check method": method_names[row.checkpoint_type],
                "Checkpoint": row.checkpoints,
                "Models": row.models,
                "AP": row.ap,
                "Top-5% Recall": row.top_05_recall,
                "Top-10% Recall": row.top_10_recall,
                "Top-20% Recall": row.top_20_recall,
                "Brier Score": row.brier,
                "Net Benefit": row.net_benefit_05,
            }
        )
    table = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
