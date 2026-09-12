from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "paper_tables" / "table_02_operational_actionability.csv"
PREFIX_METRICS = ROOT / "results" / "main" / "tpop_prefix_10_90" / "baseline_models_with_tpop_seed42.csv"
CHECKPOINT_METRICS = ROOT / "results" / "main" / "tpop_operational_checkpoints" / "operational_ap_best_with_tpop.csv"


def run(name: str, *arguments: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "src" / name), *arguments], cwd=ROOT, check=True)


def prepare(recompute: bool) -> None:
    if recompute or not PREFIX_METRICS.exists():
        run("main_baseline.py")
        run("tpop_model.py", "--seeds", "42")
    if recompute or not CHECKPOINT_METRICS.exists():
        run("main_checkpoints.py")
        run("main_tpop.py")


def values(frame: pd.DataFrame, column: str) -> str:
    return "/".join(f"{value:.3f}" for value in frame[column])


def main() -> None:
    prepare("--recompute" in sys.argv)
    prefix = pd.read_csv(PREFIX_METRICS)
    prefix = prefix[~prefix["model"].str.contains("N-gram", case=False, na=False)]
    prefix = prefix[prefix["prefix_ratio"].between(0.2, 0.6)]
    prefix = prefix.loc[prefix.groupby("prefix_ratio")["ap"].idxmax()].sort_values("prefix_ratio")
    checkpoints = pd.read_csv(CHECKPOINT_METRICS)
    fixed_order = {"First 3 events": 3, "First 5 events": 5, "First 7 events": 7}
    natural_order = {"24 h": 24, "72 h": 72, "168 h": 168}
    fixed = checkpoints[checkpoints["checkpoint_type"].eq("fixed_event")].copy()
    fixed["order"] = fixed["checkpoint_label"].map(fixed_order)
    fixed = fixed.sort_values("order")
    natural = checkpoints[checkpoints["checkpoint_type"].eq("natural_time")].copy()
    natural["order"] = natural["checkpoint_label"].map(natural_order)
    natural = natural.sort_values("order")
    table = pd.DataFrame(
        [
            {
                "Check method": "Proportional prefix",
                "Checkpoint": "20%-60%",
                "F1": f"{prefix['f1'].min():.3f}-{prefix['f1'].max():.3f}",
                "AP": f"{prefix['ap'].min():.3f}-{prefix['ap'].max():.3f}",
                "Brier Score": f"{prefix['brier'].max():.3f}-{prefix['brier'].min():.3f}",
                "Net Benefit": "-",
            },
            {
                "Check method": "Fixed-event checkpoint",
                "Checkpoint": "3/5/7",
                "F1": values(fixed, "f1"),
                "AP": values(fixed, "ap"),
                "Brier Score": values(fixed, "brier"),
                "Net Benefit": values(fixed, "net_benefit_05"),
            },
            {
                "Check method": "Natural-time checkpoint",
                "Checkpoint": "24/72/168h",
                "F1": values(natural, "f1"),
                "AP": values(natural, "ap"),
                "Brier Score": values(natural, "brier"),
                "Net Benefit": values(natural, "net_benefit_05"),
            },
        ]
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
