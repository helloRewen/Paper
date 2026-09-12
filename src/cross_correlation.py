from __future__ import annotations

from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "results" / "cross_context" / "tables"
SIGNAL_DIR = ROOT / "results" / "cross_context" / "signal_analysis"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

SIGNALS = [
    ("Activity path", "Delta AP: activity path"),
    ("Waiting time", "Delta AP: waiting time"),
    ("Structural deviation", "Delta AP: structural deviation"),
]


def exact_spearman_p(x: np.ndarray, y: np.ndarray) -> float:
    observed = float(spearmanr(x, y).statistic)
    values = [float(spearmanr(x, candidate).statistic) for candidate in permutations(y.tolist())]
    return float(np.mean(np.abs(values) >= abs(observed) - 1e-12))


def stars(p_value: float) -> str:
    if p_value <= 0.01:
        return "***"
    if p_value <= 0.05:
        return "**"
    if p_value <= 0.10:
        return "*"
    return ""


def dataset_statistics(table: pd.DataFrame, variants: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, subset in table.groupby("Dataset", sort=False):
        ratio = float(variants.loc[variants["Event log"] == dataset, "variant_case_ratio"].iloc[0])
        for signal, column in SIGNALS:
            rows.append(
                {
                    "Dataset": dataset,
                    "Variant-to-case ratio": ratio,
                    "Log signal": signal,
                    "Mean marginal contribution": float(subset[column].mean()),
                    "Dominant-contribution count": int((subset["Dominant signal"] == signal).sum()),
                }
            )
    return pd.DataFrame(rows)


def correlation_table(statistics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specifications = [
        ("Mean marginal contribution", "Mean marginal contribution"),
        ("Dominant-contribution count", "Dominant-contribution count"),
    ]
    for specification, column in specifications:
        for signal, _source in SIGNALS:
            subset = statistics[statistics["Log signal"] == signal]
            x = subset["Variant-to-case ratio"].to_numpy(dtype=float)
            y = subset[column].to_numpy(dtype=float)
            pearson = float(pearsonr(x, y).statistic)
            spearman = float(spearmanr(x, y).statistic)
            p_value = exact_spearman_p(x, y)
            rows.append(
                {
                    "Analytical specification": specification,
                    "Log signal": signal,
                    "Pearson r": pearson,
                    "Spearman rho": spearman,
                    "Exact permutation p": p_value,
                    "Significance": stars(p_value),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    table = pd.read_csv(TABLE_DIR / "table3_cross_scene_stage_signal_marginal_contribution.csv")
    variants = pd.read_csv(SIGNAL_DIR / "variant_case_ratio.csv")
    statistics = dataset_statistics(table, variants)
    correlations = correlation_table(statistics)
    statistics.to_csv(TABLE_DIR / "table_07_dataset_signal_statistics.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(TABLE_DIR / "table_07_correlation_analysis.csv", index=False, encoding="utf-8-sig")
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
