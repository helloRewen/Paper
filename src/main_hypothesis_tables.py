from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_fscore_support

from main_checkpoints import build_fixed_event_table, build_natural_time_table
from main_signal_ablation import SignalEncoder
from main_baseline import SEED, build_full_labels, build_prefix_table, load_event_log


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "main" / "checkpoint_stage_h23_h4_tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RATIO_STAGES = {
    "Early": [0.1, 0.2, 0.3],
    "Middle": [0.4, 0.5, 0.6],
    "Late": [0.7, 0.8, 0.9],
}

                                                      
FIXED_STAGES: dict[str, list[int]] = {}
NATURAL_STAGES: dict[str, list[int]] = {}

SPECS = {
    "full": ["path", "time", "structure"],
    "no_path": ["time", "structure"],
    "no_time": ["path", "structure"],
    "no_structure": ["path", "time"],
    "path_only": ["path"],
    "time_only": ["time"],
    "structure_only": ["structure"],
}

TARGETS = [
    ("rework", "Rework deviation"),
    ("rejected", "Rejection deviation"),
    ("cancelled", "Cancellation deviation"),
    ("duration_anomaly", "Duration deviation"),
]


def eval_scores(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_pred = (scores >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    ap = average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else float(np.mean(y_true))
    brier = brier_score_loss(y_true, scores)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "ap": float(ap), "brier": float(brier)}


def fit_scores(train_df: pd.DataFrame, test_df: pd.DataFrame, groups: list[str], target_col: str) -> dict[str, float]:
    y_train = train_df[target_col].to_numpy(dtype=int)
    y_test = test_df[target_col].to_numpy(dtype=int)
    if len(np.unique(y_train)) < 2:
        scores = np.repeat(float(np.mean(y_train)), len(y_test))
        return eval_scores(y_test, scores)
    encoder = SignalEncoder().fit(train_df)
    x_train = encoder.transform(train_df, groups)
    x_test = encoder.transform(test_df, groups)
    clf = LogisticRegression(max_iter=1600, class_weight="balanced", solver="liblinear", random_state=SEED)
    clf.fit(x_train, y_train)
    scores = clf.predict_proba(x_test)[:, 1]
    return eval_scores(y_test, scores)


def checkpoint_table(df: pd.DataFrame, labels: pd.DataFrame, method: str, checkpoint):
    if method == "Proportional prefix":
        return build_prefix_table(df, labels, float(checkpoint))
    if method == "Fixed event":
        return build_fixed_event_table(df, labels, int(checkpoint))
    if method == "Natural time":
        return build_natural_time_table(df, labels, int(checkpoint))
    raise ValueError(method)


def checkpoint_label(method: str, stage: str, checkpoints: Iterable) -> str:
    values = list(checkpoints)
    if method == "Proportional prefix":
        return " / ".join(f"{int(v * 100)}%" for v in values)
    if method == "Fixed event":
        return " / ".join(f"Event {int(v)}" for v in values)
    if method == "Natural time":
        return " / ".join(f"{int(v)} h" for v in values)
    return stage


def derive_operational_stages(labels: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[dict[str, list[int]], dict[str, list[int]], dict]:
    train = labels[labels["start_time"] <= cutoff].copy()
    stage_centers = {"Early": 0.2, "Middle": 0.5, "Late": 0.8}
    fixed: dict[str, list[int]] = {}
    natural: dict[str, list[int]] = {}
    evidence = {}
    for stage, ratio in stage_centers.items():
        event_positions = []
        for n in train["full_length"].to_numpy():
            n = int(n)
            event_positions.append(max(1, min(int(np.ceil(n * ratio)), n - 1 if n > 1 else 1)))
        event_series = pd.Series(event_positions)
        event_checkpoint = int(round(float(event_series.median())))
        duration_quantile = float(train["full_duration_hours"].quantile(ratio))
        duration_checkpoint = int(round(duration_quantile))
        fixed[stage] = [event_checkpoint]
        natural[stage] = [duration_checkpoint]
        evidence[stage] = {
            "stage_center_ratio": ratio,
            "event_median": float(event_series.median()),
            "event_q25": float(event_series.quantile(0.25)),
            "event_q75": float(event_series.quantile(0.75)),
            "duration_hour_quantile": duration_quantile,
            "duration_hour_checkpoint": duration_checkpoint,
        }
    return fixed, natural, evidence


def main() -> None:
    df = load_event_log()
    labels = build_full_labels(df)
    cutoff = labels["start_time"].quantile(0.7)
    fixed_stages, natural_stages, stage_evidence = derive_operational_stages(labels, cutoff)

    methods = [
        ("Proportional prefix", RATIO_STAGES),
        ("Fixed event", fixed_stages),
        ("Natural time", natural_stages),
    ]

    h23_checkpoint_rows = []
    h23_stage_rows = []
    h4_checkpoint_rows = []

    for method, stage_map in methods:
        for stage, checkpoints in stage_map.items():
            h23_subrows = []
            h4_subrows = []
            for checkpoint in checkpoints:
                prefix_df = checkpoint_table(df, labels, method, checkpoint)
                train_df = prefix_df[prefix_df["start_time"] <= cutoff].copy()
                test_df = prefix_df[prefix_df["start_time"] > cutoff].copy()

                spec_metrics = {}
                for spec, groups in SPECS.items():
                    metrics = fit_scores(train_df, test_df, groups, "anomaly_label")
                    metrics.update(
                        {
                            "Check method": method,
                            "Stage": stage,
                            "Checkpoint": checkpoint_label(method, stage, [checkpoint]),
                            "Configuration": spec,
                            "train_cases": len(train_df),
                            "test_cases": len(test_df),
                        }
                    )
                    spec_metrics[spec] = metrics
                    h23_checkpoint_rows.append(metrics)
                full = spec_metrics["full"]
                h23_subrows.append(
                    {
                        "Delta AP: activity path": full["ap"] - spec_metrics["no_path"]["ap"],
                        "Delta AP: waiting time": full["ap"] - spec_metrics["no_time"]["ap"],
                        "Delta AP: structural deviation": full["ap"] - spec_metrics["no_structure"]["ap"],
                    }
                )

                for target_col, target_name in TARGETS:
                    metrics = fit_scores(train_df, test_df, SPECS["full"], target_col)
                    metrics.update(
                        {
                            "Check method": method,
                            "Stage": stage,
                            "Checkpoint": checkpoint_label(method, stage, [checkpoint]),
                            "Deviation type": target_name,
                            "positive_cases": int(test_df[target_col].sum()),
                            "positive_rate": float(test_df[target_col].mean()),
                            "train_cases": len(train_df),
                            "test_cases": len(test_df),
                        }
                    )
                    h4_subrows.append(metrics)
                    h4_checkpoint_rows.append(metrics)

            h23_stage = pd.DataFrame(h23_subrows).mean(numeric_only=True).to_dict()
            deltas = {
                "Activity path": h23_stage["Delta AP: activity path"],
                "Waiting time": h23_stage["Delta AP: waiting time"],
                "Structural deviation": h23_stage["Delta AP: structural deviation"],
            }
            max_delta = max(deltas.values())
            main_signal = "/".join([name for name, value in deltas.items() if np.isclose(value, max_delta)])
            if stage == "Early":
                hypothesis = "H2 partially supported" if main_signal != "Activity path" else "Supports H2"
            else:
                hypothesis = "Supports H3" if ("Waiting time" in main_signal or "Structural deviation" in main_signal) else "H3 partially supported"
            h23_stage_rows.append(
                {
                    "Check method": method,
                    "Stage": stage,
                    "Checkpoint": checkpoint_label(method, stage, checkpoints),
                    "Delta AP: activity path": h23_stage["Delta AP: activity path"],
                    "Delta AP: waiting time": h23_stage["Delta AP: waiting time"],
                    "Delta AP: structural deviation": h23_stage["Delta AP: structural deviation"],
                    "Dominant contributing signal": main_signal,
                    "Hypothesis assessment": hypothesis,
                }
            )

    h4_checkpoint_df = pd.DataFrame(h4_checkpoint_rows)
    h4_stage_rows = []
    for (method, target), sub in h4_checkpoint_df.groupby(["Check method", "Deviation type"], sort=False):
        stage_df = sub.groupby("Stage", as_index=False).agg(ap=("ap", "mean"), recall=("recall", "mean"), brier=("brier", "mean"))
        stage_map = stage_df.set_index("Stage")
        h4_stage_rows.append(
            {
                "Check method": method,
                "Deviation type": target,
                "Early AP": float(stage_map.loc["Early", "ap"]),
                "Middle AP": float(stage_map.loc["Middle", "ap"]),
                "Late AP": float(stage_map.loc["Late", "ap"]),
                "Early Recall": float(stage_map.loc["Early", "recall"]),
                "Middle Recall": float(stage_map.loc["Middle", "recall"]),
                "Late Recall": float(stage_map.loc["Late", "recall"]),
                "Mean Brier Score": float(sub["brier"].mean()),
                "AP range": float(stage_df["ap"].max() - stage_df["ap"].min()),
            }
        )

    h23_checkpoint_df = pd.DataFrame(h23_checkpoint_rows)
    h23_stage_df = pd.DataFrame(h23_stage_rows)
    h4_stage_df = pd.DataFrame(h4_stage_rows)

    h23_checkpoint_df.to_csv(RESULTS_DIR / "h23_checkpoint_config_metrics.csv", index=False, encoding="utf-8-sig")
    h23_stage_df.to_csv(RESULTS_DIR / "h23_stage_comparison_table.csv", index=False, encoding="utf-8-sig")
    h4_checkpoint_df.to_csv(RESULTS_DIR / "h4_checkpoint_type_metrics.csv", index=False, encoding="utf-8-sig")
    h4_stage_df.to_csv(RESULTS_DIR / "h4_stage_comparison_table.csv", index=False, encoding="utf-8-sig")
    pd.Series(stage_evidence).to_json(RESULTS_DIR / "stage_derivation_evidence.json", force_ascii=False, indent=2)

    print("H23")
    print(h23_stage_df.to_string(index=False))
    print("H4")
    print(h4_stage_df.to_string(index=False))


if __name__ == "__main__":
    main()
