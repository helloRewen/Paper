from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from tpop_model import (
    build_bert_activity_embeddings,
    build_case_event_map,
    fit_one_seed,
    parse_tokens,
    select_features_with_xgboost_shap,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "results" / "main" / "h1_operational_checkpoints_six_models"
RESULTS_DIR = ROOT / "results" / "main" / "tpop_operational_checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

CHECKPOINT_FILES = [
    ("fixed_event", "First 3 events", SOURCE_DIR / "fixed_event_3_features.csv"),
    ("fixed_event", "First 5 events", SOURCE_DIR / "fixed_event_5_features.csv"),
    ("fixed_event", "First 7 events", SOURCE_DIR / "fixed_event_7_features.csv"),
    ("natural_time", "24 h", SOURCE_DIR / "natural_time_24h_features.csv"),
    ("natural_time", "72 h", SOURCE_DIR / "natural_time_72h_features.csv"),
    ("natural_time", "168 h", SOURCE_DIR / "natural_time_168h_features.csv"),
]


def net_benefit(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    predicted = scores >= threshold
    tp = float(np.sum((y_true == 1) & predicted))
    fp = float(np.sum((y_true == 0) & predicted))
    n = float(len(y_true))
    return tp / n - fp / n * threshold / (1.0 - threshold)


def top_k_recall(y_true: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    review_count = max(1, int(math.ceil(len(y_true) * fraction)))
    selected = np.argsort(-scores, kind="mergesort")[:review_count]
    positives = int(np.sum(y_true == 1))
    return float(np.sum(y_true[selected] == 1) / positives) if positives else 0.0


def collect_checkpoint_activities() -> list[str]:
    activities: set[str] = set()
    for _, _, path in CHECKPOINT_FILES:
        values = pd.read_csv(path, usecols=["tokens_raw"])["tokens_raw"]
        for value in values:
            activities.update(parse_tokens(value))
    return sorted(activities)


def add_governance_metrics(metrics: dict[str, object], predictions: pd.DataFrame) -> None:
    y_true = predictions["label"].to_numpy(dtype=int)
    scores = predictions["score"].to_numpy(dtype=float)
    metrics["top_05_recall"] = top_k_recall(y_true, scores, 0.05)
    metrics["top_10_recall"] = top_k_recall(y_true, scores, 0.10)
    metrics["top_20_recall"] = top_k_recall(y_true, scores, 0.20)
    metrics["net_benefit_03"] = net_benefit(y_true, scores, 0.3)
    metrics["net_benefit_05"] = net_benefit(y_true, scores, 0.5)


def select_ap_best_predictions(
    combined_metrics: pd.DataFrame,
    old_predictions: pd.DataFrame,
    tpop_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_rows = (
        combined_metrics.sort_values(
            ["checkpoint_type", "checkpoint_label", "ap"], ascending=[True, True, False]
        )
        .groupby(["checkpoint_type", "checkpoint_label"], sort=False)
        .head(1)
        .copy()
    )
    selected_predictions: list[pd.DataFrame] = []
    governance_rows: list[dict[str, object]] = []
    for row in best_rows.itertuples(index=False):
        source = tpop_predictions if row.model == "TPOP-adapted" else old_predictions
        chosen = source[
            source["checkpoint_type"].eq(row.checkpoint_type)
            & source["checkpoint_label"].eq(row.checkpoint_label)
            & source["model"].eq(row.model)
        ].copy()
        if len(chosen) != int(row.test_cases):
            raise ValueError(
                f"Prediction count mismatch for {row.checkpoint_type}/{row.checkpoint_label}/{row.model}"
            )
        selected_predictions.append(chosen)
        metrics = row._asdict()
        add_governance_metrics(metrics, chosen)
        governance_rows.append(metrics)
    return pd.DataFrame(governance_rows), pd.concat(selected_predictions, ignore_index=True)


def main() -> None:
    labels = pd.read_csv(ROOT / "results" / "main" / "prefix_10_90_baseline" / "full_labels.csv", parse_dates=["start_time"])
    cutoff = labels["start_time"].quantile(0.7)
    case_map = build_case_event_map()
    activities = collect_checkpoint_activities()
    activity_to_id, activity_embeddings = build_bert_activity_embeddings(activities)

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for checkpoint_type, checkpoint_label, path in CHECKPOINT_FILES:
        prefix_df = pd.read_csv(
            path,
            converters={"tokens_raw": parse_tokens, "start_time": pd.to_datetime},
        )
        train_all = prefix_df[prefix_df["start_time"] <= cutoff].copy().sort_values("start_time")
        validation_size = max(512, int(len(train_all) * 0.15))
        feature_train = train_all.iloc[:-validation_size].copy()
        selected_features, _feature_ranking = select_features_with_xgboost_shap(feature_train)

        metrics, predictions, _history = fit_one_seed(
            prefix_df=prefix_df,
            ratio=float(prefix_df["checkpoint"].iloc[0]),
            cutoff=cutoff,
            case_map=case_map,
            activity_to_id=activity_to_id,
            activity_embeddings=activity_embeddings,
            selected_features=selected_features,
            seed=SEED,
            max_pretrain_epochs=10,
            max_gsat_epochs=12,
        )
        metrics["checkpoint_type"] = checkpoint_type
        metrics["checkpoint_label"] = checkpoint_label
        add_governance_metrics(metrics, predictions)
        predictions["checkpoint_type"] = checkpoint_type
        predictions["checkpoint_label"] = checkpoint_label
        metric_rows.append(metrics)
        prediction_frames.append(predictions)
        print(
            f"{checkpoint_type}/{checkpoint_label}: F1={metrics['f1']:.4f}, "
            f"AP={metrics['ap']:.4f}, Brier={metrics['brier']:.4f}, "
            f"NB(0.5)={metrics['net_benefit_05']:.4f}",
            flush=True,
        )

    tpop_metrics = pd.DataFrame(metric_rows)
    tpop_predictions = pd.concat(prediction_frames, ignore_index=True)
    tpop_metrics.to_csv(RESULTS_DIR / "tpop_operational_metrics.csv", index=False, encoding="utf-8-sig")

    old_metrics = pd.read_csv(SOURCE_DIR / "h1_operational_checkpoint_metrics.csv")
    old_predictions = pd.read_csv(SOURCE_DIR / "h1_operational_checkpoint_predictions.csv")
    candidate_metrics = old_metrics[old_metrics["model"].ne("Sequence N-gram LR")].copy()
    combined_metrics = pd.concat([candidate_metrics, tpop_metrics], ignore_index=True, sort=False)
    combined_metrics.to_csv(
        RESULTS_DIR / "operational_models_without_ngram_with_tpop.csv", index=False, encoding="utf-8-sig"
    )
    best_metrics, _best_predictions = select_ap_best_predictions(
        combined_metrics, old_predictions, tpop_predictions
    )
    best_metrics.to_csv(
        RESULTS_DIR / "operational_ap_best_with_tpop.csv", index=False, encoding="utf-8-sig"
    )

    pgv_summary = (
        best_metrics.groupby("checkpoint_type")
        .agg(
            checkpoints=("checkpoint_label", lambda values: "/".join(values)),
            models=("model", lambda values: "/".join(values)),
            ap=("ap", "mean"),
            top_05_recall=("top_05_recall", "mean"),
            top_10_recall=("top_10_recall", "mean"),
            top_20_recall=("top_20_recall", "mean"),
            brier=("brier", "mean"),
            net_benefit_05=("net_benefit_05", "mean"),
        )
        .reset_index()
    )
    pgv_summary.to_csv(
        RESULTS_DIR / "operational_pgv_summary_with_tpop.csv", index=False, encoding="utf-8-sig"
    )

    print("\nPGV summary\n", pgv_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
