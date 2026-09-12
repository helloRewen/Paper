from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_fscore_support

from cross_pgv import (
    DATASET_LABEL,
    FIXED_EVENTS,
    NATURAL_HOURS,
    RATIOS,
    build_external_fixed_table,
    build_external_natural_table,
    build_external_ratio_table,
)
from external_validation import (
    LOADERS,
    build_labels,
    deterministic_temporal_sample,
    infer_main_path,
)
from tpop_model import (
    build_bert_activity_embeddings,
    fit_one_seed,
    select_features_with_xgboost_shap,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_METRICS = (
    ROOT
    / "results"
    / "cross_context"
    / "appendix_pgv"
    / "appendix_a2_a3_all_model_metrics.csv"
)
RESULTS_DIR = ROOT / "results" / "cross_context" / "tpop"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "small_business_credit_approval",
    "bpi2017_offer_log",
    "bpi2014_rabobank_ict",
    "bpi2020_request_for_payment",
]
SEED = 42
NB_THRESHOLD = 0.5


def checkpoint_display(check_type: str, raw_value: float | int) -> str:
    if check_type == "Proportional prefix":
        return f"{int(round(float(raw_value) * 100))}%"
    if check_type == "Fixed-event checkpoint":
        return f"Event {int(raw_value)}"
    return f"{int(raw_value)}h"


def top_k_recall(y_true: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return 0.0
    count = max(1, int(math.ceil(len(y_true) * fraction)))
    selected = np.argsort(-scores, kind="mergesort")[:count]
    return float(np.sum(y_true[selected] == 1) / positives)


def net_benefit(y_true: np.ndarray, scores: np.ndarray, threshold: float = NB_THRESHOLD) -> float:
    predicted = scores >= threshold
    tp = float(np.sum((y_true == 1) & predicted))
    fp = float(np.sum((y_true == 0) & predicted))
    n = float(len(y_true))
    return tp / n - fp / n * threshold / (1.0 - threshold)


def add_tokens(table: pd.DataFrame, case_map: dict[str, tuple[list[str], np.ndarray]]) -> pd.DataFrame:
    out = table.copy()
    out["caseid"] = out["caseid"].astype(str)
    out["tokens_raw"] = [
        case_map[caseid][0][: int(prefix_length)]
        for caseid, prefix_length in zip(out["caseid"], out["prefix_event_count"])
    ]
    return out


def add_governance_metrics(metrics: dict[str, object], predictions: pd.DataFrame) -> None:
    y_true = predictions["label"].to_numpy(dtype=int)
    scores = predictions["score"].to_numpy(dtype=float)
    metrics["top5_recall"] = top_k_recall(y_true, scores, 0.05)
    metrics["top10_recall"] = top_k_recall(y_true, scores, 0.10)
    metrics["top20_recall"] = top_k_recall(y_true, scores, 0.20)
    metrics["net_benefit"] = net_benefit(y_true, scores)
    metrics["positive_cases"] = int(y_true.sum())
    metrics["positive_rate"] = float(y_true.mean())


def build_table(
    check_type: str,
    checkpoint_raw: float | int,
    traces,
    labels: pd.DataFrame,
    mandatory_flow: list[str],
    explicit_states: set[str],
) -> pd.DataFrame:
    if check_type == "Proportional prefix":
        return build_external_ratio_table(
            traces, labels, float(checkpoint_raw), mandatory_flow, explicit_states
        )
    if check_type == "Fixed-event checkpoint":
        return build_external_fixed_table(
            traces, labels, mandatory_flow, explicit_states, int(checkpoint_raw)
        )
    return build_external_natural_table(
        traces, labels, mandatory_flow, explicit_states, int(checkpoint_raw)
    )


def validate_checkpoint(
    metrics: dict[str, object],
    predictions: pd.DataFrame,
    existing_metrics: pd.DataFrame,
) -> None:
    if predictions["caseid"].duplicated().any():
        raise RuntimeError("Duplicate case IDs detected in TPOP test predictions")
    if predictions[["label", "score"]].isna().any().any():
        raise RuntimeError("Missing labels or scores detected in TPOP predictions")
    if not predictions["score"].between(0.0, 1.0).all():
        raise RuntimeError("TPOP scores fall outside [0, 1]")

    current = existing_metrics[
        existing_metrics["dataset_key"].eq(metrics["dataset_key"])
        & existing_metrics["check_type"].eq(metrics["check_type"])
        & np.isclose(
            existing_metrics["checkpoint_raw"].astype(float),
            float(metrics["checkpoint_raw"]),
        )
    ]
    expected_cases = current["test_cases"].astype(int).unique()
    expected_positives = current["positive_cases"].astype(int).unique()
    if len(expected_cases) != 1 or expected_cases[0] != len(predictions):
        raise RuntimeError(
            f"Test-case count mismatch: {len(predictions)} versus {expected_cases.tolist()}"
        )
    if len(expected_positives) != 1 or expected_positives[0] != int(predictions["label"].sum()):
        raise RuntimeError(
            "Positive-case count mismatch: "
            f"{int(predictions['label'].sum())} versus {expected_positives.tolist()}"
        )

    y_true = predictions["label"].to_numpy(dtype=int)
    scores = predictions["score"].to_numpy(dtype=float)
    pred = (scores >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    checks = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap": average_precision_score(y_true, scores),
        "brier": brier_score_loss(y_true, scores),
    }
    for name, recomputed in checks.items():
        if not np.isclose(float(metrics[name]), float(recomputed), atol=1e-10):
            raise RuntimeError(f"Metric mismatch for {name}: {metrics[name]} versus {recomputed}")


def select_best(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.sort_values(
            ["dataset_key", "check_type", "checkpoint_raw", "ap", "brier", "top20_recall"],
            ascending=[True, True, True, False, True, False],
        )
        .groupby(["dataset_key", "check_type", "checkpoint_raw"], as_index=False)
        .head(1)
        .copy()
    )


def build_pgv_summary(best: pd.DataFrame) -> pd.DataFrame:
    return (
        best.groupby(["dataset_key", "dataset", "check_type"], as_index=False)
        .agg(
            checkpoints=("checkpoint", lambda values: "/".join(values.astype(str))),
            models=("model", lambda values: "/".join(values.astype(str))),
            ap=("ap", "mean"),
            top5_recall=("top5_recall", "mean"),
            top10_recall=("top10_recall", "mean"),
            top20_recall=("top20_recall", "mean"),
            brier=("brier", "mean"),
            net_benefit=("net_benefit", "mean"),
        )
        .sort_values(["dataset_key", "check_type"])
        .reset_index(drop=True)
    )


def main() -> None:
    existing_all = pd.read_csv(SOURCE_METRICS)
    existing = existing_all[existing_all["dataset_key"].isin(DATASETS)].copy()

    metric_rows: list[dict[str, object]] = []

    checkpoint_plan = (
        [("Proportional prefix", value) for value in RATIOS]
        + [("Fixed-event checkpoint", value) for value in FIXED_EVENTS]
        + [("Natural-time checkpoint", value) for value in NATURAL_HOURS]
    )

    for dataset_key in DATASETS:
        print(f"Loading {dataset_key}", flush=True)
        traces = deterministic_temporal_sample(LOADERS[dataset_key](), dataset_key)
        labels, explicit_states, label_note = build_labels(dataset_key, traces)
        mandatory_flow = infer_main_path(traces, labels)
        case_map = {
            str(trace.caseid): (
                list(trace.activities),
                np.asarray(trace.times, dtype="datetime64[ns]"),
            )
            for trace in traces
        }
        activities = sorted({activity for trace in traces for activity in trace.activities})
        activity_to_id, activity_embeddings = build_bert_activity_embeddings(activities)

        for check_type, checkpoint_raw in checkpoint_plan:
            table = build_table(
                check_type,
                checkpoint_raw,
                traces,
                labels,
                mandatory_flow,
                explicit_states,
            )
            table = add_tokens(table, case_map)
            cutoff = table["start_time"].quantile(0.7)
            train_all = table[table["start_time"] <= cutoff].sort_values("start_time")
            validation_size = max(512, int(len(train_all) * 0.15))
            feature_train = train_all.iloc[:-validation_size].copy()
            if feature_train["anomaly_label"].nunique() < 2:
                raise RuntimeError(
                    f"Insufficient label variation for {dataset_key}/{check_type}/{checkpoint_raw}"
                )
            selected_features, _feature_ranking = select_features_with_xgboost_shap(feature_train)

            metrics, predictions, _history = fit_one_seed(
                prefix_df=table,
                ratio=float(checkpoint_raw),
                cutoff=cutoff,
                case_map=case_map,
                activity_to_id=activity_to_id,
                activity_embeddings=activity_embeddings,
                selected_features=selected_features,
                seed=SEED,
                max_pretrain_epochs=10,
                max_gsat_epochs=12,
            )
            checkpoint = checkpoint_display(check_type, checkpoint_raw)
            metrics.update(
                {
                    "dataset_key": dataset_key,
                    "dataset": DATASET_LABEL[dataset_key],
                    "check_type": check_type,
                    "checkpoint_raw": checkpoint_raw,
                    "checkpoint": checkpoint,
                    "model": "TPOP-adapted",
                }
            )
            add_governance_metrics(metrics, predictions)
            validate_checkpoint(metrics, predictions, existing)
            metric_rows.append(metrics)
            print(
                f"  {check_type}/{checkpoint}: F1={metrics['f1']:.4f}, "
                f"AP={metrics['ap']:.4f}, Brier={metrics['brier']:.4f}, "
                f"NB={metrics['net_benefit']:.4f}",
                flush=True,
            )
            del table, train_all, feature_train, predictions
            gc.collect()

    tpop_metrics = pd.DataFrame(metric_rows)
    tpop_metrics.to_csv(RESULTS_DIR / "tpop_cross_scene_metrics.csv", index=False, encoding="utf-8-sig")

    removed_models = {"sequence_ngram_lr", "Sequence N-gram LR"}
    candidate_existing = existing[~existing["model"].isin(removed_models)].copy()
    combined = pd.concat([candidate_existing, tpop_metrics], ignore_index=True, sort=False)
    combined.to_csv(
        RESULTS_DIR / "cross_scene_models_without_ngram_with_tpop.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best = select_best(combined)
    best.to_csv(
        RESULTS_DIR / "cross_scene_ap_best_with_tpop.csv", index=False, encoding="utf-8-sig"
    )
    pgv = build_pgv_summary(best)
    pgv.to_csv(
        RESULTS_DIR / "cross_scene_pgv_summary_with_tpop.csv", index=False, encoding="utf-8-sig"
    )

    print("\nTPOP mean performance by dataset/check type", flush=True)
    print(
        tpop_metrics.groupby(["dataset", "check_type"])[["f1", "ap", "brier", "net_benefit"]]
        .mean()
        .to_string(),
        flush=True,
    )
    print("\nAP-best model counts after replacing N-gram", flush=True)
    print(best["model"].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
