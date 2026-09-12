from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
)
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from external_metrics import (
    build_xgboost_model as build_external_xgboost_model,
    model_scores as external_model_scores,
)
from external_validation import (              
    CATEGORICAL_COLS,
    LOADERS,
    NUMERIC_COLS,
    build_labels,
    build_prefix_table as build_external_ratio_table,
    deterministic_temporal_sample,
    infer_main_path,
)
from main_checkpoints import (              
    build_fixed_event_table as build_main_fixed_event_table,
    build_natural_time_table as build_main_natural_time_table,
    evaluate_classical_models as evaluate_main_checkpoint_models,
)
from main_baseline import (              
    SEED,
    LSTMBaseline,
    PrefixSequenceDataset,
    TransformerBaseline,
    build_full_labels,
    build_vocab,
    build_prefix_table as build_main_ratio_table,
    collate_batch,
    encode_sequences,
    evaluate_classical_models as evaluate_main_ratio_models,
    load_event_log,
    make_examples,
    predict_scores,
    set_seed,
)
from external_checkpoints import (
    build_external_fixed_table,
    build_external_natural_table,
)


OUT_DIR = ROOT / "results" / "cross_context" / "appendix_pgv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RATIOS = [i / 10 for i in range(1, 10)]
FIXED_EVENTS = [3, 5, 7]
NATURAL_HOURS = [24, 72, 168]
NB_THRESHOLD = 0.5
DEEP_MODELS = ["LSTM", "Transformer"]

DATASET_ORDER = [
    "corporate_account_opening",
    "small_business_credit_approval",
    "bpi2017_offer_log",
    "bpi2014_rabobank_ict",
    "bpi2020_request_for_payment",
]

DATASET_LABEL = {
    "corporate_account_opening": "Corporate Account-opening Event Log",
    "small_business_credit_approval": "Small-business Credit Approval",
    "bpi2017_offer_log": "BPI 2017 Offer",
    "bpi2014_rabobank_ict": "BPI 2014 Rabobank ICT",
    "bpi2020_request_for_payment": "BPI 2020 RfP",
}

EXTERNAL_DATASETS = [key for key in DATASET_ORDER if key != "corporate_account_opening"]


def top_k_recall(y_true: np.ndarray, scores: np.ndarray, k: float) -> float:
    positives = int(np.sum(y_true == 1))
    if positives == 0 or len(y_true) == 0:
        return 0.0
    n_top = max(1, int(math.ceil(len(y_true) * k)))
    top_idx = np.argsort(scores)[::-1][:n_top]
    return float(np.sum(y_true[top_idx] == 1) / positives)


def net_benefit(y_true: np.ndarray, scores: np.ndarray, threshold: float = NB_THRESHOLD) -> float:
    threshold = min(max(float(threshold), 1e-6), 1 - 1e-6)
    y_pred = scores >= threshold
    n = float(len(y_true))
    tp = float(np.sum((y_pred == 1) & (y_true == 1))) / n
    fp = float(np.sum((y_pred == 1) & (y_true == 0))) / n
    return float(tp - fp * threshold / (1 - threshold))


def metric_row(base: Dict[str, object], y_true: np.ndarray, scores: np.ndarray) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= NB_THRESHOLD).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        **base,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "ap": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float(np.mean(y_true)),
        "top5_recall": top_k_recall(y_true, scores, 0.05),
        "top10_recall": top_k_recall(y_true, scores, 0.10),
        "top20_recall": top_k_recall(y_true, scores, 0.20),
        "brier": float(brier_score_loss(y_true, scores)),
        "net_benefit": net_benefit(y_true, scores, NB_THRESHOLD),
        "test_cases": int(len(y_true)),
        "positive_cases": int(np.sum(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else 0.0,
    }


def ensure_tokens(prefix_df: pd.DataFrame) -> pd.DataFrame:
    out = prefix_df.copy()
    if "tokens_raw" not in out.columns:
        out["tokens_raw"] = (
            out["path_text"]
            .fillna("")
            .astype(str)
            .map(lambda text: text.split() if text.split() else ["unknown_activity"])
        )
    return out


def deep_prediction_frame(prefix_df: pd.DataFrame, model_name: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    prefix_df = ensure_tokens(prefix_df)
    train_all = prefix_df[prefix_df["start_time"] <= cutoff].copy().sort_values("start_time")
    test_df = prefix_df[prefix_df["start_time"] > cutoff].copy().sort_values("start_time")
    if len(train_all) < 50 or len(test_df) < 20:
        return pd.DataFrame()
    if train_all["anomaly_label"].nunique() < 2 or test_df["anomaly_label"].nunique() < 2:
        return pd.DataFrame()

    if len(train_all) >= 1024:
        val_size = max(512, int(len(train_all) * 0.15))
    else:
        val_size = max(32, int(len(train_all) * 0.15))
    val_size = min(val_size, max(1, len(train_all) - 2))
    val_df = train_all.iloc[-val_size:].copy()
    train_df = train_all.iloc[:-val_size].copy()
    if train_df["anomaly_label"].nunique() < 2:
        return pd.DataFrame()

    vocab = build_vocab(train_df["tokens_raw"].tolist())
    for frame in (train_df, val_df, test_df):
        frame["tokens"] = encode_sequences(frame["tokens_raw"].tolist(), vocab)

    train_loader = DataLoader(
        PrefixSequenceDataset(make_examples(train_df)),
        batch_size=256,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        PrefixSequenceDataset(make_examples(val_df)),
        batch_size=512,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        PrefixSequenceDataset(make_examples(test_df)),
        batch_size=512,
        shuffle=False,
        collate_fn=collate_batch,
    )

    set_seed(SEED)
    device = torch.device("cpu")
    if model_name == "LSTM":
        model = LSTMBaseline(vocab_size=len(vocab)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    elif model_name == "Transformer":
        model = TransformerBaseline(vocab_size=len(vocab)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    else:
        raise ValueError(model_name)

    pos_rate = float(train_df["anomaly_label"].mean())
    pos_weight = torch.tensor([(1.0 - pos_rate) / max(pos_rate, 1e-6)], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_state = None
    best_val_ap = -1.0
    patience_left = 3

    for _epoch in range(1, 16):
        model.train()
        for x, lengths, y in train_loader:
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_scores = predict_scores(model, val_loader, device)
        val_y = val_df["anomaly_label"].to_numpy(dtype=int)
        val_ap = (
            average_precision_score(val_y, val_scores)
            if len(np.unique(val_y)) > 1
            else float(np.mean(val_y))
        )
        if val_ap > best_val_ap + 1e-5:
            best_val_ap = float(val_ap)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = 3
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        return pd.DataFrame()
    model.load_state_dict(best_state)
    scores = predict_scores(model, test_loader, device)
    return pd.DataFrame(
        {
            "caseid": test_df["caseid"].to_numpy(),
            "model": model_name,
            "label": test_df["anomaly_label"].to_numpy(dtype=int),
            "score": scores,
        }
    )


def checkpoint_display(check_type: str, raw_value: float | int) -> str:
    if check_type == "Proportional prefix":
        return f"{int(round(float(raw_value) * 100))}%"
    if check_type == "Fixed-event checkpoint":
        return f"Event {int(raw_value)}"
    if check_type == "Natural-time checkpoint":
        return f"{int(raw_value)}h"
    return str(raw_value)


def evaluate_main_prefix_table(
    dataset_key: str,
    check_type: str,
    checkpoint_raw: float | int,
    prefix_df: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> list[Dict[str, object]]:
    train_df = prefix_df[prefix_df["start_time"] <= cutoff].copy()
    test_df = prefix_df[prefix_df["start_time"] > cutoff].copy()
    if train_df["anomaly_label"].nunique() < 2 or test_df["anomaly_label"].nunique() < 2:
        return []
    if check_type == "Proportional prefix":
        _metrics_df, pred_df = evaluate_main_ratio_models(train_df, test_df)
    else:
        _metrics_df, pred_df = evaluate_main_checkpoint_models(train_df, test_df)
    rows: list[Dict[str, object]] = []
    for model, part in pred_df.groupby("model", sort=False):
        rows.append(
            metric_row(
                {
                    "dataset_key": dataset_key,
                    "dataset": DATASET_LABEL[dataset_key],
                    "check_type": check_type,
                    "checkpoint_raw": checkpoint_raw,
                    "checkpoint": checkpoint_display(check_type, checkpoint_raw),
                    "model": model,
                },
                part["label"].to_numpy(dtype=int),
                part["score"].to_numpy(dtype=float),
            )
        )
    for model_name in DEEP_MODELS:
        deep_pred = deep_prediction_frame(prefix_df, model_name, cutoff)
        if deep_pred.empty:
            continue
        rows.append(
            metric_row(
                {
                    "dataset_key": dataset_key,
                    "dataset": DATASET_LABEL[dataset_key],
                    "check_type": check_type,
                    "checkpoint_raw": checkpoint_raw,
                    "checkpoint": checkpoint_display(check_type, checkpoint_raw),
                    "model": model_name,
                },
                deep_pred["label"].to_numpy(dtype=int),
                deep_pred["score"].to_numpy(dtype=float),
            )
        )
    return rows


def compute_main_rows() -> list[Dict[str, object]]:
    df = load_event_log()
    labels = build_full_labels(df)
    cutoff = labels["start_time"].quantile(0.7)
    rows: list[Dict[str, object]] = []
    for ratio in RATIOS:
        prefix_df = build_main_ratio_table(df, labels, ratio)
        rows.extend(evaluate_main_prefix_table("corporate_account_opening", "Proportional prefix", ratio, prefix_df, cutoff))
        print(f"main ratio {ratio:.1f}")
    for checkpoint in FIXED_EVENTS:
        prefix_df = build_main_fixed_event_table(df, labels, checkpoint)
        rows.extend(
            evaluate_main_prefix_table("corporate_account_opening", "Fixed-event checkpoint", checkpoint, prefix_df, cutoff)
        )
        print(f"main fixed {checkpoint}")
    for hours in NATURAL_HOURS:
        prefix_df = build_main_natural_time_table(df, labels, hours)
        rows.extend(evaluate_main_prefix_table("corporate_account_opening", "Natural-time checkpoint", hours, prefix_df, cutoff))
        print(f"main natural {hours}")
    return rows


def evaluate_external_table(
    dataset_key: str,
    check_type: str,
    checkpoint_raw: float | int,
    table: pd.DataFrame,
) -> list[Dict[str, object]]:
    if table.empty or table["anomaly_label"].nunique() < 2:
        return []
    cutoff = table["start_time"].quantile(0.7)
    train_df = table[table["start_time"] <= cutoff].copy()
    test_df = table[table["start_time"] > cutoff].copy()
    if train_df["anomaly_label"].nunique() < 2 or test_df["anomaly_label"].nunique() < 2:
        return []
    y_test = test_df["anomaly_label"].to_numpy(dtype=int)
    rows: list[Dict[str, object]] = []
    base = {
        "dataset_key": dataset_key,
        "dataset": DATASET_LABEL[dataset_key],
        "check_type": check_type,
        "checkpoint_raw": checkpoint_raw,
        "checkpoint": checkpoint_display(check_type, checkpoint_raw),
    }

    for model, pack in external_model_scores(train_df, test_df).items():
        rows.append(metric_row({**base, "model": model}, y_test, np.asarray(pack["test_scores"], dtype=float)))

    y_train = train_df["anomaly_label"].to_numpy(dtype=int)
    xgb = build_external_xgboost_model(y_train)
    cols = NUMERIC_COLS + CATEGORICAL_COLS
    xgb.fit(train_df[cols], y_train)
    xgb_scores = xgb.predict_proba(test_df[cols])[:, 1]
    rows.append(metric_row({**base, "model": "xgboost"}, y_test, xgb_scores))
    for model_name in DEEP_MODELS:
        deep_pred = deep_prediction_frame(table, model_name, cutoff)
        if deep_pred.empty:
            continue
        rows.append(
            metric_row(
                {**base, "model": model_name},
                deep_pred["label"].to_numpy(dtype=int),
                deep_pred["score"].to_numpy(dtype=float),
            )
        )
    return rows


def compute_external_rows() -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    for dataset_key in EXTERNAL_DATASETS:
        print(f"loading {dataset_key}")
        traces = LOADERS[dataset_key]()
        traces = deterministic_temporal_sample(traces, dataset_key)
        labels, explicit_states, _label_note = build_labels(dataset_key, traces)
        mandatory_flow = infer_main_path(traces, labels)
        for ratio in RATIOS:
            table = build_external_ratio_table(traces, labels, ratio, mandatory_flow, explicit_states)
            rows.extend(evaluate_external_table(dataset_key, "Proportional prefix", ratio, table))
            print(f"  ratio {ratio:.1f}")
        for checkpoint in FIXED_EVENTS:
            table = build_external_fixed_table(traces, labels, mandatory_flow, explicit_states, checkpoint)
            rows.extend(evaluate_external_table(dataset_key, "Fixed-event checkpoint", checkpoint, table))
            print(f"  fixed {checkpoint}")
        for hours in NATURAL_HOURS:
            table = build_external_natural_table(traces, labels, mandatory_flow, explicit_states, hours)
            rows.extend(evaluate_external_table(dataset_key, "Natural-time checkpoint", hours, table))
            print(f"  natural {hours}")
    return rows


def select_best_by_checkpoint(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.sort_values(
            ["dataset_key", "check_type", "checkpoint_raw", "ap", "brier", "top20_recall"],
            ascending=[True, True, True, False, True, False],
        )
        .groupby(["dataset_key", "check_type", "checkpoint_raw"], as_index=False)
        .head(1)
        .copy()
    )


def method_setting(check_type: str) -> str:
    if check_type == "Proportional prefix":
        return "Mean across 10%-90%"
    if check_type == "Fixed-event checkpoint":
        return "Mean across events 3/5/7"
    if check_type == "Natural-time checkpoint":
        return "Mean across 24/72/168 h"
    return ""


def method_judgement(check_type: str, ap: float, nb: float) -> str:
    if check_type == "Proportional prefix":
        if ap >= 0.90 and nb > 0:
            return "Strong theoretical benchmark performance"
        return "Useful theoretical benchmark"
    if check_type == "Fixed-event checkpoint":
        if ap >= 0.80 and nb > 0:
            return "Strong fit for checkpoint-based early warning"
        return "Suitable for checkpoint-based supplementary screening"
    if check_type == "Natural-time checkpoint":
        if ap >= 0.80 and nb > 0:
            return "Suitable for periodic review"
        return "Suitable for periodic supplementary review"
    return ""


def build_appendix_a2(best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    method_order = {"Proportional prefix": 0, "Fixed-event checkpoint": 1, "Natural-time checkpoint": 2}
    for dataset_key in DATASET_ORDER:
        for check_type in ["Proportional prefix", "Fixed-event checkpoint", "Natural-time checkpoint"]:
            sub = best[(best["dataset_key"] == dataset_key) & (best["check_type"] == check_type)].copy()
            if sub.empty:
                continue
            avg = sub[
                ["ap", "top5_recall", "top10_recall", "top20_recall", "brier", "net_benefit"]
            ].mean()
            rows.append(
                {
                    "_dataset_order": DATASET_ORDER.index(dataset_key),
                    "_method_order": method_order[check_type],
                    "Dataset": DATASET_LABEL[dataset_key],
                    "Check method": check_type,
                    "Checkpoint setting": method_setting(check_type),
                    "AP": avg["ap"],
                    "Top-5% Recall": avg["top5_recall"],
                    "Top-10% Recall": avg["top10_recall"],
                    "Top-20% Recall": avg["top20_recall"],
                    "Brier Score": avg["brier"],
                    "Net Benefit": avg["net_benefit"],
                    "Main assessment": method_judgement(check_type, avg["ap"], avg["net_benefit"]),
                }
            )
    out = pd.DataFrame(rows).sort_values(["_dataset_order", "_method_order"])
    return out.drop(columns=["_dataset_order", "_method_order"])


def build_appendix_a3(best: pd.DataFrame) -> pd.DataFrame:
    out = best[best["check_type"].isin(["Fixed-event checkpoint", "Natural-time checkpoint"])].copy()
    method_order = {"Fixed-event checkpoint": 0, "Natural-time checkpoint": 1}
    out["_dataset_order"] = out["dataset_key"].map(lambda x: DATASET_ORDER.index(x))
    out["_method_order"] = out["check_type"].map(method_order)
    out = out.sort_values(["_dataset_order", "_method_order", "checkpoint_raw"])
    return out[
        [
            "dataset",
            "check_type",
            "checkpoint",
            "ap",
            "f1",
            "recall",
            "top10_recall",
            "top20_recall",
            "brier",
            "net_benefit",
        ]
    ].rename(
        columns={
            "dataset": "Dataset",
            "check_type": "Check method",
            "checkpoint": "Checkpoint",
            "ap": "AP",
            "f1": "F1",
            "recall": "Recall",
            "top10_recall": "Top-10% Recall",
            "top20_recall": "Top-20% Recall",
            "brier": "Brier Score",
            "net_benefit": "Net Benefit",
        }
    )


def write_outputs(metrics: pd.DataFrame, best: pd.DataFrame, a2: pd.DataFrame) -> None:
    metrics.to_csv(OUT_DIR / "appendix_a2_a3_all_model_metrics.csv", index=False, encoding="utf-8-sig")
    best.to_csv(OUT_DIR / "appendix_a2_a3_best_by_checkpoint_raw.csv", index=False, encoding="utf-8-sig")
    a2.to_csv(OUT_DIR / "appendix_a2_cross_scene_pgv_summary.csv", index=False, encoding="utf-8-sig")


def validate_outputs(a2: pd.DataFrame) -> None:
    expected_a2 = len(DATASET_ORDER) * 3
    if len(a2) != expected_a2:
        raise RuntimeError(f"A2 row count mismatch: {len(a2)} != {expected_a2}")
    numeric_cols = [c for c in a2.columns if c not in {"Dataset", "Check method", "Checkpoint setting", "Main assessment"}]
    if a2[numeric_cols].isna().any().any():
        raise RuntimeError("A2 contains NaN numeric values")


def main() -> None:
    rows = []
    rows.extend(compute_main_rows())
    rows.extend(compute_external_rows())
    metrics = pd.DataFrame(rows)
    best = select_best_by_checkpoint(metrics)
    a2 = build_appendix_a2(best)
    validate_outputs(a2)
    write_outputs(metrics, best, a2)
    print("A2")
    print(a2.to_string(index=False))


if __name__ == "__main__":
    main()
