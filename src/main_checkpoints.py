from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBClassifier


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "private" / "corporate_account_opening.csv"
RESULTS_DIR = ROOT / "results" / "main" / "h1_operational_checkpoints_six_models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ACTIVITY_ALIAS = {
    "\u9884\u7533\u8bf7\u5df2\u63d0\u4ea4\u5f85\u5ba1\u6838": "pre_application_submitted",
    "\u6d41\u7a0b\u94f6\u884c\u521d\u5ba1\u5c97\u9884\u5ba1\u6838\u4e2d": "pre_review_in_progress",
    "\u9884\u5ba1\u901a\u8fc7\u5f85\u7f51\u70b9\u5904\u7406": "pre_review_passed_wait_branch",
    "\u7f51\u70b9\u4e34\u67dc\u524d\u51c6\u5907\u4e2d": "branch_preparation",
    "\u67dc\u9762\u4e1a\u52a1\u5df2\u6fc0\u6d3b\u53d7\u7406\u4e2d": "counter_activated_processing",
    "\u5b8c\u6210\u7269\u54c1\u4ea4\u63a5\u5ba2\u6237\u79bb\u67dc": "handover_completed_leave_counter",
    "\u8d26\u6237\u5f00\u7acb\u5df2\u5b8c\u6210": "account_opening_completed",
    "\u4e1a\u52a1\u5b8c\u6210\u5f85\u8bc4\u4ef7": "service_completed_wait_feedback",
    "\u5df2\u626b\u63cf\u5f85\u6b21\u65e5\u5ba1\u6838": "scanned_wait_next_day_review",
    "\u5df2\u5b8c\u6210\u6b21\u65e5\u5ba1\u6838": "next_day_review_completed",
    "\u9884\u5ba1\u5f85\u5ba2\u6237\u4fee\u6539/\u8865\u5145\u8d44\u6599": "waiting_customer_revision",
    "\u6b21\u65e5\u5ba1\u6838\u9000\u56de\u4fee\u6539\u4e2d": "next_day_review_returned_for_revision",
    "\u4e1a\u52a1\u9884\u5ba1\u5df2\u62d2\u7edd": "pre_review_rejected",
    "\u5ba2\u6237\u5df2\u64a4\u5355": "customer_cancelled",
    "\u73b0\u573a\u5ba1\u6838\u672a\u901a\u8fc7\u4e14\u64a4\u5355": "onsite_failed_and_cancelled",
    "\u5f00\u901a\u7535\u5b50\u6e20\u9053\u9884\u586b\u5355": "e_channel_prefill",
    "\u5ba2\u6237\u5df2\u8bc4\u4ef7": "customer_rated",
    "\u5ba2\u6237\u9884\u7ea6\u7f51\u70b9\u5b8c\u6210": "branch_appointment_completed",
    "\u5ba2\u6237\u5df2\u53d6\u9884\u7ea6\u53f7": "appointment_number_taken",
}

MANDATORY_FLOW = [
    "pre_application_submitted",
    "pre_review_in_progress",
    "pre_review_passed_wait_branch",
    "branch_preparation",
    "counter_activated_processing",
    "account_opening_completed",
]

EXPLICIT_ANOMALY_STATES = {
    "waiting_customer_revision",
    "next_day_review_returned_for_revision",
    "pre_review_rejected",
    "customer_cancelled",
    "onsite_failed_and_cancelled",
}

NUMERIC_FEATURES = [
    "sequence_length",
    "compressed_length",
    "observed_duration_hours",
    "gap_mean_hours",
    "gap_max_hours",
    "gap_std_hours",
    "cumulative_wait_hours",
    "repeat_state_count",
    "skipped_mandatory_count",
    "prefix_event_count",
]
CATEGORICAL_FEATURES = ["first_activity", "last_activity"]
TABULAR_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def load_event_log() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="gbk")
    df["dateTime"] = pd.to_datetime(df["dateTime"])
    df["activity_alias"] = df["activity"].map(ACTIVITY_ALIAS).fillna("unknown_activity")
    return df.sort_values(["caseid", "dateTime"]).reset_index(drop=True)


def compress_repeats(seq: Sequence[str]) -> List[str]:
    out: List[str] = []
    for item in seq:
        if not out or out[-1] != item:
            out.append(item)
    return out


def build_full_labels(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for caseid, group in df.groupby("caseid", sort=False):
        seq = group["activity_alias"].tolist()
        times = group["dateTime"].tolist()
        rows.append(
            {
                "caseid": caseid,
                "start_time": times[0],
                "full_length": len(seq),
                "full_duration_hours": float((times[-1] - times[0]).total_seconds() / 3600.0),
                "rework": int(("waiting_customer_revision" in seq) or ("next_day_review_returned_for_revision" in seq)),
                "rejected": int("pre_review_rejected" in seq),
                "cancelled": int(("customer_cancelled" in seq) or ("onsite_failed_and_cancelled" in seq)),
            }
        )
    labels = pd.DataFrame(rows)
    p95 = labels["full_duration_hours"].quantile(0.95)
    labels["duration_anomaly"] = (labels["full_duration_hours"] > p95).astype(int)
    labels["anomaly_label"] = (
        (labels["duration_anomaly"] == 1)
        | (labels["rework"] == 1)
        | (labels["rejected"] == 1)
        | (labels["cancelled"] == 1)
    ).astype(int)
    return labels


def prefix_row(caseid: str, seq: Sequence[str], times: Sequence[pd.Timestamp], prefix_len: int, label_map: Dict[str, Dict]) -> Dict:
    prefix_len = max(1, min(prefix_len, len(seq)))
    seq_prefix = list(seq[:prefix_len])
    time_prefix = list(times[:prefix_len])
    seq_comp = compress_repeats(seq_prefix)
    gaps = np.diff(np.array(time_prefix, dtype="datetime64[m]")).astype("timedelta64[m]").astype(int) / 60.0
    gaps = gaps.tolist()
    transitions = list(zip(seq_comp[:-1], seq_comp[1:]))
    transition_counts: Dict[str, float] = {}
    for src, dst in transitions:
        key = f"{src}__{dst}"
        transition_counts[key] = transition_counts.get(key, 0.0) + 1.0
    total_transitions = float(sum(transition_counts.values())) or 1.0
    transition_probs = {k: v / total_transitions for k, v in transition_counts.items()}
    row = {
        "caseid": caseid,
        "prefix_event_count": prefix_len,
        "sequence_length": len(seq_prefix),
        "compressed_length": len(seq_comp),
        "observed_duration_hours": float((time_prefix[-1] - time_prefix[0]).total_seconds() / 3600.0)
        if len(time_prefix) > 1
        else 0.0,
        "gap_mean_hours": float(np.mean(gaps)) if gaps else 0.0,
        "gap_max_hours": float(np.max(gaps)) if gaps else 0.0,
        "gap_std_hours": float(np.std(gaps)) if gaps else 0.0,
        "cumulative_wait_hours": float(np.sum(gaps)) if gaps else 0.0,
        "repeat_state_count": float(sum(1 for i in range(1, len(seq_prefix)) if seq_prefix[i] == seq_prefix[i - 1])),
        "skipped_mandatory_count": float(sum(step not in seq_prefix for step in MANDATORY_FLOW)),
        "first_activity": seq_prefix[0],
        "last_activity": seq_prefix[-1],
        "path_text": " ".join(seq_comp),
        "transition_features": transition_probs,
        "tokens_raw": seq_prefix,
    }
    row.update(label_map[caseid])
    return row


def first_explicit_index(seq: Sequence[str]) -> int | None:
    for idx, act in enumerate(seq):
        if act in EXPLICIT_ANOMALY_STATES:
            return idx
    return None


def build_fixed_event_table(df: pd.DataFrame, labels: pd.DataFrame, checkpoint: int) -> pd.DataFrame:
    label_map = labels.set_index("caseid").to_dict("index")
    rows = []
    for caseid, group in df.groupby("caseid", sort=False):
        seq = group["activity_alias"].tolist()
        times = group["dateTime"].tolist()
        base = min(checkpoint, len(seq) - 1) if len(seq) > 1 else 1
        explicit = first_explicit_index(seq)
        if explicit is not None:
            base = min(base, max(1, explicit))
        rows.append(prefix_row(caseid, seq, times, max(1, base), label_map))
    out = pd.DataFrame(rows)
    out["checkpoint_type"] = "fixed_event"
    out["checkpoint"] = checkpoint
    out["checkpoint_label"] = f"First {checkpoint} events"
    return out


def build_natural_time_table(df: pd.DataFrame, labels: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    label_map = labels.set_index("caseid").to_dict("index")
    horizon = pd.Timedelta(hours=horizon_hours)
    rows = []
    for caseid, group in df.groupby("caseid", sort=False):
        seq = group["activity_alias"].tolist()
        times = group["dateTime"].tolist()
        start = times[0]
        end = times[-1]
        if end <= start + horizon:
            continue
        allowed_indices = [idx for idx, timestamp in enumerate(times) if timestamp <= start + horizon]
        prefix_len = 1 if not allowed_indices else max(1, max(allowed_indices) + 1)
        explicit = first_explicit_index(seq)
        if explicit is not None:
            prefix_len = min(prefix_len, max(1, explicit))
        rows.append(prefix_row(caseid, seq, times, prefix_len, label_map))
    out = pd.DataFrame(rows)
    out["checkpoint_type"] = "natural_time"
    out["checkpoint"] = horizon_hours
    out["checkpoint_label"] = f"{horizon_hours} h"
    return out


def build_tabular_rf_model() -> Pipeline:
    prep = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=2,
    )
    return Pipeline([("prep", prep), ("clf", clf)])


def build_xgboost_model(y_train: np.ndarray) -> Pipeline:
    pos = max(1, int(np.sum(y_train == 1)))
    neg = max(1, int(np.sum(y_train == 0)))
    prep = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    clf = XGBClassifier(
        n_estimators=260,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=neg / pos,
        random_state=SEED,
        n_jobs=2,
    )
    return Pipeline([("prep", prep), ("clf", clf)])


def build_sequence_ngram_model() -> Pipeline:
    return Pipeline(
        [
            ("vec", CountVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 3), min_df=2)),
            ("clf", LogisticRegression(max_iter=1200, class_weight="balanced", solver="liblinear", random_state=SEED)),
        ]
    )


def net_benefit(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    y_pred = scores >= threshold
    tp = float(np.sum((y_true == 1) & y_pred))
    fp = float(np.sum((y_true == 0) & y_pred))
    n = float(len(y_true))
    return tp / n - fp / n * (threshold / (1.0 - threshold))


def eval_scores(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    y_pred = (scores >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.5,
        "ap": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float(np.mean(y_true)),
        "brier": float(brier_score_loss(y_true, scores)),
        "net_benefit_03": float(net_benefit(y_true, scores, 0.3)),
        "net_benefit_05": float(net_benefit(y_true, scores, 0.5)),
    }


def evaluate_classical_models(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_train = train_df["anomaly_label"].to_numpy()
    y_test = test_df["anomaly_label"].to_numpy()
    pred_rows = []
    metric_rows = []

    models = [
        ("XGBoost", build_xgboost_model(y_train), TABULAR_FEATURES),
        ("Tabular RF", build_tabular_rf_model(), TABULAR_FEATURES),
        ("Sequence N-gram LR", build_sequence_ngram_model(), "path_text"),
    ]
    for name, model, cols in models:
        model.fit(train_df[cols], y_train)
        scores = model.predict_proba(test_df[cols])[:, 1]
        metric = eval_scores(y_test, scores)
        metric["model"] = name
        metric_rows.append(metric)
        pred_rows.append(pd.DataFrame({"caseid": test_df["caseid"].to_numpy(), "model": name, "label": y_test, "score": scores}))

    gvec = DictVectorizer(sparse=True)
    xg_train = gvec.fit_transform(train_df["transition_features"].tolist())
    xg_test = gvec.transform(test_df["transition_features"].tolist())
    if xg_train.shape[1] == 0 or xg_train.nnz == 0:
        graph_scores = np.repeat(float(y_train.mean()), len(test_df))
    else:
        graph_model = LogisticRegression(max_iter=1200, class_weight="balanced", solver="liblinear", random_state=SEED)
        graph_model.fit(xg_train, y_train)
        graph_scores = graph_model.predict_proba(xg_test)[:, 1]
    metric = eval_scores(y_test, graph_scores)
    metric["model"] = "Graph Transition LR"
    metric_rows.append(metric)
    pred_rows.append(pd.DataFrame({"caseid": test_df["caseid"].to_numpy(), "model": "Graph Transition LR", "label": y_test, "score": graph_scores}))

    return pd.DataFrame(metric_rows), pd.concat(pred_rows, ignore_index=True)


def build_vocab(train_tokens: Sequence[Sequence[str]]) -> Dict[str, int]:
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for seq in train_tokens:
        for tok in seq:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def encode_sequences(tokens_list: Sequence[Sequence[str]], vocab: Dict[str, int]) -> List[List[int]]:
    return [[vocab.get(tok, 1) for tok in seq] for seq in tokens_list]


@dataclass
class SequenceExample:
    tokens: List[int]
    label: int


class PrefixSequenceDataset(Dataset):
    def __init__(self, examples: Sequence[SequenceExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SequenceExample:
        return self.examples[idx]


def make_examples(df: pd.DataFrame) -> List[SequenceExample]:
    return [SequenceExample(tokens=row["tokens"], label=int(row["anomaly_label"])) for _, row in df.iterrows()]


def collate_batch(batch: Sequence[SequenceExample]):
    lengths = torch.tensor([len(ex.tokens) for ex in batch], dtype=torch.long)
    max_len = int(lengths.max())
    x = torch.zeros(len(batch), max_len, dtype=torch.long)
    y = torch.tensor([ex.label for ex in batch], dtype=torch.float32)
    for i, ex in enumerate(batch):
        x[i, : len(ex.tokens)] = torch.tensor(ex.tokens, dtype=torch.long)
    return x, lengths, y


class LSTMBaseline(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 32, hidden_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.out(self.dropout(h_n[-1])).squeeze(1)


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerBaseline(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 32, nhead: int = 4, ff_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.position = PositionalEncoding(emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=0.2,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(emb_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = x.eq(0)
        emb = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        emb = self.position(emb)
        encoded = self.encoder(emb, src_key_padding_mask=mask)
        valid = (~mask).unsqueeze(-1)
        pooled = (encoded * valid).sum(dim=1) / lengths.unsqueeze(1).clamp_min(1).to(encoded.dtype)
        return self.out(self.dropout(pooled)).squeeze(1)


def predict_scores(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    all_scores: List[np.ndarray] = []
    with torch.no_grad():
        for x, lengths, _ in loader:
            x = x.to(device)
            lengths = lengths.to(device)
            logits = model(x, lengths)
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_scores)


def fit_sequence_model(prefix_df: pd.DataFrame, model_name: str, cutoff: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    train_all = prefix_df[prefix_df["start_time"] <= cutoff].copy().sort_values("start_time")
    test_df = prefix_df[prefix_df["start_time"] > cutoff].copy().sort_values("start_time")
    val_size = max(512, int(len(train_all) * 0.15))
    val_df = train_all.iloc[-val_size:].copy()
    train_df = train_all.iloc[:-val_size].copy()

    vocab = build_vocab(train_df["tokens_raw"].tolist())
    for frame in (train_df, val_df, test_df):
        frame["tokens"] = encode_sequences(frame["tokens_raw"].tolist(), vocab)

    train_loader = DataLoader(PrefixSequenceDataset(make_examples(train_df)), batch_size=256, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(PrefixSequenceDataset(make_examples(val_df)), batch_size=512, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(PrefixSequenceDataset(make_examples(test_df)), batch_size=512, shuffle=False, collate_fn=collate_batch)

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
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_state = None
    best_val_ap = -1.0
    best_epoch = -1
    patience_left = 3

    for epoch in range(1, 16):
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
        val_ap = average_precision_score(val_df["anomaly_label"].to_numpy(), val_scores)
        if val_ap > best_val_ap + 1e-5:
            best_val_ap = float(val_ap)
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = 3
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    scores = predict_scores(model, test_loader, device)
    y_test = test_df["anomaly_label"].to_numpy()
    metrics = eval_scores(y_test, scores)
    metrics.update(
        {
            "model": model_name,
            "train_cases": int(len(train_df)),
            "val_cases": int(len(val_df)),
            "test_cases": int(len(test_df)),
            "vocab_size": int(len(vocab)),
            "best_epoch": int(best_epoch),
            "best_val_ap": float(best_val_ap),
        }
    )
    pred_df = pd.DataFrame({"caseid": test_df["caseid"].to_numpy(), "model": model_name, "label": y_test, "score": scores})
    return metrics, pred_df


def evaluate_checkpoint_table(prefix_df: pd.DataFrame, cutoff: pd.Timestamp, checkpoint_type: str, checkpoint_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = prefix_df[prefix_df["start_time"] <= cutoff].copy()
    test_df = prefix_df[prefix_df["start_time"] > cutoff].copy()
    metrics_df, pred_df = evaluate_classical_models(train_df, test_df)
    metrics_df["checkpoint_type"] = checkpoint_type
    metrics_df["checkpoint_label"] = checkpoint_label
    metrics_df["train_cases"] = len(train_df)
    metrics_df["test_cases"] = len(test_df)
    pred_df["checkpoint_type"] = checkpoint_type
    pred_df["checkpoint_label"] = checkpoint_label

    deep_rows = []
    deep_preds = []
    for model_name in ["LSTM", "Transformer"]:
        metrics, preds = fit_sequence_model(prefix_df, model_name, cutoff)
        metrics["checkpoint_type"] = checkpoint_type
        metrics["checkpoint_label"] = checkpoint_label
        deep_rows.append(metrics)
        preds["checkpoint_type"] = checkpoint_type
        preds["checkpoint_label"] = checkpoint_label
        deep_preds.append(preds)
    return pd.concat([metrics_df, pd.DataFrame(deep_rows)], ignore_index=True), pd.concat([pred_df, *deep_preds], ignore_index=True)


def main() -> None:
    df = load_event_log()
    labels = build_full_labels(df)
    cutoff = labels["start_time"].quantile(0.7)

    metrics_frames = []
    prediction_frames = []
    for checkpoint in [3, 5, 7]:
        prefix_df = build_fixed_event_table(df, labels, checkpoint)
        prefix_df.to_csv(RESULTS_DIR / f"fixed_event_{checkpoint}_features.csv", index=False, encoding="utf-8-sig")
        set_seed(SEED)
        metrics, preds = evaluate_checkpoint_table(prefix_df, cutoff, "fixed_event", f"First {checkpoint} events")
        metrics_frames.append(metrics)
        prediction_frames.append(preds)
        print(f"fixed event {checkpoint} done")

    for horizon_hours in [24, 72, 168]:
        prefix_df = build_natural_time_table(df, labels, horizon_hours)
        prefix_df.to_csv(RESULTS_DIR / f"natural_time_{horizon_hours}h_features.csv", index=False, encoding="utf-8-sig")
        set_seed(SEED)
        metrics, preds = evaluate_checkpoint_table(prefix_df, cutoff, "natural_time", f"{horizon_hours} h")
        metrics_frames.append(metrics)
        prediction_frames.append(preds)
        print(f"natural time {horizon_hours}h done")

    metrics_out = pd.concat(metrics_frames, ignore_index=True)
    metrics_out = metrics_out[
        [
            "checkpoint_type",
            "checkpoint_label",
            "model",
            "precision",
            "recall",
            "f1",
            "auc",
            "ap",
            "brier",
            "net_benefit_03",
            "net_benefit_05",
            "train_cases",
            "test_cases",
        ]
    ]
    metrics_out.to_csv(RESULTS_DIR / "h1_operational_checkpoint_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        RESULTS_DIR / "h1_operational_checkpoint_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(metrics_out.to_string(index=False))


if __name__ == "__main__":
    main()
