from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from xgboost import DMatrix, XGBClassifier

from main_baseline import DATA_PATH, NUMERIC_FEATURES, load_event_log


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "results" / "main" / "prefix_10_90_baseline"
RESULTS_DIR = ROOT / "results" / "main" / "tpop_prefix_10_90"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BERT_MODEL = "google-bert/bert-base-uncased"
PREFIX_RATIOS = [i / 10 for i in range(1, 10)]
DEFAULT_SEEDS = [42]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(6, max(1, torch.get_num_threads())))


def parse_tokens(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    parsed = ast.literal_eval(str(value))
    return [str(item) for item in parsed]


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_pred = (scores >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(roc_auc_score(y_true, scores)),
        "ap": float(average_precision_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
    }


def build_case_event_map() -> dict[str, tuple[list[str], np.ndarray]]:
    event_log = load_event_log()
    case_map: dict[str, tuple[list[str], np.ndarray]] = {}
    for caseid, group in event_log.groupby("caseid", sort=False):
        activities = group["activity_alias"].astype(str).tolist()
        times = group["dateTime"].to_numpy(dtype="datetime64[ns]")
        case_map[str(caseid)] = (activities, times)
    return case_map


def collect_activities() -> list[str]:
    table = pd.read_csv(BASELINE_DIR / "prefix_table_90.csv", usecols=["tokens_raw"])
    activities: set[str] = set()
    for value in table["tokens_raw"]:
        activities.update(parse_tokens(value))
    return sorted(activities)


def compress_bert_embeddings(embeddings: np.ndarray, target_dim: int = 32) -> np.ndarray:
    n_components = min(target_dim, embeddings.shape[0] - 1, embeddings.shape[1])
    reduced = PCA(n_components=n_components, svd_solver="full").fit_transform(embeddings)
    reduced = reduced.astype(np.float32)
    if n_components < target_dim:
        reduced = np.pad(reduced, ((0, 0), (0, target_dim - n_components)))
    scale = np.linalg.norm(reduced, axis=1, keepdims=True)
    return reduced / np.maximum(scale, 1e-6)


def build_bert_activity_embeddings(activities: Sequence[str]) -> tuple[dict[str, int], torch.Tensor]:
    cache_path = RESULTS_DIR / "bert_activity_embeddings.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        cached_activities = [str(item) for item in cached["activities"].tolist()]
        if cached_activities == list(activities):
            matrix = torch.tensor(compress_bert_embeddings(cached["embeddings"]), dtype=torch.float32)
            return {activity: idx for idx, activity in enumerate(activities)}, matrix

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModel.from_pretrained(BERT_MODEL)
    model.eval()
    phrases = [activity.replace("_", " ") for activity in activities]
    encoded = tokenizer(phrases, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**encoded).last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
    np.savez_compressed(cache_path, activities=np.array(activities, dtype=object), embeddings=embeddings)
    compressed = compress_bert_embeddings(embeddings)
    return {activity: idx for idx, activity in enumerate(activities)}, torch.tensor(compressed)


def select_features_with_xgboost_shap(
    train_df: pd.DataFrame,
    seed: int = 42,
    cumulative_threshold: float = 0.90,
    min_features: int = 3,
    max_features: int = 8,
) -> tuple[list[str], pd.DataFrame]:
    x = train_df[NUMERIC_FEATURES].astype(float).copy()
    medians = x.median(axis=0)
    x = x.fillna(medians)
    y = train_df["anomaly_label"].to_numpy(dtype=int)
    positive = max(1, int((y == 1).sum()))
    negative = max(1, int((y == 0).sum()))
    proxy = XGBClassifier(
        n_estimators=180,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=negative / positive,
        random_state=seed,
        n_jobs=4,
    )
    proxy.fit(x, y)
    shap_values = proxy.get_booster().predict(DMatrix(x, feature_names=NUMERIC_FEATURES), pred_contribs=True)
    importance = np.abs(shap_values[:, :-1]).mean(axis=0)
    ranking = pd.DataFrame({"feature": NUMERIC_FEATURES, "mean_abs_shap": importance})
    ranking = ranking.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    total = float(ranking["mean_abs_shap"].sum())
    if total <= 0:
        selected = list(ranking["feature"].head(min_features))
        ranking["importance_share"] = 0.0
        ranking["cumulative_share"] = 0.0
    else:
        ranking["importance_share"] = ranking["mean_abs_shap"] / total
        ranking["cumulative_share"] = ranking["importance_share"].cumsum()
        cutoff_count = int((ranking["cumulative_share"] < cumulative_threshold).sum()) + 1
        count = min(max_features, max(min_features, cutoff_count))
        selected = ranking["feature"].head(count).tolist()
    ranking["selected"] = ranking["feature"].isin(selected)
    return selected, ranking


@dataclass
class ScaleParameters:
    event_mean: np.ndarray
    event_std: np.ndarray
    case_mean: np.ndarray
    case_std: np.ndarray


def fit_scale_parameters(
    train_df: pd.DataFrame,
    case_map: dict[str, tuple[list[str], np.ndarray]],
    selected_features: Sequence[str],
) -> ScaleParameters:
    event_blocks: list[np.ndarray] = []
    for row in train_df.itertuples(index=False):
        tokens = parse_tokens(row.tokens_raw)
        _, times = case_map[str(row.caseid)]
        prefix_times = times[: len(tokens)]
        elapsed = (prefix_times - prefix_times[0]) / np.timedelta64(1, "h")
        gaps = np.zeros(len(prefix_times), dtype=np.float32)
        if len(prefix_times) > 1:
            gaps[1:] = np.diff(prefix_times) / np.timedelta64(1, "h")
        position = np.arange(1, len(tokens) + 1, dtype=np.float32) / max(1, len(tokens))
        event_blocks.append(np.column_stack([elapsed, gaps, position]).astype(np.float32))
    event_values = np.concatenate(event_blocks, axis=0)
    case_values = train_df[list(selected_features)].astype(float).to_numpy(dtype=np.float32)
    event_mean = np.nanmean(event_values, axis=0)
    event_std = np.nanstd(event_values, axis=0)
    case_mean = np.nanmean(case_values, axis=0)
    case_std = np.nanstd(case_values, axis=0)
    event_std[event_std < 1e-6] = 1.0
    case_std[case_std < 1e-6] = 1.0
    return ScaleParameters(event_mean, event_std, case_mean, case_std)


@dataclass
class GraphExample:
    caseid: str
    activity_ids: torch.Tensor
    event_features: torch.Tensor
    case_features: torch.Tensor
    label: float


class PrefixGraphDataset(Dataset[GraphExample]):
    def __init__(
        self,
        frame: pd.DataFrame,
        case_map: dict[str, tuple[list[str], np.ndarray]],
        activity_to_id: dict[str, int],
        selected_features: Sequence[str],
        scales: ScaleParameters,
    ) -> None:
        self.examples: list[GraphExample] = []
        for row in frame.itertuples(index=False):
            caseid = str(row.caseid)
            tokens = parse_tokens(row.tokens_raw)
            activities, times = case_map[caseid]
            if activities[: len(tokens)] != tokens:
                raise ValueError(f"Prefix activities do not match raw log for case {caseid}")
            prefix_times = times[: len(tokens)]
            elapsed = (prefix_times - prefix_times[0]) / np.timedelta64(1, "h")
            gaps = np.zeros(len(prefix_times), dtype=np.float32)
            if len(prefix_times) > 1:
                gaps[1:] = np.diff(prefix_times) / np.timedelta64(1, "h")
            position = np.arange(1, len(tokens) + 1, dtype=np.float32) / max(1, len(tokens))
            event_features = np.column_stack([elapsed, gaps, position]).astype(np.float32)
            event_features = (event_features - scales.event_mean) / scales.event_std
            case_features = np.array(
                [float(getattr(row, feature)) for feature in selected_features], dtype=np.float32
            )
            case_features = np.nan_to_num(
                (case_features - scales.case_mean) / scales.case_std, nan=0.0, posinf=0.0, neginf=0.0
            )
            self.examples.append(
                GraphExample(
                    caseid=caseid,
                    activity_ids=torch.tensor([activity_to_id[token] for token in tokens], dtype=torch.long),
                    event_features=torch.tensor(event_features, dtype=torch.float32),
                    case_features=torch.tensor(case_features, dtype=torch.float32),
                    label=float(row.anomaly_label),
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> GraphExample:
        return self.examples[index]


def collate_graphs(batch: Sequence[GraphExample]) -> dict[str, object]:
    lengths = torch.tensor([len(item.activity_ids) for item in batch], dtype=torch.long)
    max_length = int(lengths.max())
    activity_ids = torch.zeros((len(batch), max_length), dtype=torch.long)
    event_features = torch.zeros((len(batch), max_length, 3), dtype=torch.float32)
    node_mask = torch.zeros((len(batch), max_length), dtype=torch.bool)
    case_features = torch.stack([item.case_features for item in batch])
    labels = torch.tensor([item.label for item in batch], dtype=torch.float32)
    caseids: list[str] = []
    for index, item in enumerate(batch):
        length = len(item.activity_ids)
        activity_ids[index, :length] = item.activity_ids
        event_features[index, :length] = item.event_features
        node_mask[index, :length] = True
        caseids.append(item.caseid)
    edge_mask = node_mask[:, 1:] & node_mask[:, :-1] if max_length > 1 else node_mask[:, :0]
    return {
        "caseids": caseids,
        "activity_ids": activity_ids,
        "event_features": event_features,
        "case_features": case_features,
        "node_mask": node_mask,
        "edge_mask": edge_mask,
        "labels": labels,
    }


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaptedTPOP(nn.Module):
    def __init__(
        self,
        activity_embeddings: torch.Tensor,
        case_feature_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.activity_embedding = nn.Embedding.from_pretrained(activity_embeddings, freeze=True)
        self.activity_projection = nn.Linear(activity_embeddings.shape[1], hidden_dim)
        self.event_projection = nn.Linear(3, hidden_dim)
        self.case_projection = nn.Linear(case_feature_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.base_message = MLP(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.attention_extractor = MLP(hidden_dim * 2, hidden_dim, 1, dropout)
        self.attended_message = MLP(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.output = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.epsilon = nn.Parameter(torch.tensor(0.0))

    def encode_nodes(
        self,
        activity_ids: torch.Tensor,
        event_features: torch.Tensor,
        case_features: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        activity = self.activity_projection(self.activity_embedding(activity_ids))
        event = self.event_projection(event_features)
        case = self.case_projection(case_features).unsqueeze(1)
        hidden = torch.relu(self.input_norm(activity + event + case))
        return hidden * node_mask.unsqueeze(-1)

    def message_pass(
        self,
        hidden: torch.Tensor,
        node_mask: torch.Tensor,
        edge_attention: torch.Tensor | None,
        layer: nn.Module,
    ) -> torch.Tensor:
        aggregate = torch.zeros_like(hidden)
        if hidden.shape[1] > 1:
            messages = hidden[:, :-1]
            if edge_attention is not None:
                messages = messages * edge_attention.unsqueeze(-1)
            aggregate[:, 1:] = messages
        updated = layer((1.0 + self.epsilon) * hidden + aggregate)
        return updated * node_mask.unsqueeze(-1)

    @staticmethod
    def graph_pool(hidden: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        weights = node_mask.unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

    def forward_base(self, batch: dict[str, object]) -> torch.Tensor:
        hidden = self.encode_nodes(
            batch["activity_ids"], batch["event_features"], batch["case_features"], batch["node_mask"]
        )
        hidden = self.message_pass(hidden, batch["node_mask"], None, self.base_message)
        return self.output(self.graph_pool(hidden, batch["node_mask"])).squeeze(1)

    def forward_gsat(
        self,
        batch: dict[str, object],
        training: bool,
        temperature: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encode_nodes(
            batch["activity_ids"], batch["event_features"], batch["case_features"], batch["node_mask"]
        )
        context = self.message_pass(hidden, batch["node_mask"], None, self.base_message)
        if hidden.shape[1] > 1:
            edge_pairs = torch.cat([context[:, :-1], context[:, 1:]], dim=-1)
            attention_logits = self.attention_extractor(edge_pairs).squeeze(-1)
            if training:
                uniform = torch.rand_like(attention_logits).clamp_(1e-6, 1 - 1e-6)
                logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
                attention = torch.sigmoid((attention_logits + logistic_noise) / temperature)
            else:
                attention = torch.sigmoid(attention_logits)
            attention = attention * batch["edge_mask"]
        else:
            attention_logits = hidden.new_zeros((hidden.shape[0], 0))
            attention = attention_logits
        attended = self.message_pass(hidden, batch["node_mask"], attention, self.attended_message)
        logits = self.output(self.graph_pool(attended, batch["node_mask"])).squeeze(1)
        return logits, attention, attention_logits


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def predict_model(
    model: AdaptedTPOP,
    loader: DataLoader,
    device: torch.device,
    use_gsat: bool,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    model.eval()
    caseids: list[str] = []
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            if use_gsat:
                logits, _, _ = model.forward_gsat(batch, training=False)
            else:
                logits = model.forward_base(batch)
            caseids.extend(raw_batch["caseids"])
            labels.append(batch["labels"].cpu().numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return caseids, np.concatenate(labels), np.concatenate(scores)


def fit_one_seed(
    prefix_df: pd.DataFrame,
    ratio: float,
    cutoff: pd.Timestamp,
    case_map: dict[str, tuple[list[str], np.ndarray]],
    activity_to_id: dict[str, int],
    activity_embeddings: torch.Tensor,
    selected_features: Sequence[str],
    seed: int,
    max_pretrain_epochs: int,
    max_gsat_epochs: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    train_all = prefix_df[prefix_df["start_time"] <= cutoff].copy().sort_values("start_time")
    test_df = prefix_df[prefix_df["start_time"] > cutoff].copy().sort_values("start_time")
    val_size = max(512, int(len(train_all) * 0.15))
    val_df = train_all.iloc[-val_size:].copy()
    train_df = train_all.iloc[:-val_size].copy()
    scales = fit_scale_parameters(train_df, case_map, selected_features)

    train_dataset = PrefixGraphDataset(
        train_df, case_map, activity_to_id, selected_features, scales
    )
    val_dataset = PrefixGraphDataset(val_df, case_map, activity_to_id, selected_features, scales)
    test_dataset = PrefixGraphDataset(test_df, case_map, activity_to_id, selected_features, scales)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
        collate_fn=collate_graphs,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, collate_fn=collate_graphs)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, collate_fn=collate_graphs)

    device = torch.device("cpu")
    model = AdaptedTPOP(activity_embeddings, case_feature_dim=len(selected_features)).to(device)
    positive_rate = float(train_df["anomaly_label"].mean())
    pos_weight = torch.tensor([(1.0 - positive_rate) / max(positive_rate, 1e-6)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    history: list[dict[str, float | int | str]] = []

    pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_pretrain_state = copy.deepcopy(model.state_dict())
    best_pretrain_ap = -math.inf
    best_pretrain_epoch = 0
    patience = 4
    for epoch in range(1, max_pretrain_epochs + 1):
        model.train()
        losses: list[float] = []
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            pretrain_optimizer.zero_grad()
            logits = model.forward_base(batch)
            loss = criterion(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            pretrain_optimizer.step()
            losses.append(float(loss.item()))
        _, val_y, val_scores = predict_model(model, val_loader, device, use_gsat=False)
        val_ap = float(average_precision_score(val_y, val_scores))
        history.append(
            {"phase": "pretrain", "epoch": epoch, "loss": float(np.mean(losses)), "val_ap": val_ap}
        )
        if val_ap > best_pretrain_ap + 1e-5:
            best_pretrain_ap = val_ap
            best_pretrain_epoch = epoch
            best_pretrain_state = copy.deepcopy(model.state_dict())
            patience = 4
        else:
            patience -= 1
            if patience <= 0:
                break
    model.load_state_dict(best_pretrain_state)

    gsat_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_gsat_state = copy.deepcopy(model.state_dict())
    best_gsat_ap = -math.inf
    best_gsat_epoch = 0
    patience = 5
    prior_r = 0.6
    for epoch in range(1, max_gsat_epochs + 1):
        model.train()
        losses = []
        prediction_losses = []
        information_losses = []
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            gsat_optimizer.zero_grad()
            logits, attention, _ = model.forward_gsat(batch, training=True)
            prediction_loss = criterion(logits, batch["labels"])
            valid_attention = attention[batch["edge_mask"]]
            if valid_attention.numel() > 0:
                clipped = valid_attention.clamp(1e-6, 1 - 1e-6)
                information_loss = (
                    clipped * torch.log(clipped / prior_r)
                    + (1 - clipped) * torch.log((1 - clipped) / (1 - prior_r))
                ).mean()
            else:
                information_loss = prediction_loss.new_tensor(0.0)
            loss = prediction_loss + information_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            gsat_optimizer.step()
            losses.append(float(loss.item()))
            prediction_losses.append(float(prediction_loss.item()))
            information_losses.append(float(information_loss.item()))
        _, val_y, val_scores = predict_model(model, val_loader, device, use_gsat=True)
        val_ap = float(average_precision_score(val_y, val_scores))
        history.append(
            {
                "phase": "gsat",
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "prediction_loss": float(np.mean(prediction_losses)),
                "information_loss": float(np.mean(information_losses)),
                "val_ap": val_ap,
            }
        )
        if val_ap > best_gsat_ap + 1e-5:
            best_gsat_ap = val_ap
            best_gsat_epoch = epoch
            best_gsat_state = copy.deepcopy(model.state_dict())
            patience = 5
        else:
            patience -= 1
            if patience <= 0:
                break
    model.load_state_dict(best_gsat_state)

    caseids, y_test, scores = predict_model(model, test_loader, device, use_gsat=True)
    metrics: dict[str, object] = evaluate_scores(y_test, scores)
    metrics.update(
        {
            "prefix_ratio": ratio,
            "prefix_percent": f"{int(round(ratio * 100))}%",
            "model": "TPOP-adapted",
            "seed": seed,
            "train_cases": len(train_df),
            "val_cases": len(val_df),
            "test_cases": len(test_df),
            "selected_features": "|".join(selected_features),
            "best_pretrain_epoch": best_pretrain_epoch,
            "best_pretrain_val_ap": best_pretrain_ap,
            "best_gsat_epoch": best_gsat_epoch,
            "best_gsat_val_ap": best_gsat_ap,
        }
    )
    predictions = pd.DataFrame(
        {
            "caseid": caseids,
            "prefix_ratio": ratio,
            "prefix_percent": f"{int(round(ratio * 100))}%",
            "model": "TPOP-adapted",
            "seed": seed,
            "label": y_test.astype(int),
            "score": scores,
        }
    )
    history_frame = pd.DataFrame(history)
    history_frame["prefix_ratio"] = ratio
    history_frame["seed"] = seed
    return metrics, predictions, history_frame


def summarize_tpop(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["precision", "recall", "f1", "auc", "ap", "brier"]
    grouped = seed_metrics.groupby(["prefix_ratio", "prefix_percent", "model"], sort=True)
    rows: list[dict[str, object]] = []
    for keys, group in grouped:
        row: dict[str, object] = {
            "prefix_ratio": keys[0],
            "prefix_percent": keys[1],
            "model": keys[2],
            "n_seeds": len(group),
        }
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_ci95"] = (
                float(1.96 * values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def merge_with_baselines(seed_metrics: pd.DataFrame, summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_csv(BASELINE_DIR / "prefix_10_90_baseline_metrics.csv")
    seed_42 = seed_metrics[seed_metrics["seed"] == 42].copy()
    common_columns = [
        "prefix_ratio",
        "model",
        "precision",
        "recall",
        "f1",
        "auc",
        "ap",
        "brier",
        "train_cases",
        "test_cases",
        "prefix_percent",
    ]
    combined_seed_42 = pd.concat([baseline[common_columns], seed_42[common_columns]], ignore_index=True)
    combined_seed_42 = combined_seed_42.sort_values(["prefix_ratio", "model"]).reset_index(drop=True)

    tpop_mean = summary.rename(
        columns={
            "precision_mean": "precision",
            "recall_mean": "recall",
            "f1_mean": "f1",
            "auc_mean": "auc",
            "ap_mean": "ap",
            "brier_mean": "brier",
        }
    ).copy()
    tpop_mean["model"] = "TPOP-adapted (mean)"
    tpop_mean["train_cases"] = int(seed_metrics["train_cases"].iloc[0])
    tpop_mean["test_cases"] = int(seed_metrics["test_cases"].iloc[0])
    combined_mean = pd.concat([baseline[common_columns], tpop_mean[common_columns]], ignore_index=True)
    combined_mean = combined_mean.sort_values(["prefix_ratio", "model"]).reset_index(drop=True)
    return combined_seed_42, combined_mean


def write_comparison_summary(combined: pd.DataFrame, tpop_summary: pd.DataFrame) -> None:
    average_by_model = (
        combined.groupby("model")[["f1", "ap", "brier"]]
        .mean()
        .sort_values("ap", ascending=False)
        .reset_index()
    )
    average_by_model.to_csv(RESULTS_DIR / "average_performance_by_model.csv", index=False, encoding="utf-8-sig")
    ranks = combined.copy()
    ranks["f1_rank"] = ranks.groupby("prefix_ratio")["f1"].rank(ascending=False, method="min")
    ranks["ap_rank"] = ranks.groupby("prefix_ratio")["ap"].rank(ascending=False, method="min")
    ranks["brier_rank"] = ranks.groupby("prefix_ratio")["brier"].rank(ascending=True, method="min")
    ranks.to_csv(RESULTS_DIR / "comparison_with_ranks.csv", index=False, encoding="utf-8-sig")
    payload = {
        "tpop_mean": tpop_summary.to_dict(orient="records"),
        "average_by_model": average_by_model.to_dict(orient="records"),
    }
    (RESULTS_DIR / "comparison_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TPOP-adapted on the 10%-90% proportional-prefix baseline.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--ratios", nargs="+", type=int, default=list(range(10, 100, 10)))
    parser.add_argument("--pretrain-epochs", type=int, default=15)
    parser.add_argument("--gsat-epochs", type=int, default=20)
    args = parser.parse_args()

    start = time.time()
    activities = collect_activities()
    activity_to_id, activity_embeddings = build_bert_activity_embeddings(activities)
    case_map = build_case_event_map()
    labels = pd.read_csv(BASELINE_DIR / "full_labels.csv", parse_dates=["start_time"])
    cutoff = labels["start_time"].quantile(0.7)

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []
    for percent in args.ratios:
        ratio = percent / 100.0
        prefix_df = pd.read_csv(
            BASELINE_DIR / f"prefix_table_{percent}.csv",
            converters={"tokens_raw": parse_tokens, "start_time": pd.to_datetime},
        )
        train_all = prefix_df[prefix_df["start_time"] <= cutoff].copy().sort_values("start_time")
        val_size = max(512, int(len(train_all) * 0.15))
        feature_train = train_all.iloc[:-val_size].copy()
        selected_features, feature_ranking = select_features_with_xgboost_shap(feature_train)
        feature_ranking["prefix_ratio"] = ratio
        feature_ranking["prefix_percent"] = f"{percent}%"
        feature_frames.append(feature_ranking)
        print(f"ratio={ratio:.1f} selected features: {selected_features}", flush=True)

        for seed in args.seeds:
            metrics, predictions, history = fit_one_seed(
                prefix_df=prefix_df,
                ratio=ratio,
                cutoff=cutoff,
                case_map=case_map,
                activity_to_id=activity_to_id,
                activity_embeddings=activity_embeddings,
                selected_features=selected_features,
                seed=seed,
                max_pretrain_epochs=args.pretrain_epochs,
                max_gsat_epochs=args.gsat_epochs,
            )
            metric_rows.append(metrics)
            prediction_frames.append(predictions)
            history_frames.append(history)
            print(
                f"TPOP ratio={ratio:.1f} seed={seed}: "
                f"F1={metrics['f1']:.4f}, AP={metrics['ap']:.4f}, Brier={metrics['brier']:.4f}",
                flush=True,
            )

    seed_metrics = pd.DataFrame(metric_rows).sort_values(["prefix_ratio", "seed"])
    seed_metrics.to_csv(RESULTS_DIR / "tpop_seed_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        RESULTS_DIR / "tpop_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(history_frames, ignore_index=True).to_csv(
        RESULTS_DIR / "tpop_training_history.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(feature_frames, ignore_index=True).to_csv(
        RESULTS_DIR / "tpop_shap_feature_selection.csv", index=False, encoding="utf-8-sig"
    )

    summary = summarize_tpop(seed_metrics)
    summary.to_csv(RESULTS_DIR / "tpop_summary_metrics.csv", index=False, encoding="utf-8-sig")
    combined_seed_42, combined_mean = merge_with_baselines(seed_metrics, summary)
    combined_seed_42.to_csv(
        RESULTS_DIR / "baseline_models_with_tpop_seed42.csv", index=False, encoding="utf-8-sig"
    )
    combined_mean.to_csv(
        RESULTS_DIR / "baseline_models_with_tpop_mean.csv", index=False, encoding="utf-8-sig"
    )
    write_comparison_summary(combined_mean, summary)

    manifest = {
        "method": "TPOP-adapted",
        "source_method": "Zhang et al., Transparent Business Process Outcome Prediction using GSAT",
        "bert_model": BERT_MODEL,
        "data_path": str(DATA_PATH),
        "baseline_results": str(BASELINE_DIR / "prefix_10_90_baseline_metrics.csv"),
        "time_cutoff": str(cutoff),
        "seeds": args.seeds,
        "prefix_ratios": [percent / 100.0 for percent in args.ratios],
        "feature_selection": "XGBoost exact TreeSHAP contributions; cumulative 90%, minimum 3 and maximum 8 features",
        "split": "earliest 70% train pool; latest 15% of train pool used for validation; latest 30% test",
        "graph": "one directed chain graph per case prefix; events are nodes and directly-follow relations are edges",
        "node_encoding": "frozen BERT activity embedding compressed by label-free PCA, plus visible event-time and selected prefix features",
        "information_prior_r": 0.6,
        "elapsed_seconds": time.time() - start,
    }
    (RESULTS_DIR / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Completed in {manifest['elapsed_seconds']:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
