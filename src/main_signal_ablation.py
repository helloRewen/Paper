from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from main_baseline import (
    SEED,
    build_full_labels,
    build_prefix_table,
    load_event_log,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "main" / "h23_signal_stage_ablation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREFIX_RATIOS = [i / 10 for i in range(1, 10)]

TIME_FEATURES = [
    "observed_duration_hours",
    "gap_mean_hours",
    "gap_max_hours",
    "gap_std_hours",
    "cumulative_wait_hours",
]

STRUCTURE_NUMERIC_FEATURES = [
    "sequence_length",
    "compressed_length",
    "repeat_state_count",
    "skipped_mandatory_count",
    "prefix_event_count",
]

STRUCTURE_CATEGORICAL_FEATURES = ["first_activity", "last_activity"]


def eval_scores(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_pred = (scores >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.5,
        "ap": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float(np.mean(y_true)),
        "brier": float(brier_score_loss(y_true, scores)),
    }


class SignalEncoder:
    def __init__(self) -> None:
        self.path_vec = CountVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 3), min_df=2)
        self.transition_vec = DictVectorizer(sparse=True)
        self.time_imputer = SimpleImputer(strategy="median")
        self.time_scaler = StandardScaler(with_mean=False)
        self.struct_num_imputer = SimpleImputer(strategy="median")
        self.struct_num_scaler = StandardScaler(with_mean=False)
        self.struct_cat_imputer = SimpleImputer(strategy="most_frequent")
        self.struct_cat_encoder = OneHotEncoder(handle_unknown="ignore")

    def fit(self, train_df: pd.DataFrame) -> "SignalEncoder":
        self.path_vec.fit(train_df["path_text"])
        self.transition_vec.fit(train_df["transition_features"].tolist())
        time_imp = self.time_imputer.fit_transform(train_df[TIME_FEATURES])
        self.time_scaler.fit(time_imp)
        struct_num_imp = self.struct_num_imputer.fit_transform(train_df[STRUCTURE_NUMERIC_FEATURES])
        self.struct_num_scaler.fit(struct_num_imp)
        struct_cat_imp = self.struct_cat_imputer.fit_transform(train_df[STRUCTURE_CATEGORICAL_FEATURES])
        self.struct_cat_encoder.fit(struct_cat_imp)
        return self

    def transform_group(self, df: pd.DataFrame, group: str):
        if group == "path":
            text = self.path_vec.transform(df["path_text"])
            transitions = self.transition_vec.transform(df["transition_features"].tolist())
            return hstack([text, transitions], format="csr")
        if group == "time":
            values = self.time_imputer.transform(df[TIME_FEATURES])
            return csr_matrix(self.time_scaler.transform(values))
        if group == "structure":
            numeric = self.struct_num_imputer.transform(df[STRUCTURE_NUMERIC_FEATURES])
            numeric = csr_matrix(self.struct_num_scaler.transform(numeric))
            categorical = self.struct_cat_imputer.transform(df[STRUCTURE_CATEGORICAL_FEATURES])
            categorical = self.struct_cat_encoder.transform(categorical)
            return hstack([numeric, categorical], format="csr")
        raise ValueError(group)

    def transform(self, df: pd.DataFrame, groups: Iterable[str]):
        matrices = [self.transform_group(df, group) for group in groups]
        if not matrices:
            raise ValueError("At least one signal group is required.")
        return hstack(matrices, format="csr")


def fit_eval(train_df: pd.DataFrame, test_df: pd.DataFrame, groups: list[str]) -> dict[str, float]:
    encoder = SignalEncoder().fit(train_df)
    x_train = encoder.transform(train_df, groups)
    x_test = encoder.transform(test_df, groups)
    y_train = train_df["anomaly_label"].to_numpy()
    y_test = test_df["anomaly_label"].to_numpy()
    clf = LogisticRegression(max_iter=1600, class_weight="balanced", solver="liblinear", random_state=SEED)
    clf.fit(x_train, y_train)
    scores = clf.predict_proba(x_test)[:, 1]
    return eval_scores(y_test, scores)


def main() -> None:
    df = load_event_log()
    labels = build_full_labels(df)
    cutoff = labels["start_time"].quantile(0.7)

    specs = [
        ("full", ["path", "time", "structure"]),
        ("path_only", ["path"]),
        ("time_only", ["time"]),
        ("structure_only", ["structure"]),
        ("no_path", ["time", "structure"]),
        ("no_time", ["path", "structure"]),
        ("no_structure", ["path", "time"]),
    ]

    rows = []
    for ratio in PREFIX_RATIOS:
        prefix_df = build_prefix_table(df, labels, ratio)
        train_df = prefix_df[prefix_df["start_time"] <= cutoff].copy()
        test_df = prefix_df[prefix_df["start_time"] > cutoff].copy()
        for spec_name, groups in specs:
            metrics = fit_eval(train_df, test_df, groups)
            metrics.update(
                {
                    "prefix_ratio": ratio,
                    "prefix_percent": f"{int(ratio * 100)}%",
                    "spec": spec_name,
                    "groups": "+".join(groups),
                    "train_cases": int(len(train_df)),
                    "test_cases": int(len(test_df)),
                }
            )
            rows.append(metrics)
            print(f"ratio={ratio:.1f} spec={spec_name} f1={metrics['f1']:.4f} ap={metrics['ap']:.4f}")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(RESULTS_DIR / "h23_signal_stage_ablation_metrics.csv", index=False, encoding="utf-8-sig")

    marginal_rows = []
    for ratio in PREFIX_RATIOS:
        sub = metrics_df[metrics_df["prefix_ratio"] == ratio].set_index("spec")
        full = sub.loc["full"]
        marginal_rows.append(
            {
                "prefix_ratio": ratio,
                "prefix_percent": f"{int(ratio * 100)}%",
                "full_f1": float(full["f1"]),
                "full_ap": float(full["ap"]),
                "delta_ap_path": float(full["ap"] - sub.loc["no_path", "ap"]),
                "delta_ap_time": float(full["ap"] - sub.loc["no_time", "ap"]),
                "delta_ap_structure": float(full["ap"] - sub.loc["no_structure", "ap"]),
                "delta_f1_path": float(full["f1"] - sub.loc["no_path", "f1"]),
                "delta_f1_time": float(full["f1"] - sub.loc["no_time", "f1"]),
                "delta_f1_structure": float(full["f1"] - sub.loc["no_structure", "f1"]),
                "delta_brier_path": float(sub.loc["no_path", "brier"] - full["brier"]),
                "delta_brier_time": float(sub.loc["no_time", "brier"] - full["brier"]),
                "delta_brier_structure": float(sub.loc["no_structure", "brier"] - full["brier"]),
                "path_only_ap": float(sub.loc["path_only", "ap"]),
                "time_only_ap": float(sub.loc["time_only", "ap"]),
                "structure_only_ap": float(sub.loc["structure_only", "ap"]),
                "path_only_f1": float(sub.loc["path_only", "f1"]),
                "time_only_f1": float(sub.loc["time_only", "f1"]),
                "structure_only_f1": float(sub.loc["structure_only", "f1"]),
                "path_only_brier": float(sub.loc["path_only", "brier"]),
                "time_only_brier": float(sub.loc["time_only", "brier"]),
                "structure_only_brier": float(sub.loc["structure_only", "brier"]),
            }
        )
    marginal_df = pd.DataFrame(marginal_rows)
    marginal_df.to_csv(RESULTS_DIR / "h23_signal_marginal_contribution.csv", index=False, encoding="utf-8-sig")

    stage_map = {
        0.1: "Early",
        0.2: "Early",
        0.3: "Early",
        0.4: "Middle",
        0.5: "Middle",
        0.6: "Middle",
        0.7: "Late",
        0.8: "Late",
        0.9: "Late",
    }
    stage_order = {"Early": 0, "Middle": 1, "Late": 2}
    marginal_df["stage"] = marginal_df["prefix_ratio"].map(stage_map)
    stage_summary = (
        marginal_df.groupby("stage", as_index=False)
        .agg(
            prefix_range=("prefix_percent", lambda x: " / ".join(x)),
            full_ap=("full_ap", "mean"),
            full_f1=("full_f1", "mean"),
            delta_ap_path=("delta_ap_path", "mean"),
            delta_ap_time=("delta_ap_time", "mean"),
            delta_ap_structure=("delta_ap_structure", "mean"),
            delta_f1_path=("delta_f1_path", "mean"),
            delta_f1_time=("delta_f1_time", "mean"),
            delta_f1_structure=("delta_f1_structure", "mean"),
            delta_brier_path=("delta_brier_path", "mean"),
            delta_brier_time=("delta_brier_time", "mean"),
            delta_brier_structure=("delta_brier_structure", "mean"),
            path_only_ap=("path_only_ap", "mean"),
            time_only_ap=("time_only_ap", "mean"),
            structure_only_ap=("structure_only_ap", "mean"),
        )
    )
    stage_summary = stage_summary.sort_values(by="stage", key=lambda s: s.map(stage_order))
    stage_summary.to_csv(RESULTS_DIR / "h23_signal_stage_summary.csv", index=False, encoding="utf-8-sig")
    print(marginal_df.to_string(index=False))


if __name__ == "__main__":
    main()
