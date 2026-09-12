from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from external_validation import (              
    LOADERS,
    build_labels,
    build_prefix_table as build_external_prefix_table,
    compress_repeats,
    deterministic_temporal_sample,
    infer_main_path,
)
from main_baseline import (              
    ACTIVITY_ALIAS,
    SEED,
    build_full_labels,
    build_prefix_table as build_main_prefix_table,
)


OUT_DIR = ROOT / "results" / "cross_context" / "signal_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RATIOS = [i / 10 for i in range(1, 10)]
STAGE_MAP = {
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
STAGE_ORDER = {"Early": 0, "Middle": 1, "Late": 2}

DATASETS = [
    ("corporate_account_opening", "Corporate Account-opening Event Log"),
    ("small_business_credit_approval", "Small-business Credit Approval"),
    ("bpi2017_offer_log", "BPI 2017 Offer"),
    ("bpi2014_rabobank_ict", "BPI 2014 Rabobank ICT"),
    ("bpi2020_request_for_payment", "BPI 2020 RfP"),
]

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

LABELS = [
    ("duration_anomaly", "Duration deviation", "duration"),
    ("rework", "Rework deviation", "visible"),
    ("rejected", "Rejection deviation", "visible"),
    ("cancelled", "Cancellation deviation", "visible"),
]


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
        struct_num = self.struct_num_imputer.fit_transform(train_df[STRUCTURE_NUMERIC_FEATURES])
        self.struct_num_scaler.fit(struct_num)
        struct_cat = self.struct_cat_imputer.fit_transform(train_df[STRUCTURE_CATEGORICAL_FEATURES])
        self.struct_cat_encoder.fit(struct_cat)
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
        return hstack(matrices, format="csr")


def eval_scores(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_pred = (scores >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    prevalence = float(np.mean(y_true)) if len(y_true) else 0.0
    ap = float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else prevalence
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "ap": ap,
        "ap_lift": ap - prevalence,
        "brier": float(brier_score_loss(y_true, scores)),
        "prevalence": prevalence,
        "test_cases": int(len(y_true)),
        "test_positives": int(np.sum(y_true)),
    }


def fit_eval(train_df: pd.DataFrame, test_df: pd.DataFrame, groups: list[str], label_col: str = "anomaly_label") -> dict[str, float] | None:
    if train_df[label_col].nunique() < 2 or test_df[label_col].nunique() < 2:
        return None
    encoder = SignalEncoder().fit(train_df)
    x_train = encoder.transform(train_df, groups)
    x_test = encoder.transform(test_df, groups)
    y_train = train_df[label_col].to_numpy(dtype=int)
    y_test = test_df[label_col].to_numpy(dtype=int)
    clf = LogisticRegression(max_iter=1600, class_weight="balanced", solver="liblinear", random_state=SEED)
    clf.fit(x_train, y_train)
    scores = clf.predict_proba(x_test)[:, 1]
    return eval_scores(y_test, scores)


def load_main_prefix_tables() -> tuple[dict[float, pd.DataFrame], pd.DataFrame, float, dict[str, float]]:
    df = pd.read_csv(ROOT / "data" / "private" / "corporate_account_opening.csv", encoding="gbk")
    df["dateTime"] = pd.to_datetime(df["dateTime"])
    df["activity_alias"] = df["activity"].map(ACTIVITY_ALIAS).fillna("unknown_activity")
    df = df.sort_values(["caseid", "dateTime"]).reset_index(drop=True)
    seqs = [tuple(g["activity_alias"].tolist()) for _, g in df.groupby("caseid", sort=False)]
    variant = variant_stats(seqs)
    labels = build_full_labels(df)
    cutoff = labels["start_time"].quantile(0.7)
    tables = {ratio: build_main_prefix_table(df, labels, ratio) for ratio in RATIOS}
    return tables, labels, cutoff, variant


def load_external_prefix_tables(dataset_key: str) -> tuple[dict[float, pd.DataFrame], pd.DataFrame, float, dict[str, float]]:
    traces = deterministic_temporal_sample(LOADERS[dataset_key](), dataset_key)
    seqs = [tuple(t.activities) for t in traces]
    variant = variant_stats(seqs)
    labels, explicit_states, _label_note = build_labels(dataset_key, traces)
    mandatory_flow = infer_main_path(traces, labels)
    tables = {
        ratio: build_external_prefix_table(traces, labels, ratio, mandatory_flow, explicit_states)
        for ratio in RATIOS
    }
    cutoff = labels["start_time"].quantile(0.7)
    return tables, labels, cutoff, variant


def variant_stats(seqs: list[tuple[str, ...]]) -> dict[str, float]:
    cases = len(seqs)
    variants = len(set(seqs))
    compressed = [tuple(compress_repeats(list(seq))) for seq in seqs]
    return {
        "cases": cases,
        "variants": variants,
        "variant_case_ratio": variants / cases if cases else 0.0,
        "compressed_variant_case_ratio": len(set(compressed)) / cases if cases else 0.0,
    }


def compute_h23(dataset_key: str, dataset_label: str, tables: dict[float, pd.DataFrame], cutoff) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = [
        ("full", ["path", "time", "structure"]),
        ("no_path", ["time", "structure"]),
        ("no_time", ["path", "structure"]),
        ("no_structure", ["path", "time"]),
        ("path_only", ["path"]),
        ("time_only", ["time"]),
        ("structure_only", ["structure"]),
    ]
    rows = []
    for ratio, table in tables.items():
        train_df = table[table["start_time"] <= cutoff].copy()
        test_df = table[table["start_time"] > cutoff].copy()
        for spec, groups in specs:
            metrics = fit_eval(train_df, test_df, groups)
            if metrics is None:
                continue
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "Event log": dataset_label,
                    "prefix_ratio": ratio,
                    "Prefix ratio": f"{int(ratio * 100)}%",
                    "Stage": STAGE_MAP[ratio],
                    "Configuration": spec,
                    **metrics,
                }
            )
            print(f"H23 {dataset_label} {ratio:.1f} {spec} AP={metrics['ap']:.3f}")
    metrics_df = pd.DataFrame(rows)
    marginal_rows = []
    for ratio, sub in metrics_df.groupby("prefix_ratio"):
        by = sub.set_index("Configuration")
        if not {"full", "no_path", "no_time", "no_structure", "path_only", "time_only", "structure_only"}.issubset(by.index):
            continue
        full = by.loc["full"]
        marginal_rows.append(
            {
                "dataset_key": dataset_key,
                "Event log": dataset_label,
                "prefix_ratio": ratio,
                "Prefix ratio": f"{int(ratio * 100)}%",
                "Stage": STAGE_MAP[ratio],
                "full_ap": float(full["ap"]),
                "full_f1": float(full["f1"]),
                "full_brier": float(full["brier"]),
                "delta_ap_path": float(full["ap"] - by.loc["no_path", "ap"]),
                "delta_ap_time": float(full["ap"] - by.loc["no_time", "ap"]),
                "delta_ap_structure": float(full["ap"] - by.loc["no_structure", "ap"]),
                "delta_brier_path": float(by.loc["no_path", "brier"] - full["brier"]),
                "delta_brier_time": float(by.loc["no_time", "brier"] - full["brier"]),
                "delta_brier_structure": float(by.loc["no_structure", "brier"] - full["brier"]),
                "path_only_ap": float(by.loc["path_only", "ap"]),
                "time_only_ap": float(by.loc["time_only", "ap"]),
                "structure_only_ap": float(by.loc["structure_only", "ap"]),
                "path_advantage_single": float(by.loc["path_only", "ap"] - max(by.loc["time_only", "ap"], by.loc["structure_only", "ap"])),
                "time_structure_advantage_delta": float(
                    (full["ap"] - by.loc["no_time", "ap"])
                    + (full["ap"] - by.loc["no_structure", "ap"])
                    - (full["ap"] - by.loc["no_path", "ap"])
                ),
            }
        )
    return metrics_df, pd.DataFrame(marginal_rows)


def compute_h4(dataset_key: str, dataset_label: str, tables: dict[float, pd.DataFrame], cutoff) -> pd.DataFrame:
    rows = []
    for ratio, table in tables.items():
        train_df = table[table["start_time"] <= cutoff].copy()
        test_df = table[table["start_time"] > cutoff].copy()
        for label_col, label_name, label_group in LABELS:
            if label_col not in table.columns:
                continue
            if train_df[label_col].sum() < 5 or test_df[label_col].sum() < 5:
                continue
            metrics = fit_eval(train_df, test_df, ["path", "time", "structure"], label_col=label_col)
            if metrics is None:
                continue
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "Event log": dataset_label,
                    "prefix_ratio": ratio,
                    "Prefix ratio": f"{int(ratio * 100)}%",
                    "Stage": STAGE_MAP[ratio],
                    "label_col": label_col,
                    "Deviation type": label_name,
                    "Deviation group": label_group,
                    **metrics,
                }
            )
            print(f"H4 {dataset_label} {ratio:.1f} {label_name} AP={metrics['ap']:.3f}")
    return pd.DataFrame(rows)


def stage_summary_h23(marginal: pd.DataFrame) -> pd.DataFrame:
    stage = (
        marginal.groupby(["dataset_key", "Event log", "Stage"], as_index=False)
        .agg(
            delta_ap_path=("delta_ap_path", "mean"),
            delta_ap_time=("delta_ap_time", "mean"),
            delta_ap_structure=("delta_ap_structure", "mean"),
            delta_brier_path=("delta_brier_path", "mean"),
            delta_brier_time=("delta_brier_time", "mean"),
            delta_brier_structure=("delta_brier_structure", "mean"),
            path_only_ap=("path_only_ap", "mean"),
            time_only_ap=("time_only_ap", "mean"),
            structure_only_ap=("structure_only_ap", "mean"),
            path_advantage_single=("path_advantage_single", "mean"),
            time_structure_advantage_delta=("time_structure_advantage_delta", "mean"),
        )
        .sort_values(["dataset_key", "Stage"], key=lambda s: s.map(STAGE_ORDER).fillna(s))
    )
    return stage


def dataset_summary_h23(stage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset_key, label), sub in stage.groupby(["dataset_key", "Event log"]):
        by = sub.set_index("Stage")
        early = by.loc["Early"] if "Early" in by.index else None
        late = by.loc["Late"] if "Late" in by.index else None
        if early is None or late is None:
            continue
        rows.append(
            {
                "dataset_key": dataset_key,
                "Event log": label,
                "Early path-only advantage": float(early["path_advantage_single"]),
                "Early delta AP: activity path": float(early["delta_ap_path"]),
                "Early delta AP: waiting time": float(early["delta_ap_time"]),
                "Early delta AP: structural deviation": float(early["delta_ap_structure"]),
                "H2 path-only support": bool(early["path_advantage_single"] > 0),
                "H2 marginal-contribution support": bool(early["delta_ap_path"] > max(early["delta_ap_time"], early["delta_ap_structure"])),
                "Late-stage relative gain of time and structure": float(late["time_structure_advantage_delta"] - early["time_structure_advantage_delta"]),
                "H3 support": bool(late["time_structure_advantage_delta"] > early["time_structure_advantage_delta"]),
            }
        )
    return pd.DataFrame(rows)


def h4_summaries(h4: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    type_stage = (
        h4.groupby(["dataset_key", "Event log", "Deviation type", "Deviation group", "Stage"], as_index=False)
        .agg(
            ap=("ap", "mean"),
            ap_lift=("ap_lift", "mean"),
            f1=("f1", "mean"),
            recall=("recall", "mean"),
            brier=("brier", "mean"),
            prevalence=("prevalence", "mean"),
            test_positives=("test_positives", "mean"),
        )
    )
    rows = []
    for (dataset_key, label), sub in type_stage.groupby(["dataset_key", "Event log"]):
        early = sub[sub["Stage"] == "Early"]
        duration = early[early["Deviation group"] == "duration"]
        visible = early[early["Deviation group"] == "visible"]
        all_type = sub.groupby(["Deviation type", "Deviation group"], as_index=False).agg(ap_mean=("ap", "mean"), ap_std=("ap", "std"), ap_lift_mean=("ap_lift", "mean"))
        if duration.empty or visible.empty:
            visible_early_ap_lift = np.nan
            duration_early_ap_lift = np.nan
            h4_visible = False
        else:
            visible_early_ap_lift = float(visible["ap_lift"].mean())
            duration_early_ap_lift = float(duration["ap_lift"].mean())
            h4_visible = bool(visible_early_ap_lift > duration_early_ap_lift)
        rows.append(
            {
                "dataset_key": dataset_key,
                "Event log": label,
                "Early visible-deviation mean AP lift": visible_early_ap_lift,
                "Early duration-deviation AP lift": duration_early_ap_lift,
                "Early visible-deviation advantage": visible_early_ap_lift - duration_early_ap_lift
                if pd.notna(visible_early_ap_lift) and pd.notna(duration_early_ap_lift)
                else np.nan,
                "Range of mean AP across deviation types": float(all_type["ap_mean"].max() - all_type["ap_mean"].min()) if not all_type.empty else np.nan,
                "Visible deviations outperform duration deviation in the early stage": h4_visible,
                "Number of testable deviation types": int(all_type["Deviation type"].nunique()),
            }
        )
    return type_stage, pd.DataFrame(rows)


def correlations(merged: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        valid = merged[["variant_case_ratio", col]].dropna()
        if len(valid) < 3:
            continue
        rows.append(
            {
                "Metric": col,
                "Pearson r": valid.corr(method="pearson").iloc[0, 1],
                "Spearman rho": valid.corr(method="spearman").iloc[0, 1],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    variants = []
    h23_metrics = []
    h23_marginal = []
    h4_metrics = []

    for dataset_key, dataset_label in DATASETS:
        print(f"\n=== {dataset_label} ===")
        if dataset_key == "corporate_account_opening":
            tables, _labels, cutoff, variant = load_main_prefix_tables()
        else:
            tables, _labels, cutoff, variant = load_external_prefix_tables(dataset_key)
        variants.append({"dataset_key": dataset_key, "Event log": dataset_label, **variant})
        m, marginal = compute_h23(dataset_key, dataset_label, tables, cutoff)
        h23_metrics.append(m)
        h23_marginal.append(marginal)
        h4_metrics.append(compute_h4(dataset_key, dataset_label, tables, cutoff))

    variant_df = pd.DataFrame(variants)
    h23_metrics_df = pd.concat(h23_metrics, ignore_index=True)
    h23_marginal_df = pd.concat(h23_marginal, ignore_index=True)
    h23_stage = stage_summary_h23(h23_marginal_df)
    h23_dataset = dataset_summary_h23(h23_stage).merge(variant_df, on=["dataset_key", "Event log"], how="left")

    h4_df = pd.concat(h4_metrics, ignore_index=True)
    h4_type_stage, h4_dataset = h4_summaries(h4_df)
    h4_dataset = h4_dataset.merge(variant_df, on=["dataset_key", "Event log"], how="left")

    h23_corr = correlations(
        h23_dataset,
        ["Early path-only advantage", "Early delta AP: activity path", "Late-stage relative gain of time and structure"],
    )
    h4_corr = correlations(
        h4_dataset,
        ["Early visible-deviation advantage", "Range of mean AP across deviation types"],
    )

    for name, df in [
        ("variant_case_ratio.csv", variant_df),
        ("h23_all_config_metrics.csv", h23_metrics_df),
        ("h23_marginal_by_prefix.csv", h23_marginal_df),
        ("h23_stage_summary.csv", h23_stage),
        ("h23_dataset_support_summary.csv", h23_dataset),
        ("h23_variant_correlations.csv", h23_corr),
        ("h4_type_prefix_metrics.csv", h4_df),
        ("h4_type_stage_summary.csv", h4_type_stage),
        ("h4_dataset_support_summary.csv", h4_dataset),
        ("h4_variant_correlations.csv", h4_corr),
    ]:
        df.to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")

    print("\n=== H2/H3 dataset support summary ===")
    print(
        h23_dataset[
            [
                "Event log",
                "variant_case_ratio",
                "Early path-only advantage",
                "Early delta AP: activity path",
                "Early delta AP: waiting time",
                "Early delta AP: structural deviation",
                "H2 path-only support",
                "H2 marginal-contribution support",
                "Late-stage relative gain of time and structure",
                "H3 support",
            ]
        ].sort_values("variant_case_ratio").to_string(
            index=False,
            formatters={
                "variant_case_ratio": lambda x: f"{x:.4f}",
                "Early path-only advantage": lambda x: f"{x:.3f}",
                "Early delta AP: activity path": lambda x: f"{x:.3f}",
                "Early delta AP: waiting time": lambda x: f"{x:.3f}",
                "Early delta AP: structural deviation": lambda x: f"{x:.3f}",
                "Late-stage relative gain of time and structure": lambda x: f"{x:.3f}",
            },
        )
    )

    print("\n=== H2/H3 correlations with variant/case ratio ===")
    print(
        h23_corr.to_string(
            index=False,
            formatters={"Pearson r": lambda x: f"{x:.3f}", "Spearman rho": lambda x: f"{x:.3f}"},
        )
    )

    print("\n=== H4 dataset support summary ===")
    print(
        h4_dataset[
            [
                "Event log",
                "variant_case_ratio",
                "Early visible-deviation mean AP lift",
                "Early duration-deviation AP lift",
                "Early visible-deviation advantage",
                "Range of mean AP across deviation types",
                "Visible deviations outperform duration deviation in the early stage",
                "Number of testable deviation types",
            ]
        ].sort_values("variant_case_ratio").to_string(
            index=False,
            formatters={
                "variant_case_ratio": lambda x: f"{x:.4f}",
                "Early visible-deviation mean AP lift": lambda x: f"{x:.3f}",
                "Early duration-deviation AP lift": lambda x: f"{x:.3f}",
                "Early visible-deviation advantage": lambda x: f"{x:.3f}",
                "Range of mean AP across deviation types": lambda x: f"{x:.3f}",
            },
        )
    )

    print("\n=== H4 correlations with variant/case ratio ===")
    print(
        h4_corr.to_string(
            index=False,
            formatters={"Pearson r": lambda x: f"{x:.3f}", "Spearman rho": lambda x: f"{x:.3f}"},
        )
    )
    print(f"\noutputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
