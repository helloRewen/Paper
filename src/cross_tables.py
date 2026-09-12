from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from external_validation import (              
    LOADERS,
    build_labels,
    deterministic_temporal_sample,
    infer_main_path,
)
from main_checkpoints import (              
    build_fixed_event_table as build_main_fixed_event_table,
    build_natural_time_table as build_main_natural_time_table,
)
from main_baseline import (              
    build_full_labels,
    load_event_log,
)
from external_checkpoints import (
    build_external_fixed_table,
    build_external_natural_table,
)
from cross_hypotheses import (              
    LABELS,
    fit_eval,
)


OUT_DIR = ROOT / "results" / "cross_context" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_MODEL_METRICS = (
    ROOT / "results" / "cross_context" / "tpop" / "cross_scene_models_without_ngram_with_tpop.csv"
)
BEST_CHECKPOINT_METRICS = (
    ROOT / "results" / "cross_context" / "tpop" / "cross_scene_ap_best_with_tpop.csv"
)
RATIO_H23_METRICS = (
    ROOT / "results" / "cross_context" / "signal_analysis" / "h23_all_config_metrics.csv"
)
RATIO_H4_SUMMARY = ROOT / "results" / "cross_context" / "signal_analysis" / "h4_type_stage_summary.csv"

DATASETS = [
    ("corporate_account_opening", "Corporate Account-opening Event Log"),
    ("small_business_credit_approval", "Small-business Credit Approval"),
    ("bpi2017_offer_log", "BPI 2017 Offer"),
    ("bpi2014_rabobank_ict", "BPI 2014 Rabobank ICT"),
    ("bpi2020_request_for_payment", "BPI 2020 RfP"),
]
DATASET_ORDER = {label: i for i, (_key, label) in enumerate(DATASETS)}

STAGES = ["Early", "Middle", "Late"]
STAGE_Q = {"Early": 0.2, "Middle": 0.5, "Late": 0.8}
RATIO_STAGE_LABEL = {"Early": "10%-30%", "Middle": "40%-60%", "Late": "70%-90%"}
STATIC_FIXED = {"Early": 3, "Middle": 5, "Late": 7}
STATIC_NATURAL = {"Early": 24, "Middle": 72, "Late": 168}

SPECS = [
    ("Full", ["path", "time", "structure"]),
    ("No path", ["time", "structure"]),
    ("No time", ["path", "structure"]),
    ("No structure", ["path", "time"]),
    ("Path only", ["path"]),
    ("Time only", ["time"]),
    ("Structure only", ["structure"]),
]
SPEC_SOURCE_NAME = {
    "full": "Full",
    "no_path": "No path",
    "no_time": "No time",
    "no_structure": "No structure",
    "path_only": "Path only",
    "time_only": "Time only",
    "structure_only": "Structure only",
}
SPEC_ORDER = {name: i for i, (name, _groups) in enumerate(SPECS)}
DEV_ORDER = {"Rework deviation": 0, "Rejection deviation": 1, "Cancellation deviation": 2, "Duration deviation": 3}
METHOD_ORDER = {"Proportional prefix": 0, "Fixed-event checkpoint": 1, "Natural-time checkpoint": 2}


@dataclass
class DatasetContext:
    key: str
    label: str
    labels: pd.DataFrame
    cutoff: pd.Timestamp
    main_events: pd.DataFrame | None = None
    traces: object | None = None
    explicit_states: set[str] | None = None
    mandatory_flow: list[str] | None = None


def load_context(dataset_key: str, dataset_label: str) -> DatasetContext:
    if dataset_key == "corporate_account_opening":
        events = load_event_log()
        labels = build_full_labels(events)
        cutoff = labels["start_time"].quantile(0.7)
        return DatasetContext(dataset_key, dataset_label, labels, cutoff, main_events=events)

    traces = deterministic_temporal_sample(LOADERS[dataset_key](), dataset_key)
    labels, explicit_states, _note = build_labels(dataset_key, traces)
    cutoff = labels["start_time"].quantile(0.7)
    mandatory_flow = infer_main_path(traces, labels)
    return DatasetContext(
        dataset_key,
        dataset_label,
        labels,
        cutoff,
        traces=traces,
        explicit_states=explicit_states,
        mandatory_flow=mandatory_flow,
    )


def derive_dynamic_checkpoints(ctx: DatasetContext) -> tuple[dict[str, int], dict[str, int], list[dict]]:
    train = ctx.labels[ctx.labels["start_time"] <= ctx.cutoff].copy()
    fixed: dict[str, int] = {}
    natural: dict[str, int] = {}
    evidence: list[dict] = []
    for stage in STAGES:
        q = STAGE_Q[stage]
        positions = []
        for n_raw in train["full_length"].to_numpy():
            n = int(n_raw)
            proposed = int(math.ceil(n * q))
            positions.append(max(1, min(proposed, n - 1 if n > 1 else 1)))
        position_series = pd.Series(positions, dtype=float)
        fixed_value = int(round(float(position_series.median())))
        duration_raw = float(train["full_duration_hours"].quantile(q))
        natural_value = max(1, int(round(duration_raw)))
        fixed[stage] = fixed_value
        natural[stage] = natural_value
        evidence.append(
            {
                "Dataset": ctx.label,
                "Stage": stage,
                "Training-set process-progress/duration quantile": q,
                "Fixed-event checkpoint": fixed_value,
                "Event-position P25": float(position_series.quantile(0.25)),
                "Median event position": float(position_series.median()),
                "Event-position P75": float(position_series.quantile(0.75)),
                "Natural-time checkpoint (hours)": natural_value,
                "Training-set completion-duration quantile (hours)": duration_raw,
                "Training cases": int(len(train)),
            }
        )
    return fixed, natural, evidence


def build_operational_table(ctx: DatasetContext, method: str, checkpoint: int) -> pd.DataFrame:
    if ctx.key == "corporate_account_opening":
        if method == "Fixed-event checkpoint":
            return build_main_fixed_event_table(ctx.main_events, ctx.labels, checkpoint)
        return build_main_natural_time_table(ctx.main_events, ctx.labels, checkpoint)

    if method == "Fixed-event checkpoint":
        return build_external_fixed_table(
            ctx.traces,
            ctx.labels,
            ctx.mandatory_flow,
            ctx.explicit_states,
            checkpoint,
        )
    return build_external_natural_table(
        ctx.traces,
        ctx.labels,
        ctx.mandatory_flow,
        ctx.explicit_states,
        checkpoint,
    )


def split_table(ctx: DatasetContext, table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = table[table["start_time"] <= ctx.cutoff].copy()
    test = table[table["start_time"] > ctx.cutoff].copy()
    return train, test


def normalize_model(value: str) -> str:
    mapping = {
        "Tabular RF": "Random Forest",
        "tabular_rf": "Random Forest",
        "sequence_ngram_lr": "Sequence N-gram LR",
        "graph_transition_lr": "Graph Transition LR",
        "xgboost": "XGBoost",
    }
    return mapping.get(str(value), str(value))


def select_best_checkpoint_models() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(ALL_MODEL_METRICS, encoding="utf-8-sig")
    expected_groups = len(DATASETS) * (9 + 3 + 3)
    grouped = raw.groupby(["dataset_key", "check_type", "checkpoint_raw"], dropna=False)
    assert grouped.ngroups == expected_groups, (grouped.ngroups, expected_groups)
    assert grouped.size().eq(6).all(), "Each checkpoint must contain six candidate models."
    numeric = ["f1", "ap", "brier", "net_benefit"]
    assert raw[numeric].notna().all().all()
    assert raw["f1"].between(0, 1).all() and raw["ap"].between(0, 1).all()
    assert raw["brier"].between(0, 1).all()

    best = (
        raw.sort_values(
            ["dataset_key", "check_type", "checkpoint_raw", "ap", "brier", "top20_recall"],
            ascending=[True, True, True, False, True, False],
        )
        .groupby(["dataset_key", "check_type", "checkpoint_raw"], as_index=False)
        .head(1)
        .copy()
    )
    saved_best = pd.read_csv(BEST_CHECKPOINT_METRICS, encoding="utf-8-sig")
    check_cols = ["dataset_key", "check_type", "checkpoint_raw", "model", "ap", "f1", "brier", "net_benefit"]
    merged = best[check_cols].merge(
        saved_best[check_cols],
        on=["dataset_key", "check_type", "checkpoint_raw"],
        suffixes=("_new", "_saved"),
        validate="one_to_one",
    )
    assert (merged["model_new"] == merged["model_saved"]).all()
    for col in ["ap", "f1", "brier", "net_benefit"]:
        assert np.allclose(merged[f"{col}_new"], merged[f"{col}_saved"], atol=1e-12)
    return raw, best


def join_values(values, digits: int = 3) -> str:
    return " / ".join(f"{float(v):.{digits}f}" for v in values)


def build_table1(best: pd.DataFrame) -> pd.DataFrame:
    method_specs = [
        ("Proportional prefix", [0.2, 0.3, 0.4, 0.5, 0.6], "20% / 30% / 40% / 50% / 60%"),
        ("Fixed-event checkpoint", [3, 5, 7], "First 3/5/7 events"),
        ("Natural-time checkpoint", [24, 72, 168], "24h / 72h / 168h"),
    ]
    rows = []
    for dataset_key, dataset_label in DATASETS:
        for method, checkpoints, setting in method_specs:
            sub = best[(best["dataset_key"] == dataset_key) & (best["check_type"] == method)].copy()
            selected = []
            for checkpoint in checkpoints:
                hit = sub[np.isclose(sub["checkpoint_raw"].astype(float), float(checkpoint))]
                assert len(hit) == 1, (dataset_label, method, checkpoint, len(hit))
                selected.append(hit.iloc[0])
            rows.append(
                {
                    "Dataset": dataset_label,
                    "Prefix-construction method": method,
                    "Checkpoint setting": setting,
                    "AP-best model at each checkpoint": " / ".join(normalize_model(x["model"]) for x in selected),
                    "F1": join_values(x["f1"] for x in selected),
                    "AP": join_values(x["ap"] for x in selected),
                    "Brier Score": join_values(x["brier"] for x in selected),
                    "Net Benefit": join_values(x["net_benefit"] for x in selected),
                }
            )
    return pd.DataFrame(rows)


def load_ratio_h23_stage_metrics() -> pd.DataFrame:
    ratio = pd.read_csv(RATIO_H23_METRICS, encoding="utf-8-sig")
    ratio["Configuration"] = ratio["Configuration"].map(SPEC_SOURCE_NAME)
    assert ratio["Configuration"].notna().all()
    out = (
        ratio.groupby(["Event log", "Stage", "Configuration"], as_index=False)
        .agg(AP=("ap", "mean"), F1=("f1", "mean"), Brier=("brier", "mean"))
        .rename(columns={"Event log": "Dataset"})
    )
    out["Check method"] = "Proportional prefix"
    out["Checkpoint"] = out["Stage"].map(RATIO_STAGE_LABEL)
    return out[["Dataset", "Check method", "Stage", "Checkpoint", "Configuration", "AP", "F1", "Brier"]]


def compute_dynamic_h23(contexts: list[DatasetContext]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    checkpoint_rows = []
    for ctx in contexts:
        fixed, natural, evidence = derive_dynamic_checkpoints(ctx)
        checkpoint_rows.extend(evidence)
        for method, stage_map in [("Fixed-event checkpoint", fixed), ("Natural-time checkpoint", natural)]:
            for stage in STAGES:
                checkpoint = stage_map[stage]
                table = build_operational_table(ctx, method, checkpoint)
                train, test = split_table(ctx, table)
                for spec_name, groups in SPECS:
                    metrics = fit_eval(train, test, groups)
                    assert metrics is not None, (ctx.label, method, stage, spec_name)
                    rows.append(
                        {
                            "Dataset": ctx.label,
                            "Check method": method,
                            "Stage": stage,
                            "Checkpoint": f"Event {checkpoint}" if method == "Fixed-event checkpoint" else f"{checkpoint}h",
                            "Configuration": spec_name,
                            "AP": metrics["ap"],
                            "F1": metrics["f1"],
                            "Brier": metrics["brier"],
                        }
                    )
                print(f"H23 {ctx.label} {method} {stage} completed")
    return pd.DataFrame(rows), pd.DataFrame(checkpoint_rows)


def build_table2(ratio_stage: pd.DataFrame, dynamic: pd.DataFrame) -> pd.DataFrame:
    all_metrics = pd.concat([ratio_stage, dynamic], ignore_index=True)
    assert len(all_metrics) == len(DATASETS) * 3 * 3 * len(SPECS)
    assert not all_metrics.duplicated(["Dataset", "Check method", "Stage", "Configuration"]).any()
    rows = []
    for (dataset, method, config), sub in all_metrics.groupby(["Dataset", "Check method", "Configuration"]):
        by = sub.set_index("Stage")
        assert set(by.index) == set(STAGES)
        checkpoint_text = "/".join(str(by.loc[s, "Checkpoint"]) for s in STAGES)
        method_display = method if method == "Proportional prefix" else f"{method}({checkpoint_text})"
        row = {"Dataset": dataset, "Check method": method_display, "Model configuration": config}
        for stage in STAGES:
            row[f"{stage}AP"] = float(by.loc[stage, "AP"])
            row[f"{stage}F1"] = float(by.loc[stage, "F1"])
            row[f"{stage}Brier"] = float(by.loc[stage, "Brier"])
        rows.append(row)
    out = pd.DataFrame(rows)
    out["_d"] = out["Dataset"].map(DATASET_ORDER)
    out["_m"] = out["Check method"].map(
        lambda x: 0 if x == "Proportional prefix" else (1 if str(x).startswith("Fixed event") else 2)
    )
    out["_s"] = out["Model configuration"].map(SPEC_ORDER)
    return out.sort_values(["_d", "_m", "_s"]).drop(columns=["_d", "_m", "_s"])


def build_table3(ratio_stage: pd.DataFrame, dynamic: pd.DataFrame) -> pd.DataFrame:
    all_metrics = pd.concat([ratio_stage, dynamic], ignore_index=True)
    rows = []
    for (dataset, method, stage), sub in all_metrics.groupby(["Dataset", "Check method", "Stage"]):
        by = sub.set_index("Configuration")
        assert {x[0] for x in SPECS}.issubset(by.index)
        full = by.loc["Full"]
        values = {
            "Activity path": float(full["AP"] - by.loc["No path", "AP"]),
            "Waiting time": float(full["AP"] - by.loc["No time", "AP"]),
            "Structural deviation": float(full["AP"] - by.loc["No structure", "AP"]),
        }
        main_signal = max(values, key=values.get)
        judgement = (
            "Supports H2" if stage == "Early" and main_signal == "Activity path" else
            "H2 not supported" if stage == "Early" else
            "Provides stage-specific evidence for H3" if main_signal in {"Waiting time", "Structural deviation"} else
            "No stage-specific evidence for H3"
        )
        rows.append(
            {
                "Dataset": dataset,
                "Check method": method,
                "Stage": stage,
                "Checkpoint basis": str(full["Checkpoint"]),
                "Delta AP: activity path": values["Activity path"],
                "Delta AP: waiting time": values["Waiting time"],
                "Delta AP: structural deviation": values["Structural deviation"],
                "Dominant signal": main_signal,
                "Stage-specific assessment": judgement,
            }
        )
    out = pd.DataFrame(rows)
    out["_d"] = out["Dataset"].map(DATASET_ORDER)
    out["_m"] = out["Check method"].map(METHOD_ORDER)
    out["_s"] = out["Stage"].map({s: i for i, s in enumerate(STAGES)})
    out = out.sort_values(["_d", "_m", "_s"]).drop(columns=["_d", "_m", "_s"])
    assert len(out) == len(DATASETS) * 3 * 3
    return out


def load_ratio_h4() -> pd.DataFrame:
    ratio = pd.read_csv(RATIO_H4_SUMMARY, encoding="utf-8-sig")
    rows = []
    for (dataset, deviation), subset in ratio.groupby(["Event log", "Deviation type"], sort=False):
        by_stage = subset.set_index("Stage")
        if not set(STAGES).issubset(by_stage.index):
            continue
        ap_values = [float(by_stage.loc[stage, "ap"]) for stage in STAGES]
        rows.append(
            {
                "Dataset": dataset,
                "Check method": "Proportional prefix",
                "Deviation type": deviation,
                "Early AP": ap_values[0],
                "Middle AP": ap_values[1],
                "Late AP": ap_values[2],
                "Early Recall": float(by_stage.loc["Early", "recall"]),
                "Middle Recall": float(by_stage.loc["Middle", "recall"]),
                "Late Recall": float(by_stage.loc["Late", "recall"]),
                "Mean Brier Score": float(subset["brier"].mean()),
                "AP range": float(max(ap_values) - min(ap_values)),
            }
        )
    return pd.DataFrame(rows)


def compute_static_h4(contexts: list[DatasetContext]) -> pd.DataFrame:
    rows = []
    for ctx in contexts:
        for method, checkpoints in [
            ("Fixed-event checkpoint", STATIC_FIXED),
            ("Natural-time checkpoint", STATIC_NATURAL),
        ]:
            stage_metrics: dict[tuple[str, str], dict] = {}
            for stage in STAGES:
                checkpoint = checkpoints[stage]
                table = build_operational_table(ctx, method, checkpoint)
                train, test = split_table(ctx, table)
                for label_col, label_name, _group in LABELS:
                    if label_col not in table.columns:
                        continue
                    if int(train[label_col].sum()) < 5 or int(test[label_col].sum()) < 5:
                        continue
                    metrics = fit_eval(train, test, ["path", "time", "structure"], label_col=label_col)
                    if metrics is not None:
                        stage_metrics[(stage, label_name)] = metrics
                print(f"H4 {ctx.label} {method} {stage} completed")

            for _label_col, label_name, _group in LABELS:
                if not all((stage, label_name) in stage_metrics for stage in STAGES):
                    continue
                aps = [float(stage_metrics[(stage, label_name)]["ap"]) for stage in STAGES]
                recalls = [float(stage_metrics[(stage, label_name)]["recall"]) for stage in STAGES]
                briers = [float(stage_metrics[(stage, label_name)]["brier"]) for stage in STAGES]
                setting = "3/5/7" if method == "Fixed-event checkpoint" else "24/72/168h"
                rows.append(
                    {
                        "Dataset": ctx.label,
                        "Check method": f"{method}({setting})",
                        "Deviation type": label_name,
                        "Early AP": aps[0],
                        "Middle AP": aps[1],
                        "Late AP": aps[2],
                        "Early Recall": recalls[0],
                        "Middle Recall": recalls[1],
                        "Late Recall": recalls[2],
                        "Mean Brier Score": float(np.mean(briers)),
                        "AP range": float(max(aps) - min(aps)),
                    }
                )
    return pd.DataFrame(rows)


def build_table4(ratio: pd.DataFrame, operational: pd.DataFrame) -> pd.DataFrame:
    out = pd.concat([ratio, operational], ignore_index=True)
    out["_d"] = out["Dataset"].map(DATASET_ORDER)
    out["_m"] = out["Check method"].map(
        lambda x: 0 if x == "Proportional prefix" else (1 if str(x).startswith("Fixed event") else 2)
    )
    out["_v"] = out["Deviation type"].map(DEV_ORDER)
    out = out.sort_values(["_d", "_m", "_v"]).drop(columns=["_d", "_m", "_v"])
    metric_cols = [c for c in out.columns if c not in {"Dataset", "Check method", "Deviation type"}]
    assert out[metric_cols].notna().all().all()
    assert out[[c for c in metric_cols if c != "AP range"]].apply(lambda s: s.between(0, 1).all()).all()
    return out


def build_quality_audit(
    contexts: list[DatasetContext],
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    table3: pd.DataFrame,
    table4: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for ctx in contexts:
        train = ctx.labels[ctx.labels["start_time"] <= ctx.cutoff]
        test = ctx.labels[ctx.labels["start_time"] > ctx.cutoff]
        rows.append(
            {
                "Dataset": ctx.label,
                "Cases": int(len(ctx.labels)),
                "Training cases": int(len(train)),
                "Test cases": int(len(test)),
                "Temporal split point": str(ctx.cutoff),
                "Training-set composite-deviation rate": float(train["anomaly_label"].mean()),
                "Test-set composite-deviation rate": float(test["anomaly_label"].mean()),
                "Table 1 rows": int((table1["Dataset"] == ctx.label).sum()),
                "Table 2 rows": int((table2["Dataset"] == ctx.label).sum()),
                "Table 3 rows": int((table3["Dataset"] == ctx.label).sum()),
                "Table 4 rows": int((table4["Dataset"] == ctx.label).sum()),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(frames: dict[str, pd.DataFrame]) -> None:
    for name, frame in frames.items():
        frame.to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    _raw_models, best = select_best_checkpoint_models()
    table1 = build_table1(best)

    contexts = [load_context(key, label) for key, label in DATASETS]
    ratio_h23 = load_ratio_h23_stage_metrics()
    dynamic_h23, checkpoints = compute_dynamic_h23(contexts)
    table3 = build_table3(ratio_h23, dynamic_h23)
    assert len(table1) == len(DATASETS) * 3
    assert len(table3) == len(DATASETS) * 3 * 3

    write_outputs(
        {
            "table1_cross_scene_prefix_performance": table1,
            "table3_cross_scene_stage_signal_marginal_contribution": table3,
        }
    )
    print("\nDynamic checkpoints")
    print(checkpoints[["Dataset", "Stage", "Fixed-event checkpoint", "Natural-time checkpoint (hours)"]].to_string(index=False))
    print("\nOutput row counts")
    for name, frame in {
        "table1": table1,
        "table3": table3,
    }.items():
        print(name, len(frame))
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
