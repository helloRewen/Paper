from __future__ import annotations

import gzip
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DATA_DIR = ROOT / "data" / "public"
PRIVATE_DATA_DIR = ROOT / "data" / "private"
RESULTS_DIR = ROOT / "results" / "cross_context" / "validation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREFIX_RATIOS = [0.3, 0.5, 0.7]
MAX_CASES: Dict[str, int] = {}

NUMERIC_COLS = [
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
CATEGORICAL_COLS = ["first_activity", "last_activity"]


@dataclass
class TraceRecord:
    caseid: str
    activities: List[str]
    times: List[pd.Timestamp]
    attrs: Dict[str, object] = field(default_factory=dict)


def slugify(text: object) -> str:
    value = "" if pd.isna(text) else str(text)
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unknown"


def compress_repeats(seq: List[str]) -> List[str]:
    out: List[str] = []
    for item in seq:
        if not out or out[-1] != item:
            out.append(item)
    return out


def parse_timestamp(value: object) -> Optional[pd.Timestamp]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(None)
    return pd.Timestamp(ts)


def xes_value(elem: ET.Element) -> object:
    tag = elem.tag.split("}")[-1]
    val = elem.attrib.get("value")
    if tag == "boolean":
        return str(val).lower() == "true"
    if tag == "int":
        try:
            return int(val)
        except Exception:
            return val
    if tag == "float":
        try:
            return float(val)
        except Exception:
            return val
    return val


def iter_xes_traces(path: Path, keep_lifecycle: Optional[str] = None) -> Iterable[TraceRecord]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for _, elem in ET.iterparse(f, events=("end",)):
            if elem.tag.split("}")[-1] != "trace":
                continue
            trace_attrs: Dict[str, object] = {}
            events: List[tuple[str, pd.Timestamp]] = []
            for child in list(elem):
                child_tag = child.tag.split("}")[-1]
                if child_tag == "event":
                    event_attrs = {e.attrib.get("key"): xes_value(e) for e in list(child)}
                    lifecycle = str(event_attrs.get("lifecycle:transition", "")).lower()
                    if keep_lifecycle and lifecycle and lifecycle != keep_lifecycle.lower():
                        continue
                    act = slugify(event_attrs.get("concept:name"))
                    ts = parse_timestamp(event_attrs.get("time:timestamp"))
                    if ts is not None:
                        events.append((act, ts))
                else:
                    key = child.attrib.get("key")
                    if key:
                        trace_attrs[key] = xes_value(child)
            elem.clear()
            if not events:
                continue
            events.sort(key=lambda x: x[1])
            caseid = str(trace_attrs.get("concept:name") or trace_attrs.get("case:concept:name") or trace_attrs.get("id") or len(events))
            yield TraceRecord(caseid=caseid, activities=[a for a, _ in events], times=[t for _, t in events], attrs=trace_attrs)


def load_small_business_credit_approval() -> List[TraceRecord]:
    path = PRIVATE_DATA_DIR / "small_business_credit_approval.xlsx"
    df = pd.read_excel(path, sheet_name=0, dtype=str)
    df.columns = [str(column).strip().upper() for column in df.columns]
    required = ["SERIALNO", "OBJECTNO", "PHASENO", "PHASENAME", "BEGINTIME", "ENDTIME"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["OBJECTNO"] = df["OBJECTNO"].astype(str).str.strip()
    df["PHASENO"] = df["PHASENO"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(4)
    df["PHASENAME"] = df["PHASENAME"].astype(str).str.strip()
    df["SERIALNO"] = df["SERIALNO"].astype(str).str.strip()
    df["BEGINTIME"] = pd.to_datetime(df["BEGINTIME"], errors="coerce")
    df["ENDTIME"] = pd.to_datetime(df["ENDTIME"], errors="coerce")
    invalid = df["OBJECTNO"].eq("") | df["PHASENO"].eq("") | df["BEGINTIME"].isna() | df["ENDTIME"].isna() | df["ENDTIME"].lt(df["BEGINTIME"])
    invalid_cases = set(df.loc[invalid, "OBJECTNO"].dropna().astype(str))
    df = df[~df["OBJECTNO"].isin(invalid_cases)].copy()
    df = df.sort_values(["OBJECTNO", "BEGINTIME", "ENDTIME", "SERIALNO"])
    traces: List[TraceRecord] = []
    for caseid, group in df.groupby("OBJECTNO", sort=False):
        traces.append(
            TraceRecord(
                str(caseid),
                group["PHASENO"].tolist(),
                group["BEGINTIME"].tolist(),
                {
                    "phase_names": group["PHASENAME"].tolist(),
                    "end_times": group["ENDTIME"].tolist(),
                },
            )
        )
    return traces


def load_bpi2014_rabobank_ict() -> List[TraceRecord]:
    df = pd.read_csv(PUBLIC_DATA_DIR / "bpi2014_rabobank_ict.csv", sep=";")
    df["dateTime"] = pd.to_datetime(df["DateStamp"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Incident ID", "IncidentActivity_Type", "dateTime"])
    df["activity_alias"] = df["IncidentActivity_Type"].map(slugify)
    df = df.sort_values(["Incident ID", "dateTime"])
    traces: List[TraceRecord] = []
    for caseid, g in df.groupby("Incident ID", sort=False):
        traces.append(TraceRecord(str(caseid), g["activity_alias"].tolist(), g["dateTime"].tolist(), {}))
    return traces


def load_xes_file(filename: str, keep_lifecycle: Optional[str] = None) -> List[TraceRecord]:
    return list(iter_xes_traces(PUBLIC_DATA_DIR / filename, keep_lifecycle=keep_lifecycle))


def deterministic_temporal_sample(traces: List[TraceRecord], dataset: str) -> List[TraceRecord]:
    limit = MAX_CASES.get(dataset)
    if not limit or len(traces) <= limit:
        return traces
    ordered = sorted(traces, key=lambda t: t.times[0])
    idx = np.linspace(0, len(ordered) - 1, num=limit, dtype=int)
    return [ordered[i] for i in idx]


def trace_duration_hours(t: TraceRecord) -> float:
    end_times = t.attrs.get("end_times")
    end_time = max(pd.to_datetime(end_times)) if end_times else t.times[-1]
    return float((end_time - t.times[0]).total_seconds() / 3600.0)


def has_any(seq: List[str], keywords: List[str]) -> bool:
    return any(any(k in act for k in keywords) for act in seq)


def build_labels(dataset: str, traces: List[TraceRecord]) -> tuple[pd.DataFrame, set[str], str]:
    rows = []
    durations = np.array([trace_duration_hours(t) for t in traces], dtype=float)
    if dataset == "small_business_credit_approval" and len(durations):
        starts = pd.Series([t.times[0] for t in traces])
        cutoff = starts.quantile(0.7)
        p95 = float(np.quantile(durations[starts.le(cutoff).to_numpy()], 0.95))
    else:
        p95 = float(np.quantile(durations, 0.95)) if len(durations) else 0.0
    all_activities = {a for t in traces for a in t.activities}

    explicit_states: set[str] = set()
    label_note = ""
    if dataset == "small_business_credit_approval":
        explicit_states = {"3000", "3040", "5000", "8000"}
    elif dataset == "bpi2017_offer_log":
        explicit_states = {"o_refused", "o_cancelled"}
    elif dataset == "bpi2020_request_for_payment":
        explicit_states = {a for a in all_activities if any(k in a for k in ["rejected", "cancelled", "cancel"])}

    for t, dur in zip(traces, durations):
        seq = t.activities
        vc = pd.Series(seq).value_counts()
        rework = 0
        rejected = 0
        cancelled = 0
        exception = 0

        if dataset == "small_business_credit_approval":
            rework = int(any(code in seq for code in ["3000", "3040", "5000"]))
            rejected = int(seq[-1] == "8000")
            label_note = "Small-business credit approval: duration p95 plus return, reassessment, or rejection states."
        elif dataset == "bpi2017_offer_log":
            rework = int(vc.get("o_returned", 0) > 1 or vc.get("o_create_offer", 0) > 1)
            rejected = int("o_refused" in seq)
            cancelled = int("o_cancelled" in seq)
            label_note = "BPI 2017 Offer: duration p95 plus offer returned/refused/cancelled states."
        elif dataset == "bpi2014_rabobank_ict":
            rework = int(vc.get("reassignment", 0) >= 2)
            label_note = "BPI 2014 Rabobank ICT: duration p95 plus repeated reassignment as coordination-friction outcome."
        elif dataset == "bpi2020_request_for_payment":
            rejected = int(has_any(seq, ["rejected"]))
            cancelled = int(has_any(seq, ["cancelled", "cancel"]))
            rework = int(vc.max() > 1)
            label_note = "BPI 2020 Request for Payment: duration p95 plus rejected/cancelled/repeated approval states."

        duration_anomaly = int(dur > p95)
        anomaly_label = int(duration_anomaly or rework or rejected or cancelled or exception)
        rows.append(
            {
                "caseid": t.caseid,
                "start_time": t.times[0],
                "start_month": t.times[0].to_period("M").strftime("%Y-%m"),
                "full_length": len(seq),
                "full_duration_hours": dur,
                "duration_anomaly": duration_anomaly,
                "rework": rework,
                "rejected": rejected,
                "cancelled": cancelled,
                "external_exception": exception,
                "anomaly_label": anomaly_label,
            }
        )
    return pd.DataFrame(rows), explicit_states, label_note


def infer_main_path(traces: List[TraceRecord], labels: pd.DataFrame, max_len: int = 8) -> List[str]:
    label_map = labels.set_index("caseid")["anomaly_label"].to_dict()
    normal_traces = [t for t in traces if label_map.get(t.caseid) == 0]
    base = normal_traces or traces
    activity_positions: Dict[str, List[int]] = {}
    threshold = max(3, int(0.15 * len(base)))
    for t in base:
        seen = {}
        for pos, act in enumerate(compress_repeats(t.activities)):
            seen.setdefault(act, pos)
        for act, pos in seen.items():
            activity_positions.setdefault(act, []).append(pos)
    candidates = [
        (act, len(pos), float(np.median(pos)))
        for act, pos in activity_positions.items()
        if len(pos) >= threshold
    ]
    candidates.sort(key=lambda x: (-x[1], x[2], x[0]))
    top = sorted(candidates[:max_len], key=lambda x: (x[2], -x[1], x[0]))
    return [act for act, _, _ in top]


def choose_prefix_len(seq: List[str], ratio: float, explicit_states: set[str]) -> int:
    n = len(seq)
    base = max(1, int(math.ceil(n * ratio)))
    base = min(base, n - 1) if n > 1 else 1
    first_explicit = None
    for idx, act in enumerate(seq):
        if act in explicit_states:
            first_explicit = idx
            break
    if first_explicit is not None:
        base = min(base, max(1, first_explicit))
    return max(1, base)


def prefix_row(t: TraceRecord, label: Dict[str, object], ratio: float, mandatory_flow: List[str], explicit_states: set[str]) -> Dict[str, object]:
    prefix_len = choose_prefix_len(t.activities, ratio, explicit_states)
    seq_prefix = t.activities[:prefix_len]
    time_prefix = t.times[:prefix_len]
    seq_comp = compress_repeats(seq_prefix)
    if len(time_prefix) > 1:
        gaps = np.diff(np.array(time_prefix, dtype="datetime64[m]")).astype("timedelta64[m]").astype(int) / 60.0
        gaps = gaps.tolist()
    else:
        gaps = []
    transitions = list(zip(seq_comp[:-1], seq_comp[1:]))
    transition_counts: Dict[str, float] = {}
    for src, dst in transitions:
        key = f"{src}__{dst}"
        transition_counts[key] = transition_counts.get(key, 0.0) + 1.0
    total = float(sum(transition_counts.values())) or 1.0
    transition_probs = {k: v / total for k, v in transition_counts.items()}
    row = {
        "caseid": t.caseid,
        "prefix_ratio": ratio,
        "prefix_event_count": prefix_len,
        "sequence_length": len(seq_prefix),
        "compressed_length": len(seq_comp),
        "observed_duration_hours": float((time_prefix[-1] - time_prefix[0]).total_seconds() / 3600.0) if len(time_prefix) > 1 else 0.0,
        "gap_mean_hours": float(np.mean(gaps)) if gaps else 0.0,
        "gap_max_hours": float(np.max(gaps)) if gaps else 0.0,
        "gap_std_hours": float(np.std(gaps)) if gaps else 0.0,
        "cumulative_wait_hours": float(np.sum(gaps)) if gaps else 0.0,
        "repeat_state_count": float(sum(1 for i in range(1, len(seq_prefix)) if seq_prefix[i] == seq_prefix[i - 1])),
        "skipped_mandatory_count": float(sum(step not in seq_prefix for step in mandatory_flow)),
        "first_activity": seq_prefix[0],
        "last_activity": seq_prefix[-1],
        "path_text": " ".join(seq_comp),
        "transition_features": transition_probs,
    }
    row.update(label)
    return row


def build_prefix_table(traces: List[TraceRecord], labels: pd.DataFrame, ratio: float, mandatory_flow: List[str], explicit_states: set[str]) -> pd.DataFrame:
    label_map = labels.set_index("caseid").to_dict("index")
    return pd.DataFrame([prefix_row(t, label_map[t.caseid], ratio, mandatory_flow, explicit_states) for t in traces if t.caseid in label_map])


def build_tabular_model() -> Pipeline:
    prep = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_COLS),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLS,
            ),
        ]
    )
    return Pipeline(
        [
            ("prep", prep),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=160,
                    min_samples_leaf=4,
                    class_weight="balanced_subsample",
                    random_state=SEED,
                    n_jobs=1,
                ),
            ),
        ]
    )


def build_sequence_model() -> Pipeline:
    return Pipeline(
        [
            ("vec", CountVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 3), min_df=2, max_features=50000)),
            ("clf", LogisticRegression(max_iter=1200, class_weight="balanced", solver="liblinear", random_state=SEED)),
        ]
    )


def scores_to_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.5
    ap = average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else float(np.mean(y_true))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "ap": float(ap),
        "threshold": float(threshold),
    }


def choose_threshold(y_train: np.ndarray, train_scores: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        f1 = scores_to_metrics(y_train, train_scores, float(t))["f1"]
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def evaluate_models(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    y_train = train_df["anomaly_label"].to_numpy()
    y_test = test_df["anomaly_label"].to_numpy()
    out: Dict[str, Dict[str, float]] = {}

    tab = build_tabular_model()
    cols = NUMERIC_COLS + CATEGORICAL_COLS
    tab.fit(train_df[cols], y_train)
    tr = tab.predict_proba(train_df[cols])[:, 1]
    te = tab.predict_proba(test_df[cols])[:, 1]
    out["tabular_rf"] = scores_to_metrics(y_test, te, choose_threshold(y_train, tr))

    seq = build_sequence_model()
    seq.fit(train_df["path_text"], y_train)
    tr = seq.predict_proba(train_df["path_text"])[:, 1]
    te = seq.predict_proba(test_df["path_text"])[:, 1]
    out["sequence_ngram_lr"] = scores_to_metrics(y_test, te, choose_threshold(y_train, tr))

    gvec = DictVectorizer(sparse=True)
    xg_train = gvec.fit_transform(train_df["transition_features"].tolist())
    xg_test = gvec.transform(test_df["transition_features"].tolist())
    if xg_train.shape[1] == 0 or len(np.unique(y_train)) < 2:
        tr = np.repeat(float(np.mean(y_train)), len(y_train))
        te = np.repeat(float(np.mean(y_train)), len(y_test))
    else:
        gclf = LogisticRegression(max_iter=1200, class_weight="balanced", solver="liblinear", random_state=SEED)
        gclf.fit(xg_train, y_train)
        tr = gclf.predict_proba(xg_train)[:, 1]
        te = gclf.predict_proba(xg_test)[:, 1]
    out["graph_transition_lr"] = scores_to_metrics(y_test, te, choose_threshold(y_train, tr))

    return out


def topk_recall(y_true: np.ndarray, scores: np.ndarray, frac: float = 0.1) -> float:
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return 0.0
    k = max(1, int(math.ceil(len(y_true) * frac)))
    order = np.argsort(scores)[::-1][:k]
    return float(np.sum(y_true[order]) / np.sum(y_true))


def run_wait_rule(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float]:
    y_train = train_df["anomaly_label"].to_numpy()
    y_test = test_df["anomaly_label"].to_numpy()
    train_scores = train_df["cumulative_wait_hours"].to_numpy(dtype=float)
    test_scores = test_df["cumulative_wait_hours"].to_numpy(dtype=float)
    mx = float(np.max(train_scores)) if len(train_scores) else 0.0
    if mx > 0:
        train_scores = train_scores / mx
        test_scores = np.clip(test_scores / mx, 0, 1)
    t = choose_threshold(y_train, train_scores)
    return scores_to_metrics(y_test, test_scores, t)


def run_friction_rule(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float]:
    y_train = train_df["anomaly_label"].to_numpy()
    y_test = test_df["anomaly_label"].to_numpy()
    train_scores = (
        train_df["repeat_state_count"].to_numpy(dtype=float)
        + train_df["skipped_mandatory_count"].to_numpy(dtype=float)
        + (train_df["compressed_length"].to_numpy(dtype=float) > 4).astype(float)
    )
    test_scores = (
        test_df["repeat_state_count"].to_numpy(dtype=float)
        + test_df["skipped_mandatory_count"].to_numpy(dtype=float)
        + (test_df["compressed_length"].to_numpy(dtype=float) > 4).astype(float)
    )
    mx = float(np.max(train_scores)) if len(train_scores) else 0.0
    if mx > 0:
        train_scores = train_scores / mx
        test_scores = np.clip(test_scores / mx, 0, 1)
    t = choose_threshold(y_train, train_scores)
    return scores_to_metrics(y_test, test_scores, t)


def dataset_summary(dataset: str, traces: List[TraceRecord], labels: pd.DataFrame, sampled_from: int, label_note: str) -> Dict[str, object]:
    return {
        "dataset": dataset,
        "cases": len(traces),
        "sampled_from_cases": sampled_from,
        "events": int(sum(len(t.activities) for t in traces)),
        "activities": int(len({a for t in traces for a in t.activities})),
        "start_min": str(min(t.times[0] for t in traces)),
        "start_max": str(max(t.times[0] for t in traces)),
        "anomaly_rate": float(labels["anomaly_label"].mean()),
        "duration_rate": float(labels["duration_anomaly"].mean()),
        "rework_rate": float(labels["rework"].mean()),
        "rejected_rate": float(labels["rejected"].mean()),
        "cancelled_rate": float(labels["cancelled"].mean()),
        "external_exception_rate": float(labels["external_exception"].mean()),
        "label_note": label_note,
    }


def markdown_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    if df.empty:
        return "No rows."
    view = df.copy()
    if max_rows is not None:
        view = view.head(max_rows)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.4f}")
    headers = [str(c) for c in view.columns]
    rows = [[str(v) for v in row] for row in view.to_numpy()]
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep_line = "| " + " | ".join("-" * w for w in widths) + " |"
    body = ["| " + " | ".join(cell.ljust(w) for cell, w in zip(row, widths)) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body])


LOADERS = {
    "small_business_credit_approval": load_small_business_credit_approval,
    "bpi2014_rabobank_ict": load_bpi2014_rabobank_ict,
    "bpi2017_offer_log": lambda: load_xes_file("bpi2017_offer_log.xes.gz", keep_lifecycle="complete"),
    "bpi2020_request_for_payment": lambda: load_xes_file("bpi2020_request_for_payment.xes.gz", keep_lifecycle=None),
}


def main() -> None:
    all_metrics = []
    summaries = []
    label_notes = []
    best_rows = []

    for dataset, loader in LOADERS.items():
        print(f"Loading {dataset}...")
        traces = loader()
        sampled_from = len(traces)
        traces = deterministic_temporal_sample(traces, dataset)
        labels, explicit_states, label_note = build_labels(dataset, traces)
        mandatory_flow = infer_main_path(traces, labels)
        summaries.append(dataset_summary(dataset, traces, labels, sampled_from, label_note))
        label_notes.append({"dataset": dataset, "explicit_states": ", ".join(sorted(explicit_states)), "mandatory_flow": " > ".join(mandatory_flow), "label_note": label_note})
        print(f"  cases={len(traces)} events={sum(len(t.activities) for t in traces)} anomaly={labels['anomaly_label'].mean():.3f}")

        for ratio in PREFIX_RATIOS:
            prefix_df = build_prefix_table(traces, labels, ratio, mandatory_flow, explicit_states)
            cutoff = prefix_df["start_time"].quantile(0.7)
            train_df = prefix_df[prefix_df["start_time"] <= cutoff].copy()
            test_df = prefix_df[prefix_df["start_time"] > cutoff].copy()
            if len(train_df) < 50 or len(test_df) < 20 or train_df["anomaly_label"].nunique() < 2 or test_df["anomaly_label"].nunique() < 2:
                print(f"  skip ratio={ratio}: insufficient split")
                continue
            model_metrics = evaluate_models(train_df, test_df)
            model_metrics["wait_rule"] = run_wait_rule(train_df, test_df)
            model_metrics["friction_rule"] = run_friction_rule(train_df, test_df)

            for model, metrics in model_metrics.items():
                row = {
                    "dataset": dataset,
                    "prefix_ratio": ratio,
                    "model": model,
                    "train_cases": len(train_df),
                    "test_cases": len(test_df),
                    "cutoff_time": str(cutoff),
                    **metrics,
                }
                all_metrics.append(row)
                print(f"  ratio={ratio:.1f} {model}: f1={metrics['f1']:.3f} auc={metrics['auc']:.3f} ap={metrics['ap']:.3f}")

    metrics_df = pd.DataFrame(all_metrics)
    summary_df = pd.DataFrame(summaries)
    label_df = pd.DataFrame(label_notes)
    if not metrics_df.empty:
        best = metrics_df.sort_values(["dataset", "prefix_ratio", "f1"], ascending=[True, True, False]).groupby(["dataset", "prefix_ratio"], as_index=False).head(1)
        best_dataset = metrics_df.sort_values(["dataset", "f1"], ascending=[True, False]).groupby("dataset", as_index=False).head(1)
    else:
        best = pd.DataFrame()
        best_dataset = pd.DataFrame()

    summary_df.to_csv(RESULTS_DIR / "dataset_summary.csv", index=False, encoding="utf-8-sig")
    label_df.to_csv(RESULTS_DIR / "label_definitions.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(RESULTS_DIR / "temporal_holdout_metrics.csv", index=False, encoding="utf-8-sig")
    best.to_csv(RESULTS_DIR / "best_by_dataset_prefix.csv", index=False, encoding="utf-8-sig")
    best_dataset.to_csv(RESULTS_DIR / "best_by_dataset.csv", index=False, encoding="utf-8-sig")

    report = [
        "# Cross-context validation report",
        "",
        "## Dataset summary",
        markdown_table(summary_df),
        "",
        "## Label definitions",
        markdown_table(label_df),
        "",
        "## Best model by dataset and prefix",
        markdown_table(best) if not best.empty else "No valid metrics.",
        "",
        "## Best result by dataset",
        markdown_table(best_dataset) if not best_dataset.empty else "No valid metrics.",
        "",
        "Note: metrics use a 70/30 chronological holdout within each dataset. F1 thresholds are selected on the training split only. BPI 2019 is a deterministic temporal sample if the full log exceeds the configured case limit.",
    ]
    (RESULTS_DIR / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Saved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
