from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from external_validation import compress_repeats


def leakage_control_len(seq: list[str], proposed: int, explicit_states: set[str]) -> int:
    proposed = max(1, min(proposed, len(seq) - 1 if len(seq) > 1 else 1))
    first_explicit = None
    for index, activity in enumerate(seq):
        if activity in explicit_states:
            first_explicit = index
            break
    if first_explicit is not None:
        proposed = min(proposed, max(1, first_explicit))
    return max(1, proposed)


def external_prefix_row_from_len(trace, label: Dict[str, object], prefix_len: int, mandatory_flow: list[str]) -> Dict[str, object]:
    seq_prefix = trace.activities[:prefix_len]
    time_prefix = trace.times[:prefix_len]
    seq_comp = compress_repeats(seq_prefix)
    if len(time_prefix) > 1:
        gaps = np.diff(np.array(time_prefix, dtype="datetime64[m]")).astype("timedelta64[m]").astype(int) / 60.0
        gaps = gaps.tolist()
    else:
        gaps = []
    transition_counts: Dict[str, float] = {}
    for source, target in zip(seq_comp[:-1], seq_comp[1:]):
        key = f"{source}__{target}"
        transition_counts[key] = transition_counts.get(key, 0.0) + 1.0
    total = float(sum(transition_counts.values())) or 1.0
    row = {
        "caseid": trace.caseid,
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
        "repeat_state_count": float(sum(1 for index in range(1, len(seq_prefix)) if seq_prefix[index] == seq_prefix[index - 1])),
        "skipped_mandatory_count": float(sum(step not in seq_prefix for step in mandatory_flow)),
        "first_activity": seq_prefix[0],
        "last_activity": seq_prefix[-1],
        "path_text": " ".join(seq_comp),
        "transition_features": {key: value / total for key, value in transition_counts.items()},
    }
    row.update(label)
    return row


def build_external_fixed_table(traces, labels: pd.DataFrame, mandatory_flow: list[str], explicit_states: set[str], checkpoint: int) -> pd.DataFrame:
    label_map = labels.set_index("caseid").to_dict("index")
    rows = []
    for trace in traces:
        if trace.caseid not in label_map:
            continue
        prefix_len = leakage_control_len(trace.activities, checkpoint, explicit_states)
        rows.append(external_prefix_row_from_len(trace, label_map[trace.caseid], prefix_len, mandatory_flow))
    return pd.DataFrame(rows)


def build_external_natural_table(traces, labels: pd.DataFrame, mandatory_flow: list[str], explicit_states: set[str], hours: int) -> pd.DataFrame:
    label_map = labels.set_index("caseid").to_dict("index")
    rows = []
    horizon = pd.Timedelta(hours=hours)
    for trace in traces:
        if trace.caseid not in label_map or not trace.times:
            continue
        start = trace.times[0]
        if trace.times[-1] <= start + horizon:
            continue
        allowed = [index for index, timestamp in enumerate(trace.times) if timestamp <= start + horizon]
        proposed = 1 if not allowed else max(1, max(allowed) + 1)
        prefix_len = leakage_control_len(trace.activities, proposed, explicit_states)
        rows.append(external_prefix_row_from_len(trace, label_map[trace.caseid], prefix_len, mandatory_flow))
    return pd.DataFrame(rows)
