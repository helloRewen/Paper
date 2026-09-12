from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from main_baseline import build_full_labels, load_event_log


OUTPUT = ROOT / "results" / "paper_tables" / "table_01_dataset_statistics.csv"


def main() -> None:
    event_log = load_event_log()
    labels = build_full_labels(event_log)
    lengths = labels["full_length"]
    durations = labels["full_duration_hours"]
    rows = [
        ("Events / cases / activities", f"{len(event_log):,} / {len(labels):,} / {event_log['activity_alias'].nunique():,}"),
        ("Events per case, mean / median", f"{lengths.mean():.2f} / {lengths.median():.0f}"),
        ("Events per case, IQR / P95 / max", f"{lengths.quantile(0.25):.0f}-{lengths.quantile(0.75):.0f} / {lengths.quantile(0.95):.0f} / {lengths.max():.0f}"),
        ("Total duration (hours), mean / median", f"{durations.mean():.2f} / {durations.median():.2f}"),
        ("Total duration (hours), IQR / P95 / max", f"{durations.quantile(0.25):.2f}-{durations.quantile(0.75):.2f} / {durations.quantile(0.95):.2f} / {durations.max():,.2f}"),
        ("Rejected cases / share", f"{labels['rejected'].sum():,} / {labels['rejected'].mean():.2%}"),
        ("Withdrawn cases / share", f"{labels['cancelled'].sum():,} / {labels['cancelled'].mean():.2%}"),
        ("Rework cases / share", f"{labels['rework'].sum():,} / {labels['rework'].mean():.2%}"),
    ]
    table = pd.DataFrame(rows, columns=["Indicator", "Value"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
