from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "cross_context" / "tables" / "table1_cross_scene_prefix_performance.csv"
OUTPUT = ROOT / "results" / "paper_tables" / "table_a1_cross_context_performance.csv"


def main() -> None:
    if "--recompute" in sys.argv or not SOURCE.exists():
        for name in ["cross_pgv.py", "cross_tpop.py", "cross_hypotheses.py", "cross_tables.py"]:
            subprocess.run([sys.executable, str(ROOT / "src" / name)], cwd=ROOT, check=True)
    table = pd.read_csv(SOURCE)
    first = table.columns[0]
    table = table[~table[first].astype(str).str.contains("Corporate Account-opening", case=False, na=False)]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
