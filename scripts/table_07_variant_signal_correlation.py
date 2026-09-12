from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "cross_context" / "tables" / "table_07_correlation_analysis.csv"
OUTPUT = ROOT / "results" / "paper_tables" / "table_07_variant_signal_correlation.csv"


def main() -> None:
    if "--recompute" in sys.argv or not SOURCE.exists():
        for name in ["cross_pgv.py", "cross_tpop.py", "cross_hypotheses.py", "cross_tables.py", "cross_correlation.py"]:
            subprocess.run([sys.executable, str(ROOT / "src" / name)], cwd=ROOT, check=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
