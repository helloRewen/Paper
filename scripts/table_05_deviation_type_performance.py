from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "main" / "checkpoint_stage_h23_h4_tables" / "h4_stage_comparison_table.csv"
OUTPUT = ROOT / "results" / "paper_tables" / "table_05_deviation_type_performance.csv"


def main() -> None:
    if "--recompute" in sys.argv or not SOURCE.exists():
        for name in ["main_baseline.py", "main_checkpoints.py", "main_signal_ablation.py", "main_hypothesis_tables.py"]:
            subprocess.run([sys.executable, str(ROOT / "src" / name)], cwd=ROOT, check=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
