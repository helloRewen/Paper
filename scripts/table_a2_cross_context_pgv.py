from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "cross_context" / "tpop" / "cross_scene_pgv_summary_with_tpop.csv"
OUTPUT = ROOT / "results" / "paper_tables" / "table_a2_cross_context_pgv.csv"


def main() -> None:
    if "--recompute" in sys.argv or not SOURCE.exists():
        for name in ["cross_pgv.py", "cross_tpop.py"]:
            subprocess.run([sys.executable, str(ROOT / "src" / name)], cwd=ROOT, check=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
