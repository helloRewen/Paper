from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    "table_01_dataset_statistics.py",
    "table_02_operational_actionability.py",
    "table_03_feature_ablation_performance.py",
    "table_04_signal_marginal_contribution.py",
    "table_05_deviation_type_performance.py",
    "table_06_pgv_evaluation.py",
    "table_07_variant_signal_correlation.py",
    "table_a1_cross_context_performance.py",
    "table_a2_cross_context_pgv.py",
    "verify_package.py",
]


def main() -> None:
    for name in STEPS:
        subprocess.run([sys.executable, str(ROOT / "scripts" / name)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
