from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
import tokenize


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PUBLIC_DATA = {
    "bpi2014_rabobank_ict.csv",
    "bpi2017_offer_log.xes.gz",
    "bpi2020_request_for_payment.xes.gz",
}
REQUIRED_PRIVATE_FILES = {"README.md"}
OPTIONAL_PRIVATE_DATA = {
    "corporate_account_opening.csv",
    "small_business_credit_approval.xlsx",
}
EXPECTED_TABLE_SCRIPTS = {
    "table_01_dataset_statistics.py",
    "table_02_operational_actionability.py",
    "table_03_feature_ablation_performance.py",
    "table_04_signal_marginal_contribution.py",
    "table_05_deviation_type_performance.py",
    "table_06_pgv_evaluation.py",
    "table_07_variant_signal_correlation.py",
    "table_a1_cross_context_performance.py",
    "table_a2_cross_context_pgv.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    public_files = {path.name for path in (ROOT / "data" / "public").iterdir() if path.is_file()}
    if public_files != EXPECTED_PUBLIC_DATA:
        raise RuntimeError(f"Unexpected public-data files: {sorted(public_files ^ EXPECTED_PUBLIC_DATA)}")
    private_files = {path.name for path in (ROOT / "data" / "private").iterdir() if path.is_file()}
    if not REQUIRED_PRIVATE_FILES.issubset(private_files):
        raise RuntimeError(f"Missing private-data documentation: {sorted(REQUIRED_PRIVATE_FILES - private_files)}")
    unexpected_private = private_files - REQUIRED_PRIVATE_FILES - OPTIONAL_PRIVATE_DATA
    if unexpected_private:
        raise RuntimeError(f"Unexpected private-data files: {sorted(unexpected_private)}")
    table_scripts = {path.name for path in (ROOT / "scripts").glob("table_*.py")}
    if table_scripts != EXPECTED_TABLE_SCRIPTS:
        raise RuntimeError(f"Unexpected table scripts: {sorted(table_scripts ^ EXPECTED_TABLE_SCRIPTS)}")
    for path in [*sorted((ROOT / "scripts").glob("*.py")), *sorted((ROOT / "src").glob("*.py"))]:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        comments = [
            token
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.COMMENT
        ]
        if comments:
            raise RuntimeError(f"Comment tokens found in {path.name}")
    manifest_path = ROOT / "docs" / "file_manifest.csv"
    files = [path for path in ROOT.rglob("*") if path.is_file() and path != manifest_path]
    rows = [
        {
            "relative_path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(files)
    ]
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"files={len(rows)}")
    print(f"bytes={sum(row['size_bytes'] for row in rows)}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
