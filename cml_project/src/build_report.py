from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = project_root / "artifacts"
    metrics_path = artifacts_dir / "metrics.json"
    validation_path = artifacts_dir / "data_validation.json"
    metrics_md_path = artifacts_dir / "metrics.txt"

    if not metrics_path.exists():
        raise FileNotFoundError("Missing artifacts/metrics.json. Run `python src/train.py` first.")
    if not validation_path.exists():
        raise FileNotFoundError(
            "Missing artifacts/data_validation.json. Run `python src/validate_data.py` first."
        )
    if not metrics_md_path.exists():
        raise FileNotFoundError("Missing artifacts/metrics.txt. Run `python src/train.py` first.")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    metrics_md = metrics_md_path.read_text(encoding="utf-8").strip()

    report_lines = [
        "# CML Pipeline Lab Report",
        "",
        "## Pipeline Jobs",
        "",
        "1. `data-validation`",
        "2. `train-evaluate`",
        "3. `cml-report`",
        "",
        "## Data Quality Gate",
        "",
        f"- Validation status: `{validation['status']}`",
        f"- Train rows: `{validation['rows']['train']}`",
        f"- Test rows: `{validation['rows']['test']}`",
        "",
        metrics_md,
        "",
        f"Selected model: `{metrics['best_model']}`",
        "",
        '![Confusion Matrix](./artifacts/confusion_matrix.png "Confusion Matrix")',
    ]

    report_path = project_root / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Report generated -> {report_path}")


if __name__ == "__main__":
    main()
