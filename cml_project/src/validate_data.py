from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing


def validate_columns(df: pd.DataFrame, expected_columns: list[str]) -> None:
    missing = sorted(set(expected_columns) - set(df.columns))
    extra = sorted(set(df.columns) - set(expected_columns))
    if missing or extra:
        raise ValueError(
            f"Column mismatch. Missing={missing if missing else '[]'} "
            f"Extra={extra if extra else '[]'}"
        )


def validate_no_nulls(df: pd.DataFrame, split: str) -> None:
    null_count = int(df.isnull().sum().sum())
    if null_count > 0:
        raise ValueError(f"{split} dataset has {null_count} null values")


def validate_target(df: pd.DataFrame, split: str) -> None:
    if not pd.api.types.is_numeric_dtype(df["target"]):
        raise ValueError(f"{split} target must be numeric")
    if not df["target"].between(0.0, 5.5).all():
        raise ValueError(
            f"{split} target contains values outside expected California Housing range"
        )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "processed"
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Missing train/test files. Run `python src/get_data.py` first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    expected = list(fetch_california_housing(as_frame=True).feature_names) + ["target"]
    validate_columns(train_df, expected)
    validate_columns(test_df, expected)
    validate_no_nulls(train_df, "Train")
    validate_no_nulls(test_df, "Test")
    validate_target(train_df, "Train")
    validate_target(test_df, "Test")

    summary = {
        "status": "passed",
        "rows": {
            "train": int(len(train_df)),
            "test": int(len(test_df)),
        },
        "columns_checked": len(expected),
        "target_stats": {
            "train_min": float(train_df["target"].min()),
            "train_max": float(train_df["target"].max()),
            "train_mean": float(train_df["target"].mean()),
            "test_min": float(test_df["target"].min()),
            "test_max": float(test_df["target"].max()),
            "test_mean": float(test_df["target"].mean()),
        },
    }

    summary_path = artifacts_dir / "data_validation.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "## Data Validation",
        "",
        "- Status: `passed`",
        f"- Train rows: `{summary['rows']['train']}`",
        f"- Test rows: `{summary['rows']['test']}`",
        f"- Columns checked: `{summary['columns_checked']}`",
        (
            "- Train target range: "
            f"`[{summary['target_stats']['train_min']:.3f}, {summary['target_stats']['train_max']:.3f}]`"
        ),
        (
            "- Test target range: "
            f"`[{summary['target_stats']['test_min']:.3f}, {summary['target_stats']['test_max']:.3f}]`"
        ),
    ]
    (artifacts_dir / "data_validation.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )

    print(f"Data validation passed -> {summary_path}")


if __name__ == "__main__":
    main()
