from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer


def validate_columns(df: pd.DataFrame, expected_columns: list[str]) -> None:
    missing = sorted(set(expected_columns) - set(df.columns))
    extra = sorted(set(df.columns) - set(expected_columns))
    if missing or extra:
        raise ValueError(
            f"Column mismatch. Missing={missing if missing else '[]'} "
            f"Extra={extra if extra else '[]'}"
        )


def validate_target(df: pd.DataFrame) -> None:
    values = set(df["target"].unique())
    if not values.issubset({0, 1}):
        raise ValueError(f"Target must be binary in {{0,1}}. Found={sorted(values)}")


def validate_no_nulls(df: pd.DataFrame, split: str) -> None:
    null_count = int(df.isnull().sum().sum())
    if null_count > 0:
        raise ValueError(f"{split} dataset has {null_count} null values")


def main() -> None:
    data_dir = Path("data/processed")
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Missing train/test files. Run `python src/get_data.py` first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    expected = list(load_breast_cancer(as_frame=True).feature_names) + ["target"]
    validate_columns(train_df, expected)
    validate_columns(test_df, expected)
    validate_target(train_df)
    validate_target(test_df)
    validate_no_nulls(train_df, "Train")
    validate_no_nulls(test_df, "Test")

    summary = {
        "status": "passed",
        "rows": {
            "train": int(len(train_df)),
            "test": int(len(test_df)),
        },
        "target_distribution": {
            "train": {
                "0": int((train_df["target"] == 0).sum()),
                "1": int((train_df["target"] == 1).sum()),
            },
            "test": {
                "0": int((test_df["target"] == 0).sum()),
                "1": int((test_df["target"] == 1).sum()),
            },
        },
        "columns_checked": len(expected),
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
    ]
    (artifacts_dir / "data_validation.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Data validation passed -> {summary_path}")


if __name__ == "__main__":
    main()
