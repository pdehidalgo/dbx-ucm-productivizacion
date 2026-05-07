from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "processed"
    artifacts_dir = project_root / "artifacts"
    data_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dataset = fetch_california_housing(as_frame=True)
    features = dataset.data
    target = pd.Series(dataset.target, name="target")
    frame = pd.concat([features, target], axis=1)

    train_df, test_df = train_test_split(
        frame,
        test_size=0.2,
        random_state=42,
    )

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = {
        "dataset_name": "california_housing",
        "rows": {
            "total": int(len(frame)),
            "train": int(len(train_df)),
            "test": int(len(test_df)),
        },
        "columns": list(frame.columns),
        "target": {
            "name": "target",
            "min": float(frame["target"].min()),
            "max": float(frame["target"].max()),
            "mean": float(frame["target"].mean()),
        },
    }
    (artifacts_dir / "data_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Train rows: {len(train_df)} -> {train_path}")
    print(f"Test rows: {len(test_df)} -> {test_path}")


if __name__ == "__main__":
    main()
