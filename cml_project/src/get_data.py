from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def main() -> None:
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    target = pd.Series(dataset.target, name="target")
    frame = pd.concat([features, target], axis=1)

    train_df, test_df = train_test_split(
        frame,
        test_size=0.2,
        random_state=42,
        stratify=frame["target"],
    )

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train rows: {len(train_df)} -> {train_path}")
    print(f"Test rows: {len(test_df)} -> {test_path}")


if __name__ == "__main__":
    main()
