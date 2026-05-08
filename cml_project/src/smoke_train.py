from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "processed"
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            "Missing train.csv. Run `python src/get_data.py` first."
        )

    train_df = pd.read_csv(train_path)
    sample_df = train_df.sample(n=min(3000, len(train_df)), random_state=42)

    feature_cols = [col for col in sample_df.columns if col != "target"]
    x = sample_df[feature_cols]
    y = sample_df["target"]

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ]
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_val)

    rmse = float(mean_squared_error(y_val, preds) ** 0.5)
    if not (0.0 < rmse < 2.0):
        raise ValueError(f"Smoke training RMSE out of expected band: {rmse:.4f}")

    summary = {
        "status": "passed",
        "model": "linear_regression_smoke",
        "sample_size": int(len(sample_df)),
        "rmse": rmse,
        "check": "0 < rmse < 2.0",
    }
    (artifacts_dir / "smoke_train.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    md = [
        "## Smoke Train",
        "",
        "- Status: `passed`",
        "- Model: `linear_regression_smoke`",
        f"- Sample size: `{summary['sample_size']}`",
        f"- RMSE: `{summary['rmse']:.4f}`",
        "- Gate: `0 < rmse < 2.0`",
    ]
    (artifacts_dir / "smoke_train.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Smoke training passed with RMSE={rmse:.4f}")


if __name__ == "__main__":
    main()
