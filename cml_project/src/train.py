from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def format_markdown_table(df: pd.DataFrame) -> str:
    header = "| model | accuracy | precision | recall | f1 |"
    separator = "|---|---:|---:|---:|---:|"
    rows = [
        f"| {row.model} | {row.accuracy:.4f} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, separator, *rows])


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "processed"
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    feature_cols = [col for col in train_df.columns if col != "target"]
    x_train = train_df[feature_cols]
    y_train = train_df["target"]
    x_test = test_df[feature_cols]
    y_test = test_df["target"]

    models: dict[str, Pipeline | RandomForestClassifier] = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=400, random_state=42)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline | RandomForestClassifier] = {}

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        preds = model.predict(x_test)

        metrics = {
            "model": model_name,
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds)),
            "recall": float(recall_score(y_test, preds)),
            "f1": float(f1_score(y_test, preds)),
        }
        results.append(metrics)
        fitted_models[model_name] = model

    leaderboard = pd.DataFrame(results).sort_values(by="f1", ascending=False).reset_index(drop=True)
    best_model_name = str(leaderboard.iloc[0]["model"])
    best_model = fitted_models[best_model_name]
    best_preds = best_model.predict(x_test)

    summary = {
        "best_model": best_model_name,
        "selection_metric": "f1",
        "models": leaderboard.to_dict(orient="records"),
    }
    (artifacts_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    leaderboard.to_csv(artifacts_dir / "leaderboard.csv", index=False)

    metrics_md_lines = [
        "## Model Metrics",
        "",
        format_markdown_table(leaderboard),
        "",
        f"Best model by `f1`: `{best_model_name}`",
    ]
    (artifacts_dir / "metrics.txt").write_text("\n".join(metrics_md_lines) + "\n", encoding="utf-8")

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        best_preds,
        normalize="true",
        cmap=plt.cm.Blues,
    )
    plt.title(f"Confusion Matrix ({best_model_name})")
    plt.tight_layout()
    plt.savefig(artifacts_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"Training finished. Best model: {best_model_name}")


if __name__ == "__main__":
    main()
