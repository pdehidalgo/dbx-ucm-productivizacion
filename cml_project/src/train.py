from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def format_markdown_table(df: pd.DataFrame) -> str:
    """Build a Markdown table with the model evaluation metrics."""
    header = "| model | rmse | mae | r2 |"
    separator = "|---|---:|---:|---:|"
    rows = [
        f"| {row.model} | {row.rmse:.4f} | {row.mae:.4f} | {row.r2:.4f} |"
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

    models: dict[str, object] = {
        "linear_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", LinearRegression()),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=18,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_depth=10,
            max_iter=400,
            random_state=42,
        ),
    }

    results: list[dict[str, float | str]] = []
    fitted_models: dict[str, object] = {}

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        preds = model.predict(x_test)

        metrics = {
            "model": model_name,
            "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
            "mae": float(mean_absolute_error(y_test, preds)),
            "r2": float(r2_score(y_test, preds)),
        }
        results.append(metrics)
        fitted_models[model_name] = model

    leaderboard = (
        pd.DataFrame(results).sort_values(by="rmse", ascending=True).reset_index(drop=True)
    )
    best_model_name = str(leaderboard.iloc[0]["model"])
    best_model = fitted_models[best_model_name]
    best_preds = best_model.predict(x_test)

    summary = {
        "best_model": best_model_name,
        "selection_metric": "rmse",
        "models": leaderboard.to_dict(orient="records"),
    }
    (artifacts_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    leaderboard.to_csv(artifacts_dir / "leaderboard.csv", index=False)
    joblib.dump(best_model, artifacts_dir / "best_model.joblib")

    metrics_md_lines = [
        "## Model Metrics",
        "",
        format_markdown_table(leaderboard),
        "",
        f"Best model by `rmse`: `{best_model_name}`",
    ]
    (artifacts_dir / "metrics.txt").write_text("\n".join(metrics_md_lines) + "\n", encoding="utf-8")

    residuals = y_test - best_preds
    plt.figure(figsize=(8, 6))
    plt.scatter(best_preds, residuals, alpha=0.25, edgecolor="none")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Predicted target")
    plt.ylabel("Residual (y_true - y_pred)")
    plt.title(f"Residual Plot ({best_model_name})")
    plt.tight_layout()
    plt.savefig(artifacts_dir / "residuals_plot.png", dpi=150)
    plt.close()

    if hasattr(best_model, "feature_importances_"):
        importance_values = best_model.feature_importances_
    elif hasattr(best_model, "named_steps") and hasattr(best_model.named_steps.get("reg"), "coef_"):
        importance_values = abs(best_model.named_steps["reg"].coef_)
    else:
        importance_values = [0.0 for _ in feature_cols]

    importance_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importance_values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(artifacts_dir / "feature_importance.csv", index=False)

    print(f"Training finished. Best model: {best_model_name}")


if __name__ == "__main__":
    main()
