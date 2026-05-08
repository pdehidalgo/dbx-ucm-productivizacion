# Laboratorio: CML para California Housing en CI

Este laboratorio implementa un pipeline de `iterative/cml` para un caso de **regresion tabular** con el dataset **California Housing**.

## Objetivo

Construir un flujo de CI aplicable a MLOps real que:

1. valida datos (schema + calidad),
2. ejecuta una puerta de smoke training en CI,
3. entrena y compara modelos de regresion,
4. publica un reporte automatico en PR con CML.

## Relacion con CI / CT / CD

- **CI (implementado aqui)**: `lint/tests` (si se anaden), validacion de datos, smoke train y packaging de artefactos.
- **CT (simulado en el job train-evaluate)**: entrenamiento completo y evaluacion comparativa con criterio de seleccion.
- **CD (conceptual en el notebook base)**: promover un modelo ya evaluado; no se reentrena al desplegar.

## Pipeline en GitHub Actions

Workflow:

- `/.github/workflows/cml_project.yml`

Jobs encadenados:

1. `data-validation`
2. `train-evaluate`
3. `cml-report`

## Dataset y modelos

- Dataset: `fetch_california_housing` de `scikit-learn`.
- Modelos comparados:
  - `LinearRegression` (baseline lineal),
  - `RandomForestRegressor` (baseline no lineal),
  - `HistGradientBoostingRegressor` (baseline boosting).

Metrica principal de seleccion:

- `RMSE` (menor es mejor).

Metricas reportadas:

- `RMSE`, `MAE`, `R2`.

## Estructura

```text
cml_project/
- README.md
- pyproject.toml
- src/
  - build_report.py
  - get_data.py
  - smoke_train.py
  - train.py
  - validate_data.py
```

## Ejecucion local end-to-end

```bash
uv sync --locked --package cml_project
uv run --project cml_project python cml_project/src/get_data.py
uv run --project cml_project python cml_project/src/validate_data.py
uv run --project cml_project python cml_project/src/smoke_train.py
uv run --project cml_project python cml_project/src/train.py
uv run --project cml_project python cml_project/src/build_report.py
```

O bien:

```bash
cd cml_project
make install
make run
```

Artefactos esperados:

- `artifacts/data_summary.json`
- `artifacts/data_validation.json`
- `artifacts/smoke_train.json`
- `artifacts/metrics.json`
- `artifacts/leaderboard.csv`
- `artifacts/feature_importance.csv`
- `artifacts/residuals_plot.png`
- `artifacts/best_model.joblib`
- `report.md`

## Referencias

- https://github.com/iterative/cml#getting-started
- https://github.com/iterative-test/cml-example-minimal
- https://github.com/iterative-test/cml-example-dvc
