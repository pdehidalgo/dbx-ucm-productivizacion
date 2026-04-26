# Laboratorio: CML para Pipelines de ML en CI

Este laboratorio adapta el enfoque oficial de `iterative/cml` (getting started + ejemplos practicos) a un caso docente para explicar pipelines de MLOps en GitHub Actions.

## Objetivo del laboratorio

Construir un pipeline CI de 3 etapas que:

1. valida datos,
2. entrena y evalua varios modelos,
3. publica un reporte automatico en la Pull Request con CML.

## Pipeline en GitHub Actions

El workflow esta en:

- `/.github/workflows/cml_project.yml`

Y ejecuta estos jobs encadenados:

1. `data-validation`
2. `train-evaluate`
3. `cml-report`

El job final genera un comentario en la PR con:

- resumen de validacion de datos,
- tabla de metricas por modelo,
- mejor modelo seleccionado,
- matriz de confusion del mejor modelo.

## Dataset y modelo del ejemplo base

- Dataset: `Breast Cancer Wisconsin` de `scikit-learn` (clasificacion binaria tabular).
- Modelos comparados:
  - `LogisticRegression` (baseline lineal),
  - `RandomForestClassifier` (baseline no lineal).

## Variantes recomendadas para el curso

1. Clasificacion tabular (intro a pipelines)
- Dataset: `Titanic` (Kaggle/OpenML) o `Adult Income` (OpenML).
- Modelos: `LogisticRegression`, `RandomForest`, `XGBoost`.

2. Regresion tabular (metricas de negocio)
- Dataset: `California Housing`.
- Modelos: `LinearRegression`, `RandomForestRegressor`, `XGBoostRegressor`.

3. Series temporales (pipeline por ventanas)
- Dataset: `Bike Sharing` o `M4 subset`.
- Modelos: baseline `naive`, `LightGBM`, `Prophet` (si quieres comparar enfoques).

## Estructura

```text
cml_project/
- README.md
- pyproject.toml
- src/
  - build_report.py
  - get_data.py
  - train.py
  - validate_data.py
```

## Ejecucion local

```bash
uv sync --locked --package cml_project
uv run --project cml_project python cml_project/src/get_data.py
uv run --project cml_project python cml_project/src/validate_data.py
uv run --project cml_project python cml_project/src/train.py
uv run --project cml_project python cml_project/src/build_report.py
```

Tambien puedes ejecutar el flujo completo con:

```bash
cd cml_project
make install
make run
```

Artefactos esperados:

- `artifacts/data_validation.json`
- `artifacts/metrics.json`
- `artifacts/leaderboard.csv`
- `artifacts/confusion_matrix.png`
- `report.md`

## Referencias oficiales usadas

- https://github.com/iterative/cml#getting-started
- https://github.com/iterative-test/cml-example-minimal
- https://github.com/iterative-test/cml-example-dvc



## CML NO incluye out of the box:

- tracking avanzado tipo MLflow
- model registry
- serving
- monitorización en producción
- triggers por datos (CT real)
- feature store


## Sin CML tendrías que construir:

- integración con GitHub API
- sistema de reporting
- gestión de artefactos
- scripts de publicación
- parte del pipeline CI
