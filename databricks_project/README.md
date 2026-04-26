# Databricks Project

This project contains Databricks-focused notebooks/scripts:

- `DBFS_Example.py`
- `ML_Lab_MLflow_Databricks.py`

Run commands from repo root with `uv`:

```bash
uv sync --all-packages
uv run --project databricks_project python databricks_project/ML_Lab_MLflow_Databricks.py
uv run --project databricks_project python databricks_project/DBFS_Example.py
```
