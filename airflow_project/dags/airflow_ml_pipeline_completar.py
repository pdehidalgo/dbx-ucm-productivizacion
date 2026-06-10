from __future__ import annotations

from datetime import timedelta
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import yaml
from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.utils.dates import days_ago
from mlflow import MlflowClient
from mlflow.exceptions import RestException
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


logger = logging.getLogger(__name__)

CONFIG_CANDIDATES = (
    Path("/opt/airflow/config/config.yaml"),
    Path(__file__).resolve().parents[1] / "config" / "config.yaml",
)
TARGET_COLUMN = "species"
FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
IRIS_RENAME_MAP = {
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)": "petal_width",
}


def load_config():
    for candidate in CONFIG_CANDIDATES:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as file:
                return yaml.safe_load(file)
    raise FileNotFoundError("config.yaml not found in expected locations")


config = load_config()

DATA_PATH = config["paths"]["data"]
EXPERIMENT_NAME = config["mlflow"]["experiment_name"]
MODEL_NAME = config["mlflow"]["model_name"]
DATA_SOURCE_TYPE = config["factories"]["data_source_type"]
TRAINING_TABLE = config["factories"]["training_table"]

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


def build_iris_dataframe() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns=IRIS_RENAME_MAP)
    df[TARGET_COLUMN] = df["target"].map(dict(enumerate(iris.target_names)))
    return df[FEATURE_COLUMNS + [TARGET_COLUMN]]


class DataReader:
    def read(self, source: str) -> pd.DataFrame:
        raise NotImplementedError


class LocalCSVReader(DataReader):
    def read(self, source: str) -> pd.DataFrame:
        source_path = Path(source)
        if source_path.exists():
            df = pd.read_csv(source_path)
            return df[FEATURE_COLUMNS + [TARGET_COLUMN]]
        return build_iris_dataframe()


class DeltaTableReader(DataReader):
    def read(self, source: str) -> pd.DataFrame:
        raise NotImplementedError("DeltaTableReader queda fuera del alcance de este ejercicio")


class ReaderFactory:
    readers = {
        # TODO (1): registra aqui el lector para la clave "local".
        # TODO (2): deja identificado el camino "delta" aunque no se evalua en esta tarea.
    }

    @staticmethod
    def get_reader(source_type: str) -> DataReader:
        # TODO (3): devuelve la clase lectora correcta usando ReaderFactory.readers.
        raise NotImplementedError("Completa ReaderFactory.get_reader para soportar al menos el modo local")


class TrainerFactory:
    @staticmethod
    def get_trainer(model_type: str):
        # TODO (4): devuelve un RandomForestClassifier para "random_forest".
        # TODO (5): devuelve un LogisticRegression para "logistic_regression".
        # TODO (6): devuelve un xgb.XGBClassifier para "xgboost".
        raise NotImplementedError("Completa TrainerFactory.get_trainer con los tres tipos de modelo")


def get_best_registered_metric(client: MlflowClient, model_name: str, metric_name: str) -> float | None:
    try:
        model_versions = client.search_model_versions(f"name = '{model_name}'")
    except RestException:
        return None

    best_value = None
    for model_version in model_versions:
        run = client.get_run(model_version.run_id)
        if metric_name not in run.data.metrics:
            continue
        metric_value = float(run.data.metrics[metric_name])
        if best_value is None or metric_value > best_value:
            best_value = metric_value
    return best_value


@dag(
    dag_id="ml_pipeline_preprocessing_training_exercise",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["ml", "training", "exercise"],
)
def preprocessing_training_pipeline_exercise():
    @task()
    def load_data():
        reader = ReaderFactory.get_reader(DATA_SOURCE_TYPE)
        df = reader.read(DATA_PATH if DATA_SOURCE_TYPE == "local" else TRAINING_TABLE)
        return df.to_json()

    @task()
    def preprocess_data(data_json: str):
        df = pd.read_json(data_json)
        df = df.dropna().reset_index(drop=True)
        return df[FEATURE_COLUMNS + [TARGET_COLUMN]].to_json()

    @task()
    def train_and_register(data_json: str):
        df = pd.read_json(data_json)
        X = df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMN]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        model_type = Variable.get("model_type", default_var="random_forest")
        metrics = [metric.strip() for metric in Variable.get("metrics", default_var="accuracy,precision").split(",")]
        main_metric = Variable.get("main_metric", default_var="accuracy")
        force_register_model = Variable.get("force_register_model", default_var="false").lower() == "true"

        mlflow.set_tracking_uri("http://mlflow:5000")

        # TODO (7): activa el experimento correcto en MLflow con mlflow.set_experiment(EXPERIMENT_NAME).
        raise NotImplementedError("TODO (7): activa el experimento antes de arrancar el run")

        with mlflow.start_run(run_name=f"training_{model_type}") as run:
            model = TrainerFactory.get_trainer(model_type)
            if model is None:
                raise NotImplementedError("Completa TrainerFactory antes de ejecutar el DAG")

            mlflow.log_params(
                {
                    "model_type": model_type,
                    "main_metric": main_metric,
                    "force_register_model": force_register_model,
                    "data_source_type": DATA_SOURCE_TYPE,
                }
            )

            y_train_fit = y_train
            y_test_eval = y_test
            if model_type == "xgboost":
                label_encoder = LabelEncoder()
                y_train_fit = label_encoder.fit_transform(y_train)
                y_test_eval = label_encoder.transform(y_test)

            # TODO (8): entrena el modelo con X_train e y_train_fit.
            raise NotImplementedError("TODO (8): entrena el modelo seleccionado")

            # TODO (9): genera predicciones sobre X_test y calcula accuracy y precision.
            raise NotImplementedError("TODO (9): calcula las metricas pedidas")

            # TODO (10): registra metricas y el modelo como artifact_path='model'.
            raise NotImplementedError("TODO (10): loggea metricas y modelo en MLflow")

            new_value = 0.0
            client = MlflowClient()
            best_previous_value = get_best_registered_metric(client, MODEL_NAME, main_metric)

            should_register = False
            reason = "metric_below_threshold"

            # TODO (11): si no hay historico, registra; si lo hay, compara main_metric y respeta force_register_model.
            # TODO (12): registra o salta el registro dejando un tag model_registration_reason y un mensaje explicativo.
            raise NotImplementedError("TODO (11-12): completa la logica de registro condicional")

        return {
            "run_id": run.info.run_id,
            "model_type": model_type,
            "main_metric": main_metric,
            "metric_value": new_value,
            "registered": should_register,
        }

    raw = load_data()
    processed = preprocess_data(raw)
    train_and_register(processed)


training_exercise_dag = preprocessing_training_pipeline_exercise()
