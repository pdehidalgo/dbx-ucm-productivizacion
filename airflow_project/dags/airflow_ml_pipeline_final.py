# airflow_ml_pipelines.py
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from airflow.models import Variable
from datetime import timedelta, datetime
import pandas as pd
import os
import json
import logging
import shutil
import tempfile
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score
from dotenv import load_dotenv

load_dotenv()  

# Config
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)

config = load_config("/opt/airflow/config/config.yaml")

DATA_PATH, TEST_PATH = config["paths"]["data"], config["paths"]["test"]
PREDICTIONS_PATH = config["paths"]["predictions"]
EXPERIMENT_NAME = config["mlflow"]["experiment_name"]
MODEL_NAME = config["mlflow"]["model_name"]

DATA_SOURCE_TYPE = config["factories"]["data_source_type"]
TRAINING_TABLE = config["factories"]["training_table"]
TEST_TABLE = config["factories"]["test_table"]

FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
DRIFT_DETECTOR_ARTIFACT_PATH = "drift_detector"
DRIFT_P_VALUE = 0.05

logger = logging.getLogger(__name__)


# Default args
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

# Factory Pattern for data source readers
class DataReader:
    def read(self, source):
        raise NotImplementedError

class LocalCSVReader(DataReader):
    def read(self, source):
        # NOTE: mocking input pd.read_csv(source)
        import seaborn as sns
        iris = sns.load_dataset("iris")
        return iris.sample(30, random_state=99)

class DeltaTableReader(DataReader):
    # TODO: 1. NOT WORKING
    def read(self, source):
        from databricks import sql
        connection = sql.connect(
            server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
            http_path=os.getenv("DATABRICKS_HTTP_PATH"),
            access_token=os.getenv("DATABRICKS_TOKEN")
        )
        cursor = connection.cursor()
        cursor.execute(f"SELECT * FROM {source}")
        df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
        cursor.close()
        connection.close()
        return df

class ReaderFactory:

    # TODO: improvement
    readers = {

    }

    @staticmethod
    def get_reader(source_type):
        if source_type == "delta":
            return DeltaTableReader()
        return LocalCSVReader()

# Factory Pattern for model training
class ModelTrainer:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name

    def train_and_log(self, X_train, X_test, y_train, y_test, metrics):
        raise NotImplementedError

class RandomForestTrainer(ModelTrainer):
    def train_and_log(self, X_train, X_test, y_train, y_test, metrics):
        from sklearn.ensemble import RandomForestClassifier
        import mlflow

        mlflow.set_experiment(self.experiment_name)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        results = {}
        if "accuracy" in metrics:
            results["accuracy"] = accuracy_score(y_test, y_pred)
        if "precision" in metrics:
            results["precision"] = precision_score(y_test, y_pred, average="macro")

        for metric, value in results.items():
            mlflow.log_metric(metric, value)

        mlflow.sklearn.log_model(clf, artifact_path="model")
        return results

class LogisticRegressionTrainer(ModelTrainer):
    # TODO: 2. Test & review metrics
    def train_and_log(self, X_train, X_test, y_train, y_test, metrics):
        from sklearn.linear_model import LogisticRegression
        import mlflow

        mlflow.set_experiment(self.experiment_name)

        clf = LogisticRegression(max_iter=200)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        results = {}
        if "accuracy" in metrics:
            results["accuracy"] = accuracy_score(y_test, y_pred)
        if "precision" in metrics:
            results["precision"] = precision_score(y_test, y_pred, average="macro")

        for metric, value in results.items():
            mlflow.log_metric(metric, value)

        mlflow.sklearn.log_model(clf, artifact_path="model")
        return results
    
class XGBoostTrainer(ModelTrainer):
    # TODO: test and execute
    def train_and_log(self, X_train, X_test, y_train, y_test, metrics):
        import xgboost as xgb
        import mlflow

        mlflow.set_experiment(self.experiment_name)

        clf = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        results = {}
        if "accuracy" in metrics:
            results["accuracy"] = accuracy_score(y_test, y_pred)
        if "precision" in metrics:
            results["precision"] = precision_score(y_test, y_pred, average="macro")

        for metric, value in results.items():
            mlflow.log_metric(metric, value)

        mlflow.xgboost.log_model(clf, artifact_path="model")
        return results


class TrainerFactory:
    # All classifiers
    @staticmethod
    def get_trainer(model_type, experiment_name):
        if model_type == "random_forest":
            return RandomForestTrainer(experiment_name)
        elif model_type == "logistic_regression":
            return LogisticRegressionTrainer(experiment_name)
        elif model_type == "xgboost":
            return XGBoostTrainer(experiment_name)
        raise ValueError(f"Unsupported model_type: {model_type}")

@dag(
    dag_id="ml_pipeline_preprocessing_training",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["ml", "training"]
)
def preprocessing_training_pipeline():

    @task()
    def load_data():
        reader = ReaderFactory.get_reader(DATA_SOURCE_TYPE)
        df = reader.read(DATA_PATH if DATA_SOURCE_TYPE == "local" else TRAINING_TABLE)
        return df.to_json()

    @task()
    def preprocess_data(data_json):
        df = pd.read_json(data_json)
        df = df.dropna()
        # df = pd.get_dummies(df, drop_first=True)
        return df.to_json()

    @task()
    def train_and_register(data_json):
        from alibi_detect.cd import TabularDrift
        from alibi_detect.utils.saving import save_detector
        from sklearn.model_selection import train_test_split

        df = pd.read_json(data_json)
        X = df[FEATURE_COLUMNS]
        y = df['species']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model_type = Variable.get("model_type", default_var="random_forest")
        metrics = Variable.get("metrics", default_var="accuracy,precision").split(",")
        main_metric = Variable.get("main_metric", default_var="accuracy")
        force_register_model = Variable.get("force_register_model", default_var="false").lower() == "true"

        mlflow.set_tracking_uri("http://mlflow:5000")

        # os.makedirs("/home/airflow/mlruns", exist_ok=True)

        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.set_experiment(
                EXPERIMENT_NAME,
                # artifact_location="file:///opt/airflow/mlruns" 
                ).experiment_id
        else:
            experiment_id = experiment.experiment_id

        with mlflow.start_run(experiment_id=experiment_id):
            trainer = TrainerFactory.get_trainer(
                model_type=model_type,
                experiment_name=EXPERIMENT_NAME
            )
            print("Active run:", mlflow.active_run().info.run_id)

            results = trainer.train_and_log(X_train, X_test, y_train, y_test, metrics)
            new_value = results.get(main_metric, 0.0)

            # Build and version the drift detector with the same training run.
            drift_detector_dir = tempfile.mkdtemp(prefix="drift_detector_")
            try:
                drift_detector = TabularDrift(X_train.to_numpy(), p_val=DRIFT_P_VALUE)
                save_detector(drift_detector, drift_detector_dir)
                mlflow.log_artifacts(drift_detector_dir, artifact_path=DRIFT_DETECTOR_ARTIFACT_PATH)
                mlflow.set_tag("has_drift_detector", "true")
            finally:
                shutil.rmtree(drift_detector_dir, ignore_errors=True)

            from mlflow.exceptions import RestException
            client = mlflow.tracking.MlflowClient()

            # Code for registering the model
            try:
                model_versions = client.get_latest_versions(MODEL_NAME)
                best_value = 0.0
                for mv in model_versions:
                    m = client.get_run(mv.run_id)
                    old_value = float(m.data.metrics.get(main_metric, 0))
                    if old_value > best_value:
                        best_value = old_value
                print(f"Best metric value: {best_value}, new value is: {new_value}")
                should_register = force_register_model or (new_value >= best_value)
                if should_register:
                    mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", MODEL_NAME)
                    mlflow.set_tag(
                        "model_registration_reason",
                        "forced_by_flag" if force_register_model else "metric_threshold",
                    )
                else:
                    print(
                        "Model registration skipped "
                        f"(force_register_model={force_register_model}, "
                        f"new_value={new_value}, best_value={best_value})"
                    )
            except RestException:
                mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", MODEL_NAME)
                mlflow.set_tag("model_registration_reason", "first_registration_or_registry_missing")

        return f"Training completed. {main_metric}={new_value}"

    raw = load_data()
    processed = preprocess_data(raw)
    train_and_register(processed)


# Alibi monitoring
@task.branch(task_id="check_monitoring_condition")
def conditional_monitoring():
    df = pd.read_csv(PREDICTIONS_PATH)
    # new_data = df.tail(20)  # Asumimos últimos 20 registros
    new_data = df

    if len(new_data) < 20:
        return "skip_monitoring"

    from alibi_detect.utils.saving import load_detector

    mlflow.set_tracking_uri("http://mlflow:5000")
    client = mlflow.tracking.MlflowClient()

    model_versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
    if not model_versions:
        raise ValueError(f"No registered versions found for model '{MODEL_NAME}'")

    latest_model_version = max(model_versions, key=lambda mv: int(mv.version))
    try:
        detector_path = mlflow.artifacts.download_artifacts(
            run_id=latest_model_version.run_id,
            artifact_path=DRIFT_DETECTOR_ARTIFACT_PATH,
            dst_path="/tmp",
        )
        detector = load_detector(detector_path)
    except Exception as exc:
        raise ValueError(
            "Latest registered model does not contain a compatible drift detector artifact "
            f"(model_version={latest_model_version.version}, run_id={latest_model_version.run_id})"
        ) from exc

    current = df.iloc[-20:].drop(columns=["prediction"])

    preds = detector.predict(current.to_numpy())

    def _jsonify(value):
        if isinstance(value, dict):
            return {k: _jsonify(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonify(v) for v in value]
        if hasattr(value, "tolist"):
            return value.tolist()
        if hasattr(value, "item"):
            return value.item()
        return value

    report_dir = "/opt/airflow/data/drift_reports"
    os.makedirs(report_dir, exist_ok=True)
    timestamp_utc = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    report_path = os.path.join(
        report_dir,
        f"drift_report_{timestamp_utc}_v{latest_model_version.version}.json",
    )
    report = {
        "timestamp_utc": timestamp_utc,
        "model_name": MODEL_NAME,
        "model_version": latest_model_version.version,
        "model_run_id": latest_model_version.run_id,
        "window_size": int(len(current)),
        "preds": _jsonify(preds),
    }
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    shutil.copyfile(report_path, os.path.join(report_dir, "latest.json"))

    logger.info(
        "Loaded drift detector from model_version=%s run_id=%s",
        latest_model_version.version,
        latest_model_version.run_id,
    )
    logger.info("Drift scores: %s", preds["data"]["p_val"])
    logger.info("Drift report saved to %s", report_path)
    return "monitoring_done"

@task()
def skip_monitoring():
    print("Not enough data for monitoring")

@task()
def monitoring_done():
    print("Monitoring completed")


@dag(
    dag_id="ml_pipeline_inference",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["ml", "inference"]
)
def inference_pipeline():

    @task()
    def load_input():
        reader = ReaderFactory.get_reader(DATA_SOURCE_TYPE)
        df = reader.read(TEST_PATH if DATA_SOURCE_TYPE == "local" else "default.test_table")
        return df.to_json()

    @task()
    def predict_with_latest_model(data_json):
        from mlflow.tracking import MlflowClient
        df = pd.read_json(data_json)[['sepal_length','sepal_width','petal_length','petal_width']]

        print(df)

        # client = MlflowClient()
        # os.makedirs("/opt/airflow/artifacts", exist_ok=True)
        mlflow.set_tracking_uri("http://mlflow:5000")

        client = MlflowClient()
        models = client.search_registered_models()

        for m in models:
            print(f"Model: {m.name}")

        model_uri = f"models:/{MODEL_NAME}/latest"
        model = mlflow.pyfunc.load_model(model_uri)

        print(model)
        
        predictions = model.predict(df)
        df["prediction"] = predictions

        existing = pd.read_csv(PREDICTIONS_PATH) if os.path.exists(PREDICTIONS_PATH) else pd.DataFrame()
        combined = pd.concat([existing, df], ignore_index=True)
        combined.drop_duplicates(inplace=True)
        combined.to_csv(PREDICTIONS_PATH, index=False)
        return "Predictions updated."

    input_data = load_input()
    pred = predict_with_latest_model(input_data)
    branch = conditional_monitoring()
    pred >> branch >> [skip_monitoring(), monitoring_done()]


# Register DAGs
d1 = preprocessing_training_pipeline()
d2 = inference_pipeline()
    
