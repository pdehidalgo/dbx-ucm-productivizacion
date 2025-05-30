

from sklearn.metrics import accuracy_score, precision_score
import seaborn as sns
iris = sns.load_dataset("iris")
df = iris.sample(5, random_state=99)

import pandas as pd

from sklearn.model_selection import train_test_split

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

        # mlflow.sklearn.log_model(clf, "model")
        return results

class TrainerFactory:
    # All classifiers
    @staticmethod
    def get_trainer(model_type, experiment_name):
        if model_type == "random_forest":
            return RandomForestTrainer(experiment_name)
        raise ValueError(f"Unsupported model_type: {model_type}")



X = df[['sepal_length','sepal_width','petal_length','petal_width']]
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
Variable = {}
model_type = Variable.get("model_type", "random_forest")
metrics = Variable.get("metrics", "accuracy,precision").split(",")
main_metric = Variable.get("main_metric", "accuracy")
EXPERIMENT_NAME = "ml_pipeline_experiment"
MODEL_NAME = "model_name"

import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)


with mlflow.start_run(experiment_id=experiment.experiment_id):
    trainer = TrainerFactory.get_trainer(
        model_type=model_type,
        experiment_name=EXPERIMENT_NAME
    )
    print("Active run:", mlflow.active_run().info.run_id)

    results = trainer.train_and_log(X_train, X_test, y_train, y_test, metrics)
    new_value = results.get(main_metric, 0.0)

    from mlflow.exceptions import RestException
    client = mlflow.tracking.MlflowClient()
    try:
        model_versions = client.get_latest_versions(MODEL_NAME)
        best_value = 0.0
        for mv in model_versions:
            m = client.get_run(mv.run_id)
            old_value = float(m.data.metrics.get(main_metric, 0))
            if old_value > best_value:
                best_value = old_value
        if new_value > best_value:
            mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", MODEL_NAME)
    except RestException:
        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", MODEL_NAME)