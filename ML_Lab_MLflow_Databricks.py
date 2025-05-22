# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ![logo](https://raw.githubusercontent.com/pdehidalgo/dbx-ucm-productivizacion/main/logo_ucm.jpg)
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # MLflow
# MAGIC
# MAGIC ## En esta lección:<br>
# MAGIC * Usarás MLflow para rastrear experimentos (con validación cruzada), registrar parámetros, métricas, el pipeline y el mejor modelo
# MAGIC * Modificarás el estado de un modelo mediante el SDK
# MAGIC * Crearás e invocarás un endpoint utilizando el SDK

# COMMAND ----------

# Leemos de la tabla previamente creada
airbnb_df = spark.read.table("hive_metastore.default.airbnb")

train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)
print(train_df.cache().count())

# COMMAND ----------

airbnb_df.count()

# COMMAND ----------

display(dbutils.fs.ls("/FileStore/airbnb.csv"))

# COMMAND ----------

airbnb_df.coalesce(1).write.mode("overwrite").option("header", True).csv("/FileStore/airbnb.csv")

# COMMAND ----------

display(train_df)

# COMMAND ----------

# MAGIC %md <i18n value="9ab8c080-9012-4f38-8b01-3846c1531a80"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### MLflow Tracking

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "price"))]
assembler_inputs = index_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

rf = RandomForestRegressor(labelCol="price", maxBins=40)
stages = [string_indexer, vec_assembler, rf]
pipeline = Pipeline(stages=stages)

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder

param_grid = (ParamGridBuilder()
              .addGrid(rf.maxDepth, [2, 5])
              .addGrid(rf.numTrees, [5, 10])
              .build())

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Cross Validation
# MAGIC
# MAGIC Vamos a utilizar también validación cruzada de 3 particiones (3-fold cross validation) para identificar los hiperparámetros óptimos.
# MAGIC
# MAGIC ![Cross Validation](https://raw.githubusercontent.com/pdehidalgo/dbx-ucm-productivizacion/main/cross_validation.png)
# MAGIC
# MAGIC Con la validación cruzada de 3 particiones (*3-fold cross-validation*), entrenamos con 2/3 de los datos y evaluamos con el 1/3 restante (conjunto de validación). Repetimos este proceso 3 veces, de modo que cada partición actúe como conjunto de validación una vez. Finalmente, promediamos los resultados de las tres iteraciones.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Pasamos el **`estimator`** (pipeline), el **`evaluator`** y los **`estimatorParamMaps`** a <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html?highlight=crossvalidator#pyspark.ml.tuning.CrossValidator" target="_blank">CrossValidator</a> para que sepa:
# MAGIC
# MAGIC - Qué modelo usar  
# MAGIC - Cómo evaluar el modelo  
# MAGIC - Qué hiperparámetros configurar para el modelo
# MAGIC
# MAGIC También podemos establecer el número de particiones (folds) en las que queremos dividir nuestros datos (3), así como una semilla aleatoria para que todos tengamos la misma división de los datos.
# MAGIC

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator

evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction")

cv = CrossValidator(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=param_grid, 
                    numFolds=3, seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC n_tries = 2 (params) * 2 (items per param) * 3 (folds)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC **Q**: ¿Cuántos modelos estaríamos entrenando?

# COMMAND ----------

# cv_model = cv.fit(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Pregunta**: ¿Deberíamos poner el pipeline dentro del validador cruzado, o el validador cruzado dentro del pipeline?
# MAGIC
# MAGIC Depende de si en el pipeline hay estimadores o transformadores. Si tienes elementos como `StringIndexer` (que es un estimador) en el pipeline, entonces tendrás que reajustarlo (*refit*) en cada iteración si pones todo el pipeline dentro del validador cruzado.
# MAGIC
# MAGIC Sin embargo, si hay alguna preocupación por la fuga de datos (*data leakage*) desde las etapas anteriores, lo más seguro es colocar el pipeline **dentro del CrossValidator**, y no al revés. El CrossValidator primero divide los datos y luego ejecuta `.fit()` sobre el pipeline. Si se coloca el CrossValidator al final del pipeline, podríamos estar filtrando información del conjunto de validación hacia el de entrenamiento.
# MAGIC

# COMMAND ----------

cv = CrossValidator(estimator=rf, evaluator=evaluator, estimatorParamMaps=param_grid, 
                    numFolds=3, seed=42)

stages_with_cv = [string_indexer, vec_assembler, cv]
pipeline_cv = Pipeline(stages=stages_with_cv)

# COMMAND ----------

import mlflow
import mlflow.spark
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator
from datetime import datetime

# Configuración del experimento (esto se puede ajustar)
# mlflow.set_experiment("/Users/<tu-email>@ejemplo.com/airbnb_experimento")
run_name = f"airbnb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

with mlflow.start_run(run_name=run_name) as run:

    # Entrenamiento
    pipeline_model = pipeline_cv.fit(train_df)

    # Predicciones
    pred_df = pipeline_model.transform(test_df)

    # Métricas
    rmse = evaluator.evaluate(pred_df)
    r2 = evaluator.setMetricName("r2").evaluate(pred_df)
    
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")

    # Logging a MLflow
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # Logging hiperparámetros (opcional)
    avg_metrics = pipeline_model.stages[-1].avgMetrics
    param_maps = pipeline_model.stages[-1].getEstimatorParamMaps()
    for i, (params, metric) in enumerate(zip(param_maps, avg_metrics)):
        mlflow.log_metric(f"fold_{i}_metric", metric)

   # Guardamos el pipeline completo, que incluye al mejor modelo y las transformaciones del pipeline, necesarias para la predicción
    mlflow.spark.log_model(pipeline_model, "full_pipeline")

    # Opcional: Logging del mejor modelo (última etapa del pipeline contiene el CV), vamos a necesitar el pipeline para predecir en cualquier caso
    best_model = pipeline_model.stages[-1].bestModel
    mlflow.spark.log_model(best_model, "best_model")

 

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC **Question**: ¿Cuántos modelos estaríamos entrenando?
# MAGIC
# MAGIC Con cada CV está lanzando 3-folds que posteriormente promedia su error, el mejor modelo es aquel con menor promedio de error. 

# COMMAND ----------

list(zip(pipeline_model.stages[-1].getEstimatorParamMaps(), pipeline_model.stages[-1].avgMetrics))

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

print(run_name)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Consultar ejecuciones anteriores
# MAGIC
# MAGIC Puedes consultar ejecuciones anteriores de forma programática para reutilizar esa información en Python. La forma de hacerlo es mediante un objeto **`MlflowClient`**.
# MAGIC

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

# COMMAND ----------

client.list_experiments()

# COMMAND ----------

# MAGIC %md Utilicemos el experiment_id que obtenemos del run para recopilar los últimos experimentos

# COMMAND ----------

experiment_id = run.info.experiment_id
runs_df = mlflow.search_runs(experiment_id)

display(runs_df)

# Observamos como uno de ellos ha fallado y otro fue bien.

# COMMAND ----------

# Obtenemos el último experimento
runs = client.search_runs(experiment_id, order_by=["attributes.start_time desc"], max_results=1)
runs[0].data.metrics

# COMMAND ----------

experiment_id

# COMMAND ----------

runs[0].info

# COMMAND ----------

runs[0].info.run_id

# COMMAND ----------

list_experiments = client.search_experiments()
for experiment in list_experiments:
    print(experiment.name, experiment.experiment_id)

# COMMAND ----------

# Otra opción
# experiment = client.get_experiment_by_name(run_name)

# client.search_runs(experiment.experiment_id, order_by=["attributes.start_time desc"], max_results=1)
client.search_runs(experiment_id)[0].to_dictionary()

# COMMAND ----------

# 'use_case'
def filter_by_use_case(experiment_name: str, use_case: str):
    aux = []
    experiment = client.get_experiment_by_name(experiment_name)
    for exp in client.search_runs(experiment.experiment_id):
        try:
            if exp.to_dictionary()["data"]["tags"]['use_case'] == use_case:
                aux.append(exp)
        except:
            print("")
    return aux

# COMMAND ----------

use_case_experiments = filter_by_use_case('', 'airbnb')

# COMMAND ----------

len(use_case_experiments)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Examina los resultados en la interfaz gráfica (UI). Busca lo siguiente:<br><br>
# MAGIC
# MAGIC 1. El **`ID del experimento`**
# MAGIC 2. La ubicación de los artefactos. Aquí es donde se almacenan los artefactos en DBFS.
# MAGIC 3. La hora en que se ejecutó la corrida. **Haz clic en ella para ver más información sobre la ejecución.**
# MAGIC 4. El código que ejecutó la corrida.
# MAGIC
# MAGIC Después de hacer clic en la hora de la ejecución, observa lo siguiente:<br><br>
# MAGIC
# MAGIC 1. El ID de la corrida coincidirá con el que imprimimos anteriormente.
# MAGIC 2. El modelo que guardamos incluye una versión serializada (pickled) del modelo, así como el entorno Conda y el archivo **`MLmodel`**.
# MAGIC
# MAGIC Ten en cuenta que puedes añadir notas en la pestaña "Notes" para ayudarte a llevar un registro de información importante sobre tus modelos.
# MAGIC
# MAGIC Además, haz clic en la ejecuciòn correspondiente a la distribución log-normal y verás que el histograma está guardado en "Artifacts".
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Cargar el modelo guardado
# MAGIC Vamos a practicar <a href="https://www.mlflow.org/docs/latest/python_api/mlflow.spark.html" target="_blank">descargando</a> nuestro log-normal model.

# COMMAND ----------

f"runs:/{run.info.run_id}/log-model"

# COMMAND ----------

model_path = f"dbfs:/databricks/mlflow-tracking/{experiment_id}/{run.info.run_id}/artifacts/full_pipeline"
loaded_model = mlflow.spark.load_model(model_path)

display(loaded_model.transform(test_df))
# Nota: observamos como al dataframe de test se le han añadido dos nuevas columnas, features y prediction

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Registremos un experimento via sdk

# COMMAND ----------

# MAGIC %md
# MAGIC Sin embargo, debes tener en cuenta que:
# MAGIC
# MAGIC     mlflow.register_model() espera una URI en el formato runs:/<run_id>/artifact_path, no un path en DBFS.
# MAGIC
# MAGIC     Si tienes ya el run_id y el artifact_path, puedes construir correctamente el URI requerido.
# MAGIC
# MAGIC 🧠 ¿Por qué no usar dbfs:/...?
# MAGIC
# MAGIC Porque dbfs:/... es una ruta física, y MLflow Model Registry necesita la referencia lógica del run y el path al artefacto para poder versionarlo correctamente.

# COMMAND ----------

run

# COMMAND ----------

from mlflow import register_model

run_id = run.info.run_id
artifact_path = "full_pipeline"
model_uri = f"runs:/{run_id}/{artifact_path}"

# model_name = "main.ml_models.airbnb_price_predictor"  # En Unity Catalog
model_name = "airbnb_price_predictor"

mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------


latest_version = client.get_latest_versions(name=model_name, stages=["None"])[0]
print(f"Model registered with version: {latest_version.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Registrar modelos condicionalmente: solo si lo merecen
# MAGIC
# MAGIC Registrar modelos **solo si cumplen ciertos criterios** permite una gestión robusta y eficiente del ciclo de vida del modelo.
# MAGIC
# MAGIC ### 1. Separación clara entre entrenamiento y registro
# MAGIC
# MAGIC - Puedes entrenar tantos modelos como quieras.
# MAGIC - Solo registras un modelo si **realmente lo merece**:  
# MAGIC   - Supera una métrica objetivo  
# MAGIC   - Mejora al modelo anterior  
# MAGIC   - Se comporta bien con un *ground truth* específico
# MAGIC
# MAGIC ```python
# MAGIC if new_model_r2 > baseline_r2:
# MAGIC     mlflow.register_model(...)
# MAGIC ```
# MAGIC ### 2. Evaluación sobre un ground truth fijo
# MAGIC
# MAGIC - Congelas un conjunto de validación (por ejemplo, una muestra real de producción).
# MAGIC
# MAGIC - Evalúas múltiples modelos sobre ese mismo dataset.
# MAGIC
# MAGIC - Garantizas que cualquier nuevo modelo no solo aprende mejor, sino que generaliza mejor.
# MAGIC
# MAGIC ### 3. Automatización + gobernanza
# MAGIC
# MAGIC - Puedes automatizar todo el proceso con Airflow o Databricks Jobs.
# MAGIC
# MAGIC - Solo se registran modelos que: cumplen los umbrales de calidad, superan al modelo actualmente desplegado, pasan validaciones técnicas (latencia, estabilidad, etc.)
# MAGIC
# MAGIC ### 4. Reproducibilidad 
# MAGIC
# MAGIC - Guardas cada modelo como artefacto en MLflow antes de registrarlo. Esto permite: 
# MAGIC
# MAGIC * Versionar todos los experimentos
# MAGIC
# MAGIC * Volver fácilmente a una versión anterior
# MAGIC
# MAGIC * Probar modelos no registrados de forma segura antes de promoverlos
# MAGIC
# MAGIC
# MAGIC ```python
# MAGIC # Evaluamos el nuevo modelo con ground truth congelado
# MAGIC r2_new = evaluator.evaluate(new_model.transform(ground_truth_df))
# MAGIC
# MAGIC # Comparamos con la métrica del modelo en producción
# MAGIC r2_prod = evaluator.evaluate(prod_model.transform(ground_truth_df))
# MAGIC
# MAGIC if r2_new > r2_prod + 0.01:
# MAGIC     model_uri = f"runs:/{run.info.run_id}/full_pipeline"
# MAGIC     model_name = "main.ml_models.airbnb_price_predictor"
# MAGIC     mlflow.register_model(model_uri=model_uri, name=model_name)
# MAGIC     print("✅ Modelo registrado (mejor que el actual)")
# MAGIC else:
# MAGIC     print("⛔ Modelo descartado (no mejora lo suficiente)")
# MAGIC
# MAGIC ``
# MAGIC

# COMMAND ----------

import requests
import json
import os

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")  # o hardcodea tu URL
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")  # PAT con permisos

headers = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json"
}

endpoint_name = "airbnb-serving"
model_name = "main.ml_models.airbnb_price_predictor"

payload = {
    "name": endpoint_name,
    "config": {
        "served_models": [
            {
                "model_name": model_name,
                "model_version": "1",  # o usa stage: "Production"
                "workload_type": "CPU",
                "workload_size": "Small"
            }
        ]
    }
}

response = requests.post(
    f"{DATABRICKS_HOST}/api/2.0/serving-endpoints",
    headers=headers,
    data=json.dumps(payload)
)

print(response.status_code, response.text)
