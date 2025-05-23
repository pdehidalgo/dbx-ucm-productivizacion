# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ![logo](https://raw.githubusercontent.com/pdehidalgo/dbx-ucm-productivizacion/main/logo_ucm.jpg)
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC
# MAGIC # Entrenamiento usando la API de Funciones de Pandas
# MAGIC
# MAGIC Este notebook muestra cómo utilizar la API de funciones de Pandas (`applyInPandas`) para gestionar y escalar el entrenamiento de modelos de machine learning por dispositivo IoT.
# MAGIC
# MAGIC ## En esta práctica aprenderás a:<br>
# MAGIC  - Utilizar <a href="https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.GroupedData.applyInPandas.html" target="_blank">**.groupBy().applyInPandas()**</a> para construir múltiples modelos en paralelo, uno por cada dispositivo IoT.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Generación de datos sintéticos:
# MAGIC - **`device_id`**: representa 10 dispositivos distintos.
# MAGIC - **`record_id`**: 10.000 registros únicos simulados.
# MAGIC - **`feature_1`**, **`feature_2`**, **`feature_3`**: variables independientes para el entrenamiento del modelo.
# MAGIC - **`label`**: variable objetivo que queremos predecir.
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# Entrenamiento con Pandas Function API

# Sección 1: Generación de datos sintéticos para simular entradas por dispositivo IoT
import pyspark.sql.functions as f

df = (spark
      .range(1000*100)
      .select(f.col("id").alias("record_id"), (f.col("id")%10).alias("device_id"))
      .withColumn("feature_1", f.rand() * 1)
      .withColumn("feature_2", f.rand() * 2)
      .withColumn("feature_3", f.rand() * 3)
      .withColumn("label", (f.col("feature_1") + f.col("feature_2") + f.col("feature_3")) + f.rand())
     )

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC Define el schema de retorno

# COMMAND ----------

# Sección 2: Esquema de retorno que definirá el formato del resultado tras entrenar cada modelo
train_return_schema = "device_id integer, n_used integer, model_path string, mse float"

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Define una función de pandas que reciba todos los datos correspondientes a un dispositivo, entrene un modelo, lo guarde como una ejecución anidada (nested run) y devuelva un objeto Spark con el esquema definido anteriormente.

# COMMAND ----------

# Sección 3: Entrenamiento de un modelo por cada device_id usando applyInPandas, incluyendo:
# - Validación del esquema de entrada
# - Separación entre entrenamiento y prueba
# - Registro condicional del modelo si tiene un MSE razonable
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def train_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
    columnas_esperadas = {"device_id", "run_id", "feature_1", "feature_2", "feature_3", "label"}
    if not columnas_esperadas.issubset(df_pandas.columns):
        raise ValueError(f"Faltan columnas esperadas: {columnas_esperadas - set(df_pandas.columns)}")

    device_id = df_pandas["device_id"].iloc[0]
    n_used = df_pandas.shape[0]
    run_id = df_pandas["run_id"].iloc[0]

    X = df_pandas[["feature_1", "feature_2", "feature_3"]]
    y = df_pandas["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)

    predictions = rf.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    if mse > 5.0:
        print(f"[AVISO] Modelo para device_id={device_id} no se guarda por MSE alto: {mse:.2f}")
        return pd.DataFrame([[device_id, n_used, None, mse]], columns=["device_id", "n_used", "model_path", "mse"])

    with mlflow.start_run(run_id=run_id) as outer_run:
        experiment_id = outer_run.info.experiment_id

        with mlflow.start_run(run_name=str(device_id), nested=True, experiment_id=experiment_id) as run:
            mlflow.sklearn.log_model(rf, str(device_id))
            mlflow.log_metric("mse", mse)
            mlflow.set_tag("device", str(device_id))

            artifact_uri = f"runs:/{run.info.run_id}/{device_id}"
            return_df = pd.DataFrame([[device_id, n_used, artifact_uri, mse]], 
                                     columns=["device_id", "n_used", "model_path", "mse"])

    return return_df


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC Apply the pandas function to grouped data. 
# MAGIC
# MAGIC Ten en cuenta que la forma de aplicar esto en la práctica depende en gran medida de dónde se encuentren los datos para la inferencia. En este ejemplo, reutilizaremos los datos de entrenamiento, que ya contienen el identificador del dispositivo y el identificador de ejecución (run_id).
# MAGIC

# COMMAND ----------

# Sección 4: Aplicar la función anterior con applyInPandas agrupando por device_id
# Se añade run_id y se cachea el resultado para eficiencia
with mlflow.start_run(run_name="Entrenamiento por dispositivo") as run:
    run_id = run.info.run_id

    model_directories_df = (df
        .withColumn("run_id", f.lit(run_id))
        .groupby("device_id")
        .applyInPandas(train_model, schema=train_return_schema)
        .cache()
    )

combined_df = df.join(model_directories_df, on="device_id", how="left")
display(combined_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Define una función de pandas y un esquema de retorno para aplicar el modelo. Esto requiere solo una lectura desde el sistema de archivos por cada dispositivo.

# COMMAND ----------

# Sección 5: Aplicación de los modelos entrenados sobre las mismas observaciones originales
# Incluye validación y manejo de errores al cargar modelos
apply_return_schema = "record_id integer, prediction float"

def apply_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
    if "model_path" not in df_pandas.columns:
        raise ValueError("Falta la columna 'model_path'")

    model_path = df_pandas["model_path"].iloc[0]
    X = df_pandas[["feature_1", "feature_2", "feature_3"]]

    try:
        model = mlflow.sklearn.load_model(model_path)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo desde {model_path}: {e}")
        return pd.DataFrame(columns=["record_id", "prediction"])

    prediction = model.predict(X)

    return_df = pd.DataFrame({
        "record_id": df_pandas["record_id"],
        "prediction": prediction
    })
    return return_df

prediction_df = combined_df.groupby("device_id").applyInPandas(apply_model, schema=apply_return_schema)
display(prediction_df)


# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Servir múltiples modelos desde un modelo registrado
# MAGIC
# MAGIC MLflow permite desplegar modelos como APIs REST en tiempo real. Actualmente, un único modelo en MLflow se sirve desde una instancia (normalmente una máquina virtual). Sin embargo, en algunos casos es necesario servir múltiples modelos desde un único punto de entrada. Imagina 1000 modelos similares que deben usarse con distintas entradas. Ejecutar 1000 endpoints por separado podría desperdiciar recursos, especialmente si algunos de esos modelos se usan con poca frecuencia.
# MAGIC
# MAGIC Una solución a este problema es empaquetar varios modelos dentro de un único modelo personalizado, el cual internamente enruta las peticiones al modelo adecuado según la entrada recibida, y despliega ese "paquete" de modelos como si fuera un único modelo.
# MAGIC
# MAGIC A continuación, mostramos cómo crear un modelo personalizado de este tipo, que agrupa todos los modelos entrenados para cada dispositivo. Por cada fila de datos que se le pase, el modelo determinará el `device_id` y utilizará el modelo correspondiente entrenado para ese dispositivo para hacer la predicción.
# MAGIC
# MAGIC Primero, debemos acceder a los modelos correspondientes a cada `device_id`.
# MAGIC

# COMMAND ----------


# Sección 6: Consulta de modelos entrenados registrados en MLflow a partir del experimento
experiment_id = run.info.experiment_id

model_df = (spark.read.format("mlflow-experiment")
            .load(experiment_id)
            .filter("tags.device IS NOT NULL")
            .orderBy("end_time", ascending=False)
            .select("tags.device", "run_id")
            .limit(10))

display(model_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Creamos un diccionario mapeando los modelos

# COMMAND ----------

# Sección 7: Carga de modelos por device_id en un diccionario, con manejo de errores por si algún modelo falló
device_to_model = {}
# WARNING: collect()
for row in model_df.collect():
    try:
        model = mlflow.sklearn.load_model(f"runs:/{row['run_id']}/{row['device']}")
        device_to_model[row["device"]] = model
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo para device={row['device']}: {e}")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Creamos un modelo personalizado que recibe como atributo un mapeo entre identificadores de dispositivo y modelos, y delega la entrada al modelo correspondiente según el device_id.

# COMMAND ----------

# Sección 8: Creación de un modelo (PythonModel) que enruta la predicción al modelo correcto según device_id
from mlflow.pyfunc import PythonModel

class OriginDelegatingModel(PythonModel):

    def __init__(self, device_to_model_map):
        self.device_to_model_map = device_to_model_map

    def predict_for_device(self, row: pd.Series) -> float:
        model = self.device_to_model_map.get(str(row["device_id"]))
        if model is None:
            return float("nan")
        data = row[["feature_1", "feature_2", "feature_3"]].to_frame().T
        return model.predict(data)[0]
    
    def predict(self, model_input):
        return model_input.apply(self.predict_for_device, axis=1)




# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Probamos el modelo

# COMMAND ----------

# Sección 9: Ejemplo de uso del modelo combinado sobre un subconjunto de datos
example_model = OriginDelegatingModel(device_to_model)
example_model.predict(combined_df.toPandas().head(20))
output_example

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC A partir de aquí, podemos registrar el modelo para que sea utilizado en la inferencia de todos los device_id desde una única instancia.

# COMMAND ----------

from mlflow.models.signature import infer_signature

input_example = combined_df.select("device_id", "feature_1", "feature_2", "feature_3").toPandas().head(5)
# Ejecuta el modelo sobre el ejemplo para obtener la salida
output_example = example_model.predict(input_example)

# Infieres la firma a partir del input y output
signature = infer_signature(input_example, output_example)

# Sección 10: Registro del modelo combinado en MLflow para su reutilización o despliegue
with mlflow.start_run():
    model = OriginDelegatingModel(device_to_model)
    mlflow.pyfunc.log_model(
        "model", 
        python_model=model,
        input_example=input_example,
        signature=signature
        )

# COMMAND ----------

