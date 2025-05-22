# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ### Visión general
# MAGIC
# MAGIC Este notebook te mostrará cómo crear y consultar una tabla o DataFrame a partir de un archivo que hayas subido al DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) es el Sistema de Archivos de Databricks que te permite almacenar datos para consultarlos dentro de Databricks. Este notebook asume que ya tienes un archivo dentro del DBFS desde el cual deseas leer.
# MAGIC
# MAGIC Este notebook está escrito en **Python**, por lo que el tipo de celda predeterminado es Python. Sin embargo, puedes usar otros lenguajes utilizando la sintaxis `%LANGUAGE`. Se admiten Python, Scala, SQL y R.
# MAGIC

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/airbnb.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# Create a view or table

temp_table_name = "airbnb"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC
# MAGIC select * from `airbnb`

# COMMAND ----------

# With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
# Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
# To do so, choose your table name and uncomment the bottom line.

permanent_table_name = "airbnb"

df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

