FROM apache/airflow:2.9.0-python3.10

# Instalación de dependencias del sistema (si las necesitas)
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
 && apt-get clean

USER airflow

# Copiamos los requirements
COPY requirements.txt /requirements.txt

# Instalamos las dependencias de Python
RUN pip install --no-cache-dir -r /requirements.txt
