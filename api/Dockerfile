# Imagen de la API de churn.
# Se construye desde la raíz del repo:  docker build -f api/Dockerfile -t churntel-api .
FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY models/export_model.py models/export_model.py
COPY api/ api/
COPY base_de_datos_proyecto.csv base_de_datos_proyecto.csv

# Entrena y empaqueta el modelo final dentro de la imagen, con las mismas
# versiones de librerías que usará la API al cargarlo.
RUN python models/export_model.py --data base_de_datos_proyecto.csv \
    --out models/churn_model.joblib

EXPOSE 8001
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
