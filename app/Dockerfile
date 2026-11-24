# Imagen del tablero (Streamlit).
# Se construye desde la raíz del repo:  docker build -f app/Dockerfile -t churntel-dashboard .
FROM python:3.11-slim

WORKDIR /app

COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

COPY app/dashboard.py app/dashboard.py
COPY base_de_datos_proyecto.csv base_de_datos_proyecto.csv

ENV API_URL=http://api:8001
EXPOSE 8501
CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
