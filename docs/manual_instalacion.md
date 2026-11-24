# Manual de instalación — ChurnTel (API + tablero)

Este manual explica cómo instalar y poner en marcha la API de predicción de churn y el tablero en una máquina Linux (probado en AWS EC2, Ubuntu 24.04). Los dos componentes corren en contenedores Docker.

## 1. Requisitos

| Elemento | Detalle |
|---|---|
| Máquina | Ubuntu 22.04 o 24.04, mínimo 2 GB de RAM (en EC2: `t2.medium` o `t3.small`) y 20 GB de disco |
| Puertos abiertos | 22 (SSH), 8001 (API), 8501 (tablero); 5000 solo si se usa MLflow |
| Software | Git, Docker y Docker Compose |
| Datos | `base_de_datos_proyecto.csv` (Telco Customer Churn, 7.043 clientes) |

En EC2, los puertos se abren en **Instancia → Seguridad → Grupo de seguridad → Editar reglas de entrada**: regla *TCP personalizado* con origen `0.0.0.0/0` para 8001 y 8501.

## 2. Instalar Docker

```bash
sudo apt update
sudo apt install -y git docker.io docker-compose-v2
sudo usermod -aG docker $USER
newgrp docker
docker --version && docker compose version
```

## 3. Descargar el código

```bash
git clone https://github.com/RogerBeleno/Proyecto-Grupal-Telecomunicaciones.git
cd Proyecto-Grupal-Telecomunicaciones
```

## 4. Obtener los datos

El dataset está versionado con DVC en S3. Con acceso al bucket:

```bash
pip install "dvc[s3]"
dvc pull
```

Sin acceso al bucket, se descarga el mismo archivo desde Kaggle:

```bash
pip install kagglehub
python3 - <<'EOF'
import kagglehub, shutil, glob
p = kagglehub.dataset_download("blastchar/telco-customer-churn")
shutil.copy(glob.glob(f"{p}/*.csv")[0], "base_de_datos_proyecto.csv")
EOF
md5sum base_de_datos_proyecto.csv   # debe dar 0f9de68e012bd3aed5fa7cdc9fc421af
```

El archivo debe quedar en la raíz del repositorio.

## 5. Construir y levantar los contenedores

```bash
docker compose up -d --build
docker ps
```

Se crean dos contenedores:

| Contenedor | Qué hace | Puerto |
|---|---|---|
| `churntel-api` | Entrena y empaqueta el modelo final durante la construcción de la imagen y sirve las predicciones (FastAPI) | 8001 |
| `churntel-dashboard` | Tablero en Streamlit que consulta la API | 8501 |

## 6. Verificar

- API: `http://<IP-PUBLICA>:8001/docs` (Swagger). Probar `GET /health` y `POST /predict`.
- Tablero: `http://<IP-PUBLICA>:8501`. En la barra lateral debe aparecer **"API conectada: churn_model v1.0.0"**.

Desde la terminal:

```bash
curl http://localhost:8001/health
curl -X POST http://localhost:8001/predict -H "Content-Type: application/json" \
  -d '{"customers":[{"tenure":4,"monthly_charges":85,"contract":"Month-to-month","internet_service":"Fiber optic"}]}'
```

## 7. MLflow (opcional, pestaña "Model performance")

La pestaña de desempeño lee los experimentos del servidor MLflow que corre en la misma máquina, fuera de Docker:

```bash
python3 -m venv ~/env && source ~/env/bin/activate
pip install -r requirements.txt
nohup mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri sqlite:///mlflow.db \
  --disable-security-middleware > mlflow.log 2>&1 &
python models/train_mlflow.py --data base_de_datos_proyecto.csv --tracking http://localhost:5000
```

## 8. Operación

| Acción | Comando |
|---|---|
| Ver logs | `docker compose logs -f api` o `docker compose logs -f dashboard` |
| Reiniciar | `docker compose restart` |
| Actualizar tras un cambio en el repo | `git pull && docker compose up -d --build` |
| Apagar | `docker compose down` |

Al terminar, detener la instancia EC2 con **Stop** (no *Terminate*) para poder volver a lanzarla. Al encenderla de nuevo cambia la IP pública; los contenedores arrancan solos (`restart: unless-stopped`).

## 9. Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| El navegador no carga la página | Falta abrir el puerto en el grupo de seguridad, o se escribió `https://` en vez de `http://` |
| `COPY base_de_datos_proyecto.csv` falla al construir | Falta el archivo de datos en la raíz (paso 4) |
| El tablero dice "API no disponible" | Revisar `docker ps` y `docker compose logs api` |
| `permission denied` al usar docker | Ejecutar `newgrp docker` o volver a conectarse |
| "Model performance" muestra error | El servidor MLflow no está corriendo (paso 7); el resto del tablero funciona igual |
