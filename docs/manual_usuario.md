# Manual de usuario — Tablero ChurnTel

ChurnTel ayuda al equipo comercial de la operadora a responder la pregunta de negocio del proyecto: **¿qué clientes tienen mayor probabilidad de cancelar su servicio y a quiénes conviene contactar primero?**

Para entrar, abra en el navegador `http://<IP-del-servidor>:8501`.

## 1. Barra lateral (configuración y filtros)

- **Upload customer dataset (CSV):** permite cargar otra base de clientes con la misma estructura. Si no se carga nada, se usa la base del proyecto.
- **Estado de los datos:** "Dataset cargado correctamente" indica que se están usando datos reales. Si aparece *DEMO MODE*, los datos son simulados.
- **Estado de la API:** "API conectada: churn_model v1.0.0" indica que las predicciones vienen del modelo desplegado. Si aparece "API no disponible", avise al administrador.
- **Filtros:** tipo de contrato, servicio de internet y rango de antigüedad (meses). Todos los indicadores y gráficas se recalculan con los clientes filtrados.

## 2. Indicadores principales (parte superior)

Total de clientes, clientes que abandonaron, tasa de churn y clientes activos del grupo filtrado.

## 3. Pestañas

### 📈 Executive overview
Vista general: distribución activos/abandonos, tasa de churn por tipo de contrato, distribución de la antigüedad e ingreso mensual de los clientes activos. Sirve para dimensionar el problema.

### 👥 Customer analysis
Compara los cargos mensuales y totales de quienes se van y quienes se quedan, y la tasa de churn por servicio de internet. Sirve para entender qué perfil abandona más.

### 🚨 Churn risk
Aquí el **modelo** calcula la probabilidad de fuga de cada cliente filtrado:
- **Riesgo bajo** (< 30 %), **medio** (30–60 %) y **alto** (> 60 %).
- **Customers by risk level:** cuántos clientes hay en cada nivel.
- **Customer risk map:** cada punto es un cliente según antigüedad y cargo mensual, coloreado por riesgo. Al pasar el mouse se ve su identificador.
- **Customer prioritization:** lista de clientes de riesgo alto ordenada de mayor a menor probabilidad (`risk_score` = probabilidad × 100).
- **Download high-risk customers:** descarga esa lista en CSV para la campaña de retención.

### 🔮 Predict a customer
Para evaluar un cliente puntual (por ejemplo, durante una llamada):
1. Diligencie antigüedad, cargos, contrato, servicio de internet, método de pago, soporte técnico, seguridad en línea, facturación electrónica y si es adulto mayor.
2. Pulse **Predict**.
3. El tablero muestra la probabilidad de fuga, el nivel de riesgo (🟢 🟡 🔴) y una acción sugerida.

### 🤖 Model performance
Muestra los experimentos registrados en MLflow: número de corridas, mejor modelo por ROC-AUC, tabla comparativa (accuracy, precision, recall, F1, ROC-AUC) y una gráfica para comparar la métrica que elija. **Refresh MLflow results** actualiza los datos.

## 4. Cómo leer los resultados

- La probabilidad es una estimación, no una certeza. El modelo detecta cerca de **8 de cada 10 clientes que realmente se van** (recall 0,78), a cambio de marcar también a algunos que no se irían (precisión 0,50). Para retención es preferible contactar de más que perder clientes.
- Los clientes con contrato mes a mes, poca antigüedad, fibra óptica y pago con cheque electrónico tienden a tener mayor riesgo.

## 5. Preguntas frecuentes

| Pregunta | Respuesta |
|---|---|
| ¿Por qué cambió la lista de riesgo? | Cambiaron los filtros de la barra lateral. |
| La pestaña Churn risk muestra un aviso de "indicador heurístico" | La API no respondió; se muestra una regla de respaldo. Avise al administrador. |
| ¿Puedo usar mi propia base? | Sí, con el botón de carga de CSV, siempre que tenga las columnas de la base Telco. |
