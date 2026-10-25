# Proyecto-Grupal-Telecomunicaciones
Proyecto MIAD Despliegue soluciones analíticas

 Maqueta del prototipo — ChurnTel

Maqueta (mockup) del tablero para la Entrega 1 del proyecto de *Despliegue de soluciones analíticas* (MIAD).

**Pregunta de negocio:** ¿qué clientes de la operadora tienen mayor probabilidad de cancelar su servicio (churn) y qué características los explican, para priorizar acciones de retención?

**Datos:** Telco Customer Churn (IBM) — 7.043 clientes, 21 variables (contrato, antigüedad, servicios, cargos, método de pago, `Churn`).


| Vista | Elemento | Relación con la pregunta de negocio | Tipo |

1. Panorama del churn | A. Filtros (contrato, internet, pago, antigüedad) | Segmentar la cartera por las variables más asociadas a la fuga | Descriptivo |
  B. KPIs (clientes, tasa de churn, ingreso en riesgo, clientes en riesgo alto) | Dimensionar el problema y su impacto económico | Descriptivo |
  C. Churn por contrato y por internet | Identificar qué tipo de cliente abandona más | Descriptivo |
  D. Churn por antigüedad y cargo mensual | Entender cuándo se van y si el precio influye | Descriptivo |
2. Predecir un cliente | E. Formulario con las variables del cliente | Entrada del modelo (`POST /predict`) | Predictivo |
   F. Probabilidad de fuga con semáforo (bajo < 30 %, medio 30–60 %, alto > 60 %) | Respuesta directa: ¿este cliente se va a ir? | Predictivo |
   G. Factores principales y acción sugerida | Explicar el resultado y convertirlo en una acción de retención | Predictivo |
3. Clientes en riesgo | H. Lista priorizada y exportable | Saber a quién contactar primero (inferencia por lotes) | Predictivo |

## Contrato esperado de la API

```json
POST /predict
{ "tenure": 4, "MonthlyCharges": 85, "Contract": "Month-to-month",
  "InternetService": "Fiber optic", "PaymentMethod": "Electronic check",
  "TechSupport": "No", "OnlineSecurity": "No", "PaperlessBilling": "Yes", "SeniorCitizen": 0 }

200 OK
{ "probabilidad_fuga": 0.87, "riesgo": "alto",
  "factores": [ { "factor": "Contrato mes a mes", "efecto": "aumenta" } ],
  "modelo": "churn_clf v0.1" }

