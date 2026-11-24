import json
import os

# Si MLflow no responde, fallar rápido en vez de reintentar varios minutos.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

# ---------------------------------------------------------
# Configuracion
# ---------------------------------------------------------
st.set_page_config(
    page_title="ChurnTel | Customer Analytics",
    page_icon="📊",
    layout="wide",
)

API_URL = os.getenv("API_URL", "http://localhost:8001").rstrip("/")

st.markdown(
    """
    <style>
    .stApp { background-color: #F5F7FB; }
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: #172B4D; }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E5EAF1;
        padding: 16px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Datos
# ---------------------------------------------------------
@st.cache_data
def generate_demo_data():
    rng = np.random.default_rng(42)
    n = 7000
    churn = np.array([1] * 1869 + [0] * (n - 1869))
    rng.shuffle(churn)

    tenure = np.where(
        churn == 1,
        rng.integers(1, 45, n),
        rng.integers(1, 73, n),
    )
    monthly_charges = np.clip(rng.normal(75, 25, n), 20, 150)
    admin_tickets = rng.poisson(np.where(churn == 1, 0.9, 0.3))
    tech_tickets = rng.poisson(np.where(churn == 1, 1.5, 0.5))

    contract = np.where(
        churn == 1,
        rng.choice(
            ["Month-to-month", "One year", "Two year"],
            n,
            p=[0.80, 0.15, 0.05],
        ),
        rng.choice(
            ["Month-to-month", "One year", "Two year"],
            n,
            p=[0.40, 0.30, 0.30],
        ),
    )

    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(1, n + 1)],
            "churn": churn,
            "tenure": tenure,
            "monthly_charges": monthly_charges.round(2),
            "total_charges": (monthly_charges * tenure).round(2),
            "num_admin_tickets": admin_tickets,
            "num_tech_tickets": tech_tickets,
            "contract": contract,
            "internet_service": rng.choice(
                ["DSL", "Fiber optic", "No internet"], n,
                p=[0.35, 0.50, 0.15],
            ),
            "payment_method": rng.choice(
                ["Electronic check", "Credit card", "Bank transfer"], n
            ),
        }
    )


def prepare_data(data):
    df = data.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"([a-z0-9])([A-Z])", r"\1_\2", regex=True)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    aliases = {
        "churn": ["churn", "exited", "abandono"],
        "tenure": ["tenure", "customer_tenure"],
        "monthly_charges": ["monthly_charges", "monthly_charge"],
        "total_charges": ["total_charges", "total_charge"],
        "num_admin_tickets": ["num_admin_tickets", "admin_tickets"],
        "num_tech_tickets": ["num_tech_tickets", "tech_tickets"],
        "contract": ["contract", "contract_type"],
        "internet_service": ["internet_service", "internet_type"],
    }

    for standard, alternatives in aliases.items():
        if standard not in df.columns:
            for alternative in alternatives:
                if alternative in df.columns:
                    df = df.rename(columns={alternative: standard})
                    break

    required = ["churn", "tenure", "monthly_charges", "total_charges"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {', '.join(missing)}")

    churn_text = df["churn"].astype(str).str.strip().str.lower()
    churn_map = {
        "yes": 1, "no": 0, "true": 1, "false": 0,
        "1": 1, "0": 0, "1.0": 1, "0.0": 0,
    }
    df["churn"] = churn_text.map(churn_map)

    for column in [
        "tenure", "monthly_charges", "total_charges",
        "num_admin_tickets", "num_tech_tickets",
    ]:
        if column not in df.columns:
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required).copy()
    df["churn"] = df["churn"].astype(int)
    if not df["churn"].isin([0, 1]).all():
        raise ValueError("La columna churn debe contener Yes/No o 1/0.")

    defaults = {
        "contract": "Not available",
        "internet_service": "Not available",
        "payment_method": "Not available",
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    if "customer_id" not in df.columns:
        df["customer_id"] = [f"CUST-{i:05d}" for i in range(len(df))]

    return df


# ---------------------------------------------------------
# MLflow
# ---------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def get_mlflow_runs(tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("churn_telecomunicaciones")
    if experiment is None:
        return pd.DataFrame()

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=500,
        order_by=["metrics.roc_auc DESC"],
    )

    records = []
    for run in runs:
        metrics = run.data.metrics
        params = run.data.params
        records.append(
            {
                "Run": run.data.tags.get("mlflow.runName", run.info.run_id[:8]),
                "Status": run.info.status,
                "Accuracy": metrics.get("accuracy"),
                "Precision": metrics.get("precision"),
                "Recall": metrics.get("recall"),
                "F1-score": metrics.get("f1"),
                "ROC-AUC": metrics.get("roc_auc"),
                "Model": params.get("modelo", "Not recorded"),
                "Run ID": run.info.run_id,
                "Start time": pd.to_datetime(
                    run.info.start_time, unit="ms", errors="coerce"
                ),
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------
# API del modelo
# ---------------------------------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def api_health(api_url):
    response = requests.get(f"{api_url}/health", timeout=5)
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner="Consultando el modelo en la API...")
def api_predict(api_url, customers_json):
    response = requests.post(
        f"{api_url}/predict",
        json={"customers": json.loads(customers_json)},
        timeout=60,
    )
    response.raise_for_status()
    return pd.DataFrame(response.json()["predicciones"])


# ---------------------------------------------------------
# Sidebar: fuentes y filtros
# ---------------------------------------------------------
st.sidebar.title("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("Upload customer dataset (CSV)", type=["csv"])
local_file = "base_de_datos_proyecto.csv"

try:
    if uploaded_file is not None:
        df = prepare_data(pd.read_csv(uploaded_file))
        demo_mode = False
    elif os.path.exists(local_file):
        df = prepare_data(pd.read_csv(local_file))
        demo_mode = False
    else:
        df = generate_demo_data()
        demo_mode = True
except Exception as error:
    st.error(f"No fue posible cargar los datos: {error}")
    st.stop()

if demo_mode:
    st.sidebar.warning("DEMO MODE: datos simulados, no representan resultados reales.")
else:
    st.sidebar.success("Dataset cargado correctamente.")

st.sidebar.header("Customer filters")
contracts = sorted(df["contract"].dropna().astype(str).unique())
services = sorted(df["internet_service"].dropna().astype(str).unique())

selected_contracts = st.sidebar.multiselect(
    "Contract type", contracts, default=contracts
)
selected_services = st.sidebar.multiselect(
    "Internet service", services, default=services
)

min_tenure = int(df["tenure"].min())
max_tenure = int(df["tenure"].max())
if min_tenure == max_tenure:
    tenure_range = (min_tenure, max_tenure)
else:
    tenure_range = st.sidebar.slider(
        "Customer tenure (months)",
        min_value=min_tenure,
        max_value=max_tenure,
        value=(min_tenure, max_tenure),
    )

filtered_df = df[
    df["contract"].astype(str).isin(selected_contracts)
    & df["internet_service"].astype(str).isin(selected_services)
    & df["tenure"].between(*tenure_range)
].copy()

# ---------------------------------------------------------
# Header and KPIs
# ---------------------------------------------------------
st.title("📊 ChurnTel | Customer Analytics")
st.markdown(
    "Customer intelligence platform | Customer retention and churn analytics"
)
if demo_mode:
    st.info(
        "Estás visualizando datos de demostración. "
        "Carga el CSV real para consultar los indicadores del proyecto."
    )

if filtered_df.empty:
    st.warning("No hay clientes que coincidan con los filtros seleccionados.")
    st.stop()

total_customers = len(filtered_df)
churn_customers = int(filtered_df["churn"].sum())
active_customers = total_customers - churn_customers
churn_rate = churn_customers / total_customers * 100
monthly_revenue = filtered_df.loc[
    filtered_df["churn"] == 0, "monthly_charges"
].sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total customers", f"{total_customers:,}")
k2.metric("Churned customers", f"{churn_customers:,}")
k3.metric("Churn rate", f"{churn_rate:.1f}%")
k4.metric("Active customers", f"{active_customers:,}")

st.divider()

try:
    model_info = api_health(API_URL)
    st.sidebar.success(
        f"API conectada: {model_info['modelo']} v{model_info['version']}"
    )
except Exception:
    model_info = None
    st.sidebar.error(f"API no disponible en {API_URL}")

tab1, tab2, tab3, tab5, tab4 = st.tabs(
    [
        "📈 Executive overview", "👥 Customer analysis", "🚨 Churn risk",
        "🔮 Predict a customer", "🤖 Model performance",
    ]
)

# ---------------------------------------------------------
# Tab 1: Executive overview
# ---------------------------------------------------------
with tab1:
    st.subheader("Customer overview")
    left, right = st.columns(2)

    with left:
        distribution = pd.DataFrame(
            {
                "Status": ["Active", "Churned"],
                "Customers": [active_customers, churn_customers],
            }
        )
        fig = px.pie(
            distribution, names="Status", values="Customers",
            hole=0.6, title="Customer distribution"
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        contract_summary = (
            filtered_df.groupby("contract", as_index=False)["churn"].mean()
        )
        contract_summary["Churn rate (%)"] = contract_summary["churn"] * 100
        fig = px.bar(
            contract_summary, x="contract", y="Churn rate (%)",
            text_auto=".1f", title="Churn rate by contract",
            labels={"contract": "Contract"},
        )
        st.plotly_chart(fig, width="stretch")

    fig = px.histogram(
        filtered_df, x="tenure", color="churn", nbins=30,
        barmode="overlay", opacity=0.75,
        title="Customer tenure distribution",
        labels={"tenure": "Tenure (months)", "churn": "Churn status"},
    )
    st.plotly_chart(fig, width="stretch")
    st.metric("Monthly revenue from active customers", f"${monthly_revenue:,.2f}")

# ---------------------------------------------------------
# Tab 2: Customer analysis
# ---------------------------------------------------------
with tab2:
    st.subheader("Customer behavior analysis")
    left, right = st.columns(2)

    with left:
        fig = px.box(
            filtered_df, x="churn", y="monthly_charges", color="churn",
            title="Monthly charges by churn status",
            labels={"churn": "Churn status", "monthly_charges": "Monthly charges"},
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        fig = px.box(
            filtered_df, x="churn", y="total_charges", color="churn",
            title="Total charges by churn status",
            labels={"churn": "Churn status", "total_charges": "Total charges"},
        )
        st.plotly_chart(fig, width="stretch")

    tickets = filtered_df.copy()
    tickets["total_tickets"] = (
        tickets["num_admin_tickets"] + tickets["num_tech_tickets"]
    )
    ticket_summary = (
        tickets.groupby("churn", as_index=False)["total_tickets"].mean()
    )
    fig = px.bar(
        ticket_summary, x="churn", y="total_tickets", color="churn",
        text_auto=".2f", title="Average support tickets by churn status",
        labels={"churn": "Churn status", "total_tickets": "Average tickets"},
    )
    st.plotly_chart(fig, width="stretch")

    service_summary = (
        filtered_df.groupby("internet_service", as_index=False)
        .agg(customers=("churn", "size"), churn_rate=("churn", "mean"))
    )
    service_summary["churn_rate"] *= 100
    fig = px.bar(
        service_summary, x="internet_service", y="churn_rate",
        text_auto=".1f", title="Churn rate by internet service",
        labels={
            "internet_service": "Internet service",
            "churn_rate": "Churn rate (%)",
        },
    )
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------
# Tab 3: Churn risk (heuristic, not model prediction)
# ---------------------------------------------------------
with tab3:
    st.subheader("Customer risk analysis")

    risk_df = filtered_df.copy()
    risk_df["total_tickets"] = (
        risk_df["num_admin_tickets"] + risk_df["num_tech_tickets"]
    )
    try:
        payload = risk_df.drop(columns=["churn"]).to_json(orient="records")
        preds = api_predict(API_URL, payload)
        risk_df["churn_probability"] = preds["probabilidad_fuga"].to_numpy()
        risk_df["risk_score"] = (risk_df["churn_probability"] * 100).round(1)
        risk_df["risk_level"] = (
            preds["riesgo"]
            .map({"bajo": "Low", "medio": "Medium", "alto": "High"})
            .to_numpy()
        )
        st.caption(
            "Probabilidad de fuga calculada por el modelo desplegado en la API "
            f"({API_URL}). Riesgo bajo < 30 %, medio 30–60 %, alto > 60 %."
        )
    except Exception as error:
        st.warning(
            "No fue posible consultar la API del modelo; se muestra un "
            f"indicador heurístico de respaldo. Detalle: {error}"
        )
        risk_df["risk_score"] = (
            (risk_df["tenure"] <= 12).astype(int) * 30
            + (risk_df["total_tickets"] >= 2).astype(int) * 30
            + (risk_df["monthly_charges"] >= 90).astype(int) * 20
            + (risk_df["contract"] == "Month-to-month").astype(int) * 20
        )
        risk_df["risk_level"] = pd.cut(
            risk_df["risk_score"],
            bins=[-1, 30, 60, 100],
            labels=["Low", "Medium", "High"],
        ).astype(str)
    risk_counts = (
        risk_df["risk_level"].value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
        .rename_axis("Risk level").reset_index(name="Customers")
    )

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            risk_counts, x="Risk level", y="Customers", color="Risk level",
            title="Customers by risk level",
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        fig = px.scatter(
            risk_df, x="tenure", y="monthly_charges", color="risk_level",
            hover_data=["customer_id", "total_tickets"],
            title="Customer risk map",
        )
        st.plotly_chart(fig, width="stretch")

    high_risk = risk_df[risk_df["risk_level"] == "High"].sort_values(
        "risk_score", ascending=False
    )
    st.subheader("Customer prioritization")
    columns_to_show = [
        "customer_id", "tenure", "monthly_charges", "total_tickets",
        "risk_score", "risk_level", "churn",
    ]
    st.dataframe(
        high_risk[columns_to_show].head(50),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download high-risk customers",
        data=high_risk.to_csv(index=False).encode("utf-8"),
        file_name="high_risk_customers.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------
# Tab 5: Predict a single customer through the API
# ---------------------------------------------------------
with tab5:
    st.subheader("Predict churn for a customer")
    st.caption(
        "Ingresa los datos del cliente. La predicción la hace el modelo "
        "desplegado en la API (POST /predict)."
    )

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            tenure = st.number_input("Tenure (months)", 0, 100, 4)
            monthly = st.number_input("Monthly charges", 0.0, 200.0, 85.0)
            total = st.number_input(
                "Total charges", 0.0, 10000.0, float(4 * 85.0)
            )
        with c2:
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            internet = st.selectbox(
                "Internet service", ["Fiber optic", "DSL", "No"]
            )
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check", "Mailed check",
                    "Bank transfer (automatic)", "Credit card (automatic)",
                ],
            )
        with c3:
            tech_support = st.selectbox("Tech support", ["No", "Yes"])
            online_security = st.selectbox("Online security", ["No", "Yes"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
        submitted = st.form_submit_button("Predict")

    if submitted:
        customer = {
            "tenure": tenure, "monthly_charges": monthly,
            "total_charges": total, "contract": contract,
            "internet_service": internet, "payment_method": payment,
            "tech_support": tech_support, "online_security": online_security,
            "paperless_billing": paperless,
            "senior_citizen": 1 if senior == "Yes" else 0,
        }
        try:
            result = api_predict(API_URL, json.dumps([customer])).iloc[0]
            probability = result["probabilidad_fuga"]
            level = result["riesgo"]
            icon = {"bajo": "🟢", "medio": "🟡", "alto": "🔴"}[level]
            a, b = st.columns(2)
            a.metric("Churn probability", f"{probability:.1%}")
            b.metric("Risk level", f"{icon} {level.capitalize()}")
            st.progress(float(probability))
            if level == "alto":
                st.error(
                    "Cliente con alta probabilidad de fuga: priorizar contacto "
                    "y ofrecer una tarifa diferenciada o un contrato más largo."
                )
            elif level == "medio":
                st.warning("Riesgo moderado: incluir en campañas de fidelización.")
            else:
                st.success("Riesgo bajo: mantener seguimiento normal.")
        except Exception as error:
            st.error(f"No fue posible obtener la predicción de la API: {error}")

# ---------------------------------------------------------
# Tab 4: MLflow model performance
# ---------------------------------------------------------
with tab4:
    st.subheader("Machine learning model performance")
    default_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    tracking_uri = st.text_input("MLflow tracking URI", value=default_uri)

    if st.button("Refresh MLflow results"):
        get_mlflow_runs.clear()

    try:
        runs_df = get_mlflow_runs(tracking_uri.strip())
        if runs_df.empty:
            st.warning(
                "No se encontraron corridas para el experimento "
                "'churn_telecomunicaciones' en esta ubicación de MLflow."
            )
        else:
            metric_columns = [
                "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"
            ]
            runs_df = runs_df.sort_values("ROC-AUC", ascending=False)
            best = runs_df.iloc[0]

            a, b, c = st.columns(3)
            a.metric("Registered runs", len(runs_df))
            b.metric("Best model", str(best["Run"]))
            c.metric(
                "Best ROC-AUC",
                f"{best['ROC-AUC']:.4f}" if pd.notna(best["ROC-AUC"]) else "N/A",
            )

            st.markdown("**Model comparison**")
            st.dataframe(
                runs_df[
                    ["Run", "Model", *metric_columns, "Status"]
                ].style.format(
                    {column: "{:.4f}" for column in metric_columns},
                    na_rep="N/A",
                ),
                width="stretch",
                hide_index=True,
            )

            metric_choice = st.selectbox(
                "Metric to compare",
                metric_columns,
                index=4,
            )
            chart_df = runs_df.dropna(subset=[metric_choice])
            if not chart_df.empty:
                fig = px.bar(
                    chart_df, x="Run", y=metric_choice, color="Run",
                    title=f"Model comparison: {metric_choice}",
                    text_auto=".4f",
                )
                fig.update_layout(showlegend=False, xaxis_tickangle=-25)
                st.plotly_chart(fig, width="stretch")

            st.caption(
                "Las métricas corresponden a las corridas recuperadas de MLflow. "
                "El mejor modelo se ordena por ROC-AUC."
            )
    except Exception as error:
        st.error(
            "No fue posible consultar MLflow. Verifica que el servidor esté "
            f"activo y que la URI sea accesible. Detalle: {error}"
        )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()
st.caption("ChurnTel | Customer Analytics | Telecom Analytics Project")
