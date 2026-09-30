# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # Client Needs Attention — Logistic Classification
# MAGIC
# MAGIC Binary logistic regression to predict the `needs_attention` flag from `<catalog>.gold.client_worklist`, covering feature engineering, sklearn Pipeline preprocessing, evaluation, and MLflow experiment logging.

# COMMAND ----------

# DBTITLE 1,Configuration
# Catalog is supplied by the DAB job (base_parameters.catalog -> ${var.catalog}).
# Defaults to fe-bar-josed so the notebook also runs standalone in the dev workspace.
dbutils.widgets.text("catalog", "fe-bar-josed", "Unity Catalog")
dbutils.widgets.text("serving_endpoint", "client-needs-attention", "Serving endpoint")
catalog = dbutils.widgets.get("catalog")
serving_endpoint = dbutils.widgets.get("serving_endpoint")
print(f"Using catalog: {catalog} | serving endpoint: {serving_endpoint}")

# COMMAND ----------

# DBTITLE 1,Load Data
import pandas as pd
import numpy as np

# Load from Unity Catalog (catalog backtick-quoted to tolerate a hyphen in the name).
# Only raw client state is selected — the derived flags / priority_score are the rule that
# defines the target, so training on them would just memorise it and make what-if useless.
df_spark = spark.sql(f"""
  SELECT cw.client_id, cw.age, cw.state, cw.risk_profile,
         cw.portfolio_value, cw.num_accounts, cw.max_drift, cw.cash_weight,
         cw.taxable_unrealized_loss, cw.last_review_date,
         cw.flag_concentrated_drop AS has_concentrated_drop,
         coalesce(ira.has_traditional_ira, false) AS has_traditional_ira,
         cw.needs_attention
  FROM `{catalog}`.gold.client_worklist cw
  LEFT JOIN (
    SELECT client_id, bool_or(account_type = 'Traditional IRA') AS has_traditional_ira
    FROM `{catalog}`.gold.account_health
    GROUP BY client_id
  ) ira ON cw.client_id = ira.client_id
""")
df = df_spark.toPandas()

print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumn dtypes:")
print(df.dtypes.to_string())
print(f"\nMissing values:")
missing = df.isnull().sum()
missing_any = missing[missing > 0]
print(missing_any.to_string() if len(missing_any) > 0 else "  None")
display(df.head(5))

# COMMAND ----------

# DBTITLE 1,Feature Engineering & EDA
# Compute days since last review (relative to today — the app does the same when scoring)
reference_date = pd.Timestamp.today().normalize()
df["days_since_last_review"] = (reference_date - pd.to_datetime(df["last_review_date"])).dt.days

# Cast boolean inputs to int
bool_cols = ["has_concentrated_drop", "has_traditional_ira"]
for col in bool_cols:
    df[col] = df[col].astype(int)

# Define feature groups
numeric_features = [
    "age", "portfolio_value", "num_accounts", "max_drift",
    "cash_weight", "taxable_unrealized_loss", "days_since_last_review",
]
# Numerics as float so the serving signature accepts any JSON number from the app
df[numeric_features] = df[numeric_features].astype(float)
categorical_features = ["state", "risk_profile"]
all_features = numeric_features + bool_cols + categorical_features
target = "needs_attention"

# --- Class distribution ---
print("=== Class Distribution ===")
print(df[target].value_counts())
print(f"\nClass shares:")
print(df[target].value_counts(normalize=True).round(4))

# --- Correlation with target (leakage screen) ---
df_corr = df[numeric_features + bool_cols].copy()
df_corr["target_int"] = df[target].astype(int)
corr_with_target = (
    df_corr.corr()["target_int"]
    .drop("target_int")
    .sort_values(key=abs, ascending=False)
)
print(f"\n=== Feature-Target Correlations ===")
print(corr_with_target.round(4).to_string())

# COMMAND ----------

# DBTITLE 1,Train/Test Split & Model Training
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Prepare features and target
X = df[all_features].copy()
y = df[target]

# Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train set: {X_train.shape[0]:,} rows | Test set: {X_test.shape[0]:,} rows")

# Preprocessing pipeline
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("bool", "passthrough", bool_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

# Full pipeline: preprocessing → logistic regression
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
])

pipeline.fit(X_train, y_train)
print("Model training complete.")

# COMMAND ----------

# DBTITLE 1,Evaluation Metrics & Plots
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay,
)

# Predictions
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# Classification report
print("=== Evaluation Metrics ===")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# Plots: confusion matrix & ROC curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0], cmap="Blues")
axes[0].set_title("Confusion Matrix")

RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[1])
axes[1].set_title("ROC Curve")
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Feature Importance (Coefficients)
# Extract feature names and coefficients from the fitted pipeline
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
coefficients = pipeline.named_steps["classifier"].coef_[0]

coef_df = (
    pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    .assign(abs_coeff=lambda d: d["coefficient"].abs())
    .sort_values("abs_coeff", ascending=False)
    .drop(columns="abs_coeff")
    .reset_index(drop=True)
)

print(f"=== Logistic Regression Coefficients ({len(coef_df)} features) ===")
display(coef_df)

# Horizontal bar chart of top coefficients
top_n = min(20, len(coef_df))
top = coef_df.head(top_n).sort_values("coefficient")

fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
colors = ["#d9534f" if c < 0 else "#5cb85c" for c in top["coefficient"]]
ax.barh(top["feature"], top["coefficient"], color=colors)
ax.set_xlabel("Coefficient")
ax.set_title(f"Top {top_n} Logistic Regression Coefficients")
ax.axvline(x=0, color="black", linewidth=0.5)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,MLflow Experiment Logging
import mlflow
from mlflow.models import infer_signature

with mlflow.start_run(run_name="logistic_regression_v1") as run:
    # Log parameters
    mlflow.log_params({
        "model_type": "LogisticRegression",
        "max_iter": 1000,
        "test_size": 0.2,
        "random_state": 42,
        "n_features": len(all_features),
        "n_train_rows": X_train.shape[0],
        "n_test_rows": X_test.shape[0],
    })

    # Log metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=True),
        "recall": recall_score(y_test, y_pred, pos_label=True),
        "f1": f1_score(y_test, y_pred, pos_label=True),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    mlflow.log_metrics(metrics)

    # Log plots as artifacts
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm, cmap="Blues")
    ax_cm.set_title("Confusion Matrix")
    mlflow.log_figure(fig_cm, "confusion_matrix.png")
    plt.close(fig_cm)

    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax_roc)
    ax_roc.set_title("ROC Curve")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
    mlflow.log_figure(fig_roc, "roc_curve.png")
    plt.close(fig_roc)

    # Log model with signature. Serve predict_proba so the endpoint returns
    # [P(False), P(True)] per row — the app needs a probability for what-if scoring.
    signature = infer_signature(X_train.head(100), pipeline.predict_proba(X_train.head(100)))
    model_info = mlflow.sklearn.log_model(
        pipeline,
        name="logistic_regression_model",
        signature=signature,
        input_example=X_train.head(3),
        pyfunc_predict_fn="predict_proba",
    )

    print(f"MLflow Run ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nModel URI: {model_info.model_uri}")

# COMMAND ----------

# DBTITLE 1,Register Model in Unity Catalog
import mlflow

# Validate the artifact before registration
pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
assert pyfunc_model.metadata.signature is not None, "Model has no signature"
representative_input = pyfunc_model.input_example
assert representative_input is not None, "Model has no input example"

mlflow.models.predict(
    model_uri=model_info.model_uri,
    input_data=representative_input,
)
print("Artifact validation passed.")

# Register in Unity Catalog
mlflow.set_registry_uri("databricks-uc")
registered_model_name = f"{catalog}.gold.client_needs_attention"

registered_version = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=registered_model_name,
    await_registration_for=300,
)

model_version = registered_version.version
print(f"\nRegistered: {registered_model_name} (version {model_version})")
print(f"Model URI: models:/{registered_model_name}/{model_version}")

# COMMAND ----------

# DBTITLE 1,Deploy to Model Serving (what-if scoring for the app)
from datetime import timedelta
from mlflow.tracking import MlflowClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, ServedEntityInput, TrafficConfig, Route,
)

# Point the champion alias at the new version, then roll the endpoint to the same version
MlflowClient(registry_uri="databricks-uc").set_registered_model_alias(
    registered_model_name, "champion", model_version
)

served_name = f"client-needs-attention-v{model_version}"
served_entities = [ServedEntityInput(
    name=served_name,
    entity_name=registered_model_name,
    entity_version=str(model_version),
    workload_size="Small",
    scale_to_zero_enabled=True,
)]
traffic_config = TrafficConfig(routes=[Route(served_model_name=served_name, traffic_percentage=100)])

w = WorkspaceClient()
try:
    w.serving_endpoints.get(serving_endpoint)
    endpoint_exists = True
except NotFound:
    endpoint_exists = False

if endpoint_exists:
    print(f"Updating endpoint {serving_endpoint} -> version {model_version} ...")
    w.serving_endpoints.update_config_and_wait(
        name=serving_endpoint,
        served_entities=served_entities,
        traffic_config=traffic_config,
        timeout=timedelta(minutes=40),
    )
else:
    print(f"Creating endpoint {serving_endpoint} (version {model_version}) ...")
    w.serving_endpoints.create_and_wait(
        name=serving_endpoint,
        config=EndpointCoreConfigInput(served_entities=served_entities, traffic_config=traffic_config),
        timeout=timedelta(minutes=40),
    )

# Smoke test: score the input example through the live endpoint
resp = w.serving_endpoints.query(
    name=serving_endpoint,
    dataframe_records=X_test.head(3).to_dict("records"),
)
print(f"Endpoint {serving_endpoint} READY. Sample predictions: {resp.predictions}")
