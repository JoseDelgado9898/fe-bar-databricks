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
catalog = dbutils.widgets.get("catalog")
print(f"Using catalog: {catalog}")

# COMMAND ----------

# DBTITLE 1,Load Data
import pandas as pd
import numpy as np

# Load from Unity Catalog (catalog backtick-quoted to tolerate a hyphen in the name)
df_spark = spark.sql(f"SELECT * FROM `{catalog}`.gold.client_worklist")
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
# Compute days since last review
reference_date = pd.Timestamp("2026-09-23")
df["days_since_last_review"] = (reference_date - pd.to_datetime(df["last_review_date"])).dt.days

# Cast boolean flags to int
bool_cols = [
    "flag_concentrated_drop", "flag_drift", "flag_tax_loss_harvest",
    "flag_rmd", "flag_cash_drag", "flag_review_overdue",
]
for col in bool_cols:
    df[col] = df[col].astype(int)

# Define feature groups
numeric_features = [
    "age", "portfolio_value", "num_accounts", "max_drift",
    "cash_weight", "taxable_unrealized_loss", "priority_score",
    "days_since_last_review",
]
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

    # Log model with signature
    signature = infer_signature(X_train.head(100), pipeline.predict(X_train.head(100)))
    model_info = mlflow.sklearn.log_model(
        pipeline,
        name="logistic_regression_model",
        signature=signature,
        input_example=X_train.head(3),
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
