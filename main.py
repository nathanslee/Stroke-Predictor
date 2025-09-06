"""
Stroke Prediction (Kaggle)
--------------------------
Goal: Predict stroke (1) vs no stroke (0) from demographics & health factors.

What this script does:
1) Loads 'healthcare-dataset-stroke-data.csv'
2) Cleans data: drop id, impute BMI, encode categoricals, scale numerics
3) Splits data (stratified)
4) Trains two models: LogisticRegression, RandomForest (both class_weight='balanced')
5) Evaluates: Accuracy, Precision, Recall, F1, ROC-AUC
6) Saves a results table to results_stroke.csv
7) Produces FIVE interpretable plots with matplotlib:
   - bar_ Stroke rate by age group
   - box_ BMI by stroke status
   - imp_ RandomForest feature importance
   - cm_  Confusion matrix of best model
   - roc_ ROC curves for both models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# ---------------- 1) Load & quick clean ----------------
CSV_PATH = "healthcare-dataset-stroke-data.csv"
df = pd.read_csv(CSV_PATH)

# Drop id if present
for col in ["id", "Id", "ID"]:
    if col in df.columns:
        df = df.drop(columns=[col])

# Ensure expected columns exist
required_cols = [
    "stroke", "age", "gender", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "bmi", "smoking_status"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}\nHave: {df.columns.tolist()}")

# Target and features
y = df["stroke"].astype(int)
X = df.drop(columns=["stroke"])

# ---------------- 2) Identify types & preprocessing ----------------
# Categorical = object/category; Numeric = the rest
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

# Impute numerics with median; scale numerics
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# One-hot encode categoricals; keep dense output
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ],
    remainder="drop"
)

# ---------------- 3) Split (stratified to preserve class ratio) ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- 4) Models ----------------
log_reg = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

rf = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(
        n_estimators=500, random_state=42, class_weight="balanced"
    ))
])

models = {"LogisticRegression": log_reg, "RandomForest": rf}

# ---------------- 5) Train + Evaluate ----------------
rows = []
roc_curves = {}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    # Both models expose predict_proba in these configurations
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    rows.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1": f1, "ROC_AUC": auc
    })

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_curves[name] = (fpr, tpr, auc)

results = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False)
print("\nModel performance (sorted by ROC_AUC):")
print(results.to_string(index=False))

# Save results
results.to_csv("results_stroke.csv", index=False)
print("\nSaved: results_stroke.csv")

# Determine best model by ROC_AUC
best_name = results.iloc[0]["Model"]
best_model = models[best_name]
y_pred_best = best_model.predict(X_test)

# ---------------- 6) Plots (matplotlib only; one figure each) ----------------
# Make an output directory for figures
outdir = Path("figures")
outdir.mkdir(exist_ok=True)

# A) Stroke rate by age group (bar chart)
age_bins = [0, 30, 40, 50, 60, 70, 120]
age_labels = ["<30", "30–39", "40–49", "50–59", "60–69", "70+"]
df_age = df.copy()
df_age["age_group"] = pd.cut(df_age["age"], bins=age_bins, labels=age_labels, right=False)
stroke_rate = df_age.groupby("age_group")["stroke"].mean()  # proportion with stroke

plt.figure()
plt.bar(stroke_rate.index.astype(str), stroke_rate.values)
plt.title("Stroke Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Stroke Proportion")
plt.tight_layout()
plt.savefig(outdir / "bar_stroke_rate_by_age_group.png", dpi=130)
plt.show()

# B) BMI by stroke status (boxplot)
# Prepare arrays
bmi_no = df.loc[df["stroke"] == 0, "bmi"].dropna().values
bmi_yes = df.loc[df["stroke"] == 1, "bmi"].dropna().values

plt.figure()
plt.boxplot([bmi_no, bmi_yes], labels=["No Stroke (0)", "Stroke (1)"], showmeans=True)
plt.title("BMI by Stroke Status")
plt.xlabel("Stroke Status")
plt.ylabel("BMI")
plt.tight_layout()
plt.savefig(outdir / "box_bmi_by_stroke.png", dpi=130)
plt.show()

# C) RandomForest feature importance (bar chart)
# Extract feature names after preprocessing
prep = best_model.named_steps["prep"]
# Build feature name list from transformers:
# numerics first
num_names = num_cols.copy()
# then categorical one-hot names
cat_onehot = prep.named_transformers_["cat"].named_steps["onehot"]
cat_names = list(cat_onehot.get_feature_names_out(cat_cols))
feature_names = num_names + cat_names

if best_name == "RandomForest":
    rf_est = best_model.named_steps["clf"]
else:
    # If LogisticRegression happened to be best, still show RF importances for interpretability
    rf_est = rf.named_steps["clf"]
    rf_est.fit(prep.transform(X_train), y_train)  # fit on preprocessed data to get importances

importances = rf_est.feature_importances_
# Sort top 12 for readability
idx = np.argsort(importances)[::-1][:12]
plt.figure(figsize=(8, 5))
plt.barh([feature_names[i] for i in idx][::-1], importances[idx][::-1])
plt.title("Top Feature Importances (RandomForest)")
plt.xlabel("Importance (Gini)")
plt.tight_layout()
plt.savefig(outdir / "imp_random_forest_top12.png", dpi=130)
plt.show()

# D) Confusion matrix for best model
cm = confusion_matrix(y_test, y_pred_best)
plt.figure()
plt.imshow(cm, aspect="auto")
plt.title(f"Confusion Matrix — {best_name}")
plt.xticks([0, 1], ["Pred 0", "Pred 1"])
plt.yticks([0, 1], ["True 0", "True 1"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.colorbar()
plt.tight_layout()
plt.savefig(outdir / "cm_best_model.png", dpi=130)
plt.show()

# E) ROC curves (both models)
plt.figure()
for name, (fpr, tpr, auc) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(outdir / "roc_curves.png", dpi=130)
plt.show()
