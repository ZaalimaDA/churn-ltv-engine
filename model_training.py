import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")
 
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
 
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from xgboost import XGBClassifier
 
# ── 0. SETUP ────────────────────────────────────────────────────────────────
load_dotenv()
engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
sns.set_theme(style="whitegrid", palette="muted")
OUTPUT_DIR   = "model_outputs"
MODELS_DIR   = "saved_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
 
# ════════════════════════════════════════════════════════════════════════════
# PART 1 — LOAD & PREPARE DATA
# ════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("PART 1 — DATA PREPARATION")
print("=" * 65)
 
df = pd.read_sql("SELECT * FROM customers_features", engine)
df["total_charges"] = pd.to_numeric(
    df["total_charges"], errors="coerce").fillna(0)
print(f"\nLoaded customers_features : {df.shape[0]} rows, {df.shape[1]} cols")
 
# Target
df["churn_binary"] = (df["churn"] == "Yes").astype(int)
print(f"Churn distribution        :\n{df['churn_binary'].value_counts()}")
print(f"Churn rate                : {df['churn_binary'].mean()*100:.1f}%")
 
# ── 5 engineered features ──────────────────────────────────────────────────
ENGINEERED = [
    "charge_per_tenure",
    "service_count",
    "charge_to_value_ratio",
    "support_dependency_score",
    "tenure_contract_risk",
]
 
# ── original numeric features ──────────────────────────────────────────────
NUMERIC = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "senior_citizen",
]
 
# ── encode remaining categoricals ──────────────────────────────────────────
df["gender_enc"]      = (df["gender"] == "Male").astype(int)
df["partner_enc"]     = (df["partner"] == "Yes").astype(int)
df["dependents_enc"]  = (df["dependents"] == "Yes").astype(int)
df["paperless_enc"]   = (df["paperless_billing"] == "Yes").astype(int)
contract_map          = {"Month-to-month": 0, "One year": 1, "Two year": 2}
df["contract_enc"]    = df["contract"].map(contract_map)
 
EXTRA_CATS = [
    "gender_enc", "partner_enc",
    "dependents_enc", "paperless_enc", "contract_enc",
]
 
# ── service binary columns ─────────────────────────────────────────────────
service_raw = [
    "phone_service", "multiple_lines", "online_security",
    "online_backup", "device_protection", "tech_support",
    "streaming_tv", "streaming_movies",
]
for col in service_raw:
    df[col] = df[col].replace(
        {"No internet service": "No", "No phone service": "No"})
    df[f"{col}_enc"] = (df[col] == "Yes").astype(int)
 
SERVICE_ENC = [f"{c}_enc" for c in service_raw]
 
# ── one-hot: internet_service & payment_method ─────────────────────────────
df = pd.get_dummies(
    df,
    columns=["internet_service", "payment_method"],
    prefix=["internet", "payment"],
    drop_first=False,
)
ONEHOT = [c for c in df.columns
          if c.startswith("internet_") or c.startswith("payment_")]
 
# ── final feature list ─────────────────────────────────────────────────────
ALL_FEATURES = ENGINEERED + NUMERIC + SERVICE_ENC + EXTRA_CATS + ONEHOT
 
X = df[ALL_FEATURES].copy()
y = df["churn_binary"].copy()
 
# cast booleans to int (get_dummies can produce bool dtype)
bool_cols = X.select_dtypes(include="bool").columns.tolist()
X[bool_cols] = X[bool_cols].astype(int)
 
print(f"\nFeature matrix shape      : {X.shape}")
print(f"Engineered features used  : {len(ENGINEERED)}  → {ENGINEERED}")
print(f"Total features            : {len(ALL_FEATURES)}")
print(f"Null values in X          : {X.isnull().sum().sum()}")
 
# ── stratified 80/20 split ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain : {X_train.shape[0]} rows  ({y_train.mean()*100:.1f}% churn)")
print(f"Test  : {X_test.shape[0]}  rows  ({y_test.mean()*100:.1f}% churn)")
 
# ── scale for Logistic Regression only ────────────────────────────────────
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n" + "=" * 65)
print("PART 2 — MODEL TRAINING")
print("=" * 65)

results = {}

# ── MODEL 1 : Logistic Regression ─────────────────────────────────────────
print("\n--- Model 1: Logistic Regression ---")
lr = LogisticRegression(
    C=1.0,
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
    solver="lbfgs",
)
lr.fit(X_train_scaled, y_train)
 
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
 
results["Logistic Regression"] = dict(
    model     = lr,
    y_pred    = y_pred_lr,
    y_prob    = y_prob_lr,
    precision = precision_score(y_test, y_pred_lr),
    recall    = recall_score(y_test, y_pred_lr),
    f1        = f1_score(y_test, y_pred_lr),
    roc_auc   = roc_auc_score(y_test, y_prob_lr),
)
print(f"  Precision : {results['Logistic Regression']['precision']:.4f}")
print(f"  Recall    : {results['Logistic Regression']['recall']:.4f}")
print(f"  F1-Score  : {results['Logistic Regression']['f1']:.4f}")
print(f"  ROC-AUC   : {results['Logistic Regression']['roc_auc']:.4f}")
 
joblib.dump(lr,     f"{MODELS_DIR}/logistic_regression.pkl")
joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
print(f"  Saved: {MODELS_DIR}/logistic_regression.pkl")
print(f"  Saved: {MODELS_DIR}/scaler.pkl")

# ── MODEL 2 : Random Forest ───────────────────────────────────────────────
print("\n--- Model 2: Random Forest ---")
rf = RandomForestClassifier(
    n_estimators   = 200,
    max_depth       = 10,
    min_samples_split = 5,
    min_samples_leaf  = 2,
    class_weight    = "balanced",
    random_state    = 42,
    n_jobs          = -1,
)
rf.fit(X_train, y_train)
 
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
 
results["Random Forest"] = dict(
    model     = rf,
    y_pred    = y_pred_rf,
    y_prob    = y_prob_rf,
    precision = precision_score(y_test, y_pred_rf),
    recall    = recall_score(y_test, y_pred_rf),
    f1        = f1_score(y_test, y_pred_rf),
    roc_auc   = roc_auc_score(y_test, y_prob_rf),
)
print(f"  Precision : {results['Random Forest']['precision']:.4f}")
print(f"  Recall    : {results['Random Forest']['recall']:.4f}")
print(f"  F1-Score  : {results['Random Forest']['f1']:.4f}")
print(f"  ROC-AUC   : {results['Random Forest']['roc_auc']:.4f}")
 
joblib.dump(rf, f"{MODELS_DIR}/random_forest.pkl")
print(f"  Saved: {MODELS_DIR}/random_forest.pkl")

# ── MODEL 3 : XGBoost ────────────────────────────────────────────────────
print("\n--- Model 3: XGBoost ---")
neg        = (y_train == 0).sum()
pos        = (y_train == 1).sum()
scale_pos  = neg / pos
print(f"  scale_pos_weight = {scale_pos:.2f}  "
      f"(neg={neg}, pos={pos})")
 
xgb = XGBClassifier(
    n_estimators      = 200,
    max_depth         = 5,
    learning_rate     = 0.1,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = scale_pos,
    eval_metric       = "logloss",
    random_state      = 42,
    verbosity         = 0,
)
xgb.fit(X_train, y_train)
 
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
 
results["XGBoost"] = dict(
    model     = xgb,
    y_pred    = y_pred_xgb,
    y_prob    = y_prob_xgb,
    precision = precision_score(y_test, y_pred_xgb),
    recall    = recall_score(y_test, y_pred_xgb),
    f1        = f1_score(y_test, y_pred_xgb),
    roc_auc   = roc_auc_score(y_test, y_prob_xgb),
)
print(f"  Precision : {results['XGBoost']['precision']:.4f}")
print(f"  Recall    : {results['XGBoost']['recall']:.4f}")
print(f"  F1-Score  : {results['XGBoost']['f1']:.4f}")
print(f"  ROC-AUC   : {results['XGBoost']['roc_auc']:.4f}")
 
joblib.dump(xgb, f"{MODELS_DIR}/xgboost.pkl")
print(f"  Saved: {MODELS_DIR}/xgboost.pkl")

# ════════════════════════════════════════════════════════════════════════════
# PART 3 — EVALUATION & VISUALISATION
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PART 3 — EVALUATION & VISUALISATION")
print("=" * 65)
 
model_names = list(results.keys())
COLORS      = {
    "Logistic Regression": "#2196F3",
    "Random Forest"      : "#4CAF50",
    "XGBoost"            : "#F44336",
}
 
# ── 3a. Metrics comparison table (printed) ────────────────────────────────
print("\n3a. Metrics Summary:")
print(f"{'Model':<22} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
print("-" * 60)
for name, v in results.items():
    print(f"{name:<22} {v['precision']:>10.4f} {v['recall']:>8.4f} "
          f"{v['f1']:>8.4f} {v['roc_auc']:>9.4f}")
 
# ── 3b. Full classification reports ──────────────────────────────────────
print("\n3b. Classification Reports:")
for name, v in results.items():
    print(f"\n{name}:")
    print(classification_report(
        y_test, v["y_pred"],
        target_names=["Not Churned", "Churned"]))
 
# ── PLOT 1: Confusion matrices ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (name, v) in enumerate(results.items()):
    cm = confusion_matrix(y_test, v["y_pred"])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
        xticklabels=["Predicted No", "Predicted Yes"],
        yticklabels=["Actual No",    "Actual Yes"],
        linewidths=0.5,
    )
    tn, fp, fn, tp = cm.ravel()
    axes[i].set_title(
        f"{name}\nTP={tp}  FP={fp}  FN={fn}  TN={tn}",
        fontweight="bold", fontsize=11,
    )
plt.suptitle("Confusion Matrices — All 3 Models",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_confusion_matrices.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"\n📊 Saved: {OUTPUT_DIR}/01_confusion_matrices.png")
 
# ── PLOT 2: ROC curves ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
for name, v in results.items():
    fpr, tpr, _ = roc_curve(y_test, v["y_prob"])
    ax.plot(fpr, tpr, color=COLORS[name], linewidth=2.5,
            label=f"{name}  (AUC = {v['roc_auc']:.4f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1,
        label="Random Classifier (AUC = 0.50)")
ax.set_title("ROC Curves — All 3 Models",
             fontsize=14, fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right", fontsize=11)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_roc_curves.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Saved: {OUTPUT_DIR}/02_roc_curves.png")
 
# ── PLOT 3: Metrics bar chart comparison ─────────────────────────────────
metric_keys   = ["precision", "recall", "f1", "roc_auc"]
metric_labels = ["Precision", "Recall", "F1-Score", "ROC-AUC"]
bar_colors    = [COLORS[m] for m in model_names]
 
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
    vals = [results[m][key] for m in model_names]
    bars = axes[i].bar(
        [m.replace(" ", "\n") for m in model_names],
        vals, color=bar_colors, edgecolor="white", width=0.5,
    )
    axes[i].set_title(label, fontweight="bold")
    axes[i].set_ylim(0, 1.12)
    axes[i].set_ylabel("Score")
    for bar, val in zip(bars, vals):
        axes[i].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}", ha="center",
            fontweight="bold", fontsize=10,
        )
plt.suptitle("Model Performance Comparison",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_metrics_comparison.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Saved: {OUTPUT_DIR}/03_metrics_comparison.png")
 
# ── PLOT 4: Feature importances (RF & XGBoost) ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
for ax, (model_obj, name) in zip(
        axes, [(rf, "Random Forest"), (xgb, "XGBoost")]):
    importances = model_obj.feature_importances_
    feat_imp    = pd.Series(
        importances, index=ALL_FEATURES).sort_values(ascending=True)
    top15       = feat_imp.tail(15)
 
    # highlight engineered features in red, rest in blue
    imp_colors = [
        "#F44336" if f in ENGINEERED else "#2196F3"
        for f in top15.index
    ]
    top15.plot(kind="barh", ax=ax, color=imp_colors, edgecolor="white")
    ax.set_title(f"{name} — Top 15 Feature Importances\n"
                 f"(red = engineered features)", fontweight="bold")
    ax.set_xlabel("Importance Score")
    for j, val in enumerate(top15.values):
        ax.text(val + 0.001, j, f"{val:.4f}",
                va="center", fontsize=8)
 
plt.suptitle("Feature Importances: RF vs XGBoost",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_feature_importances.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Saved: {OUTPUT_DIR}/04_feature_importances.png")
 
# ── PLOT 5: 5-Fold Cross-Validation ──────────────────────────────────────
print("\n3c. 5-Fold Cross-Validation (F1-Score):")
cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
for name, v in results.items():
    if name == "Logistic Regression":
        scores = cross_val_score(
            v["model"], X_train_scaled, y_train,
            cv=cv, scoring="f1")
    else:
        scores = cross_val_score(
            v["model"], X_train, y_train,
            cv=cv, scoring="f1")
    cv_results[name] = scores
    print(f"  {name:<22}: mean={scores.mean():.4f}  "
          f"std={scores.std():.4f}  "
          f"folds={[round(s,3) for s in scores]}")
 
fig, ax = plt.subplots(figsize=(9, 5))
cv_data = [cv_results[m] for m in model_names]
bp      = ax.boxplot(
    cv_data, patch_artist=True,
    tick_labels=[m.replace(" ", "\n") for m in model_names],
)
for patch, color in zip(bp["boxes"], bar_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title("5-Fold Cross-Validation — F1-Score",
             fontweight="bold")
ax.set_ylabel("F1-Score")
ax.set_ylim(0.3, 0.95)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_cross_validation.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Saved: {OUTPUT_DIR}/05_cross_validation.png")
 
# ── PLOT 6: Engineered feature importances spotlight ─────────────────────
# Shows only the 5 engineered features side-by-side for both tree models
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (model_obj, name) in zip(
        axes, [(rf, "Random Forest"), (xgb, "XGBoost")]):
    importances = model_obj.feature_importances_
    feat_series = pd.Series(importances, index=ALL_FEATURES)
    eng_imp     = feat_series[ENGINEERED].sort_values(ascending=True)
    eng_colors  = ["#F44336", "#E91E63", "#FF5722", "#FF9800", "#FFC107"]
    eng_imp.plot(kind="barh", ax=ax,
                 color=eng_colors[:len(eng_imp)], edgecolor="white")
    ax.set_title(f"{name}\nEngineered Features Importance",
                 fontweight="bold")
    ax.set_xlabel("Importance Score")
    for j, val in enumerate(eng_imp.values):
        ax.text(val + 0.0005, j, f"{val:.4f}",
                va="center", fontsize=9)
 
plt.suptitle("Engineered Feature Importances — Both Tree Models",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_engineered_feature_spotlight.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Saved: {OUTPUT_DIR}/06_engineered_feature_spotlight.png")

# ════════════════════════════════════════════════════════════════════════════
# PART 4 — SAVE RESULTS TO POSTGRESQL
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PART 4 — SAVE RESULTS TO POSTGRESQL")
print("=" * 65)
 
# Metrics table
metrics_rows = [
    dict(
        model     = name,
        precision = round(v["precision"], 4),
        recall    = round(v["recall"],    4),
        f1_score  = round(v["f1"],        4),
        roc_auc   = round(v["roc_auc"],   4),
    )
    for name, v in results.items()
]
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_sql("model_metrics", engine,
                  if_exists="replace", index=False)
print("\nSaved → model_metrics table")
print(metrics_df.to_string(index=False))
 
# Best model predictions
best_name = max(results, key=lambda m: results[m]["f1"])
print(f"\nBest model by F1-Score : {best_name}")
 
preds_df = X_test.copy().reset_index(drop=True)
preds_df["actual_churn"]      = y_test.values
preds_df["predicted_churn"]   = results[best_name]["y_pred"]
preds_df["churn_probability"] = results[best_name]["y_prob"].round(4)
preds_df.to_sql("model_predictions", engine,
                if_exists="replace", index=False)
print(f"Saved → model_predictions table  ({len(preds_df)} rows)")

# ════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════
lr_r  = results["Logistic Regression"]
rf_r  = results["Random Forest"]
xgb_r = results["XGBoost"]
best  = results[best_name]