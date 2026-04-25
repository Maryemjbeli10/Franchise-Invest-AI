"""
MLFLOW — REGRESSION (ai.py)
============================
Stockage local SQLite — pas besoin de serveur séparé.
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# =========================
# CONFIGURATION MLFLOW
# =========================
MLFLOW_DB       = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Franchise_Regression"

mlflow.set_tracking_uri(MLFLOW_DB)
mlflow.set_experiment(EXPERIMENT_NAME)

os.makedirs("figures", exist_ok=True)
os.makedirs("models",  exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_excel("data.xlsx")
print("Dataset shape:", df.shape)

# =========================
# FEATURE ENGINEERING
# =========================
def feature_engineering(df):
    df = df.copy()
    df["Formations"]        = df["Formations"].map({"oui":1,"non":0,"Oui":1,"Non":0}).fillna(0)
    df["Aide_auFinancement"]= df["Aide_auFinancement"].map({"oui":1,"non":0,"Oui":1,"Non":0}).fillna(0)
    df["surface_m2"]        = df["Surface_moyenne"].str.extract(r"(\d+)").astype(float)
    df["anciennete"]        = 2026 - df["date_creation"]
    df["est_france"]        = df["adresses"].str.contains("France", case=False, na=False).astype(int)
    df["ratio_apport_investissement"] = df["apport_personnel"] / df["investissement_global"]
    df["investissement_par_m2"]       = df["investissement_global"] / df["surface_m2"]

    def parse_hours(h):
        try:
            hh, mm = map(int, str(h).strip().split(":"))
            return hh + mm / 60
        except:
            return np.nan

    if "heure_ouverture" in df.columns and "heure_fermeture" in df.columns:
        df["heure_ouv_num"]       = df["heure_ouverture"].apply(parse_hours)
        df["heure_ferm_num"]      = df["heure_fermeture"].apply(parse_hours)
        df["nb_heures_ouverture"] = df["heure_ferm_num"] - df["heure_ouv_num"]
        df["nb_heures_ouverture"] = df["nb_heures_ouverture"].apply(lambda x: x if x > 0 else np.nan)
        df.drop(columns=["heure_ouverture","heure_fermeture","heure_ouv_num","heure_ferm_num"],
                errors="ignore", inplace=True)
    return df

df = feature_engineering(df)

# =========================
# OUTLIERS + LOG-TRANSFORM
# =========================
target = "CA_potentielApres2ans"

def remove_outliers_iqr(df, col, factor=3.0):
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR    = Q3 - Q1
    before = len(df)
    df_c   = df[(df[col] >= Q1 - factor*IQR) & (df[col] <= Q3 + factor*IQR)]
    print(f"Outliers supprimés : {before - len(df_c)} lignes")
    return df_c

df            = remove_outliers_iqr(df, target, factor=3.0)
target_log    = f"{target}_log"
df[target_log]= np.log1p(df[target])

# =========================
# SPLIT + PREPROCESSING
# =========================
X = df.drop(columns=[target, target_log], errors="ignore")
y = df[target_log]

num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc",  RobustScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# MODÈLES
# =========================
models = {
    "RandomForest": {
        "model":  RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "params": {"n_estimators": 300, "random_state": 42}
    },
    "GradientBoosting": {
        "model":  GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42),
        "params": {"n_estimators": 300, "learning_rate": 0.05, "random_state": 42}
    },
    "ExtraTrees": {
        "model":  ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "params": {"n_estimators": 300, "random_state": 42}
    },
    "XGBoost": {
        "model":  XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=42, n_jobs=-1, verbosity=0),
        "params": {"n_estimators": 500, "learning_rate": 0.05,
                   "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8}
    }
}

# =========================
# BOUCLE ENTRAINEMENT + MLFLOW
# =========================
results       = {}
best_pipeline = None
best_score    = -999
best_name     = ""
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, cfg in models.items():
    print(f"\n{'='*50}\nTraining {name} ...")

    with mlflow.start_run(run_name=name):

        mlflow.set_tags({"model_type": name, "project": "Franchise_Invest_AI", "task": "regression"})
        mlflow.log_params(cfg["params"])
        mlflow.log_params({"test_size": 0.2, "cv_folds": 5,
                           "n_train": len(X_train), "n_test": len(X_test)})

        pipeline  = Pipeline([("prep", preprocessor), ("model", cfg["model"])])
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)

        mlflow.log_metrics({
            "cv_r2_mean": round(float(cv_scores.mean()), 4),
            "cv_r2_std":  round(float(cv_scores.std()),  4),
            "cv_r2_min":  round(float(cv_scores.min()),  4),
            "cv_r2_max":  round(float(cv_scores.max()),  4),
        })
        print(f"  CV R² : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        pipeline.fit(X_train, y_train)
        preds       = np.expm1(pipeline.predict(X_test))
        y_test_orig = np.expm1(y_test)

        r2   = r2_score(y_test_orig, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test_orig, preds)))
        mae  = float(mean_absolute_error(y_test_orig, preds))

        mlflow.log_metrics({"test_r2": round(r2,4), "test_rmse": round(rmse,2), "test_mae": round(mae,2)})
        results[name] = {"r2": r2, "rmse": rmse, "mae": mae,
                         "cv_mean": float(cv_scores.mean()), "cv_std": float(cv_scores.std())}
        print(f"  Test → R²:{r2:.3f} | RMSE:{rmse:,.0f} | MAE:{mae:,.0f}")

        # Réel vs Prédit
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_test_orig, preds, alpha=0.4, s=15, color="#4C72B0", edgecolors="none")
        lims = [min(y_test_orig.min(), preds.min()), max(y_test_orig.max(), preds.max())]
        ax.plot(lims, lims, "r--", lw=1.5, label="Parfait")
        ax.set_xlabel("CA réel (€)"); ax.set_ylabel("CA prédit (€)")
        ax.set_title(f"{name} — Réel vs Prédit  R²={r2:.3f}")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x/1000)}k"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x/1000)}k"))
        ax.legend(); plt.tight_layout()
        p1 = f"figures/mlflow_{name}_pred_vs_real.png"
        plt.savefig(p1, dpi=120); plt.close()
        mlflow.log_artifact(p1)

        # Résidus
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.scatter(preds, y_test_orig.values - preds, alpha=0.4, s=15, color="#DD8452", edgecolors="none")
        ax2.axhline(0, color="r", linestyle="--", lw=1.5)
        ax2.set_xlabel("CA prédit (€)"); ax2.set_ylabel("Résidu (€)")
        ax2.set_title(f"{name} — Résidus"); plt.tight_layout()
        p2 = f"figures/mlflow_{name}_residuals.png"
        plt.savefig(p2, dpi=120); plt.close()
        mlflow.log_artifact(p2)

        # Log modèle
        sig = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(pipeline, artifact_path="model", signature=sig)

        if cv_scores.mean() > best_score:
            best_score = cv_scores.mean(); best_pipeline = pipeline; best_name = name

# =========================
# RUN SUMMARY
# =========================
print(f"\nBEST MODEL : {best_name}  (CV R² = {best_score:.3f})")

with mlflow.start_run(run_name="Summary_Comparison"):
    mlflow.set_tag("type", "summary")
    mlflow.log_param("best_model", best_name)
    mlflow.log_metric("best_cv_r2", round(best_score, 4))
    for n, res in results.items():
        mlflow.log_metrics({f"{n}_cv_r2": round(res["cv_mean"],4),
                            f"{n}_rmse":  round(res["rmse"],2),
                            f"{n}_mae":   round(res["mae"],2)})

    names  = list(results.keys())
    colors = ["#4C72B0","#DD8452","#55A868","#C44E52"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bars  = axes[0].bar(names, [results[n]["cv_mean"] for n in names], color=colors, alpha=0.8)
    bars2 = axes[1].bar(names, [results[n]["rmse"]/1000 for n in names], color=colors, alpha=0.8)
    for bar, n in zip(bars, names):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                     f"{results[n]['cv_mean']:.3f}", ha="center", fontsize=9)
    for bar, n in zip(bars2, names):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                     f"{results[n]['rmse']/1000:.1f}k", ha="center", fontsize=9)
    axes[0].set_title("CV R² moyen"); axes[0].set_ylim(0,1)
    axes[1].set_title("RMSE test (k€)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15); ax.grid(True, alpha=0.3, axis="y")
    plt.suptitle(f"Best: {best_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    sp = "figures/mlflow_summary_comparison.png"
    plt.savefig(sp, dpi=150); plt.close()
    mlflow.log_artifact(sp)

joblib.dump(best_pipeline, "models/best_model.pkl")
print("Regression DONE ✔  — mlflow.db créé dans le dossier projet")
