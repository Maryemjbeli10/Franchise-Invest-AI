"""
MLFLOW — SEGMENTATION (segmentation.py)
=========================================
Stockage local SQLite — pas besoin de serveur séparé.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, silhouette_score, accuracy_score)

# =========================
# CONFIGURATION MLFLOW
# =========================
MLFLOW_DB       = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Franchise_Segmentation"

mlflow.set_tracking_uri(MLFLOW_DB)
mlflow.set_experiment(EXPERIMENT_NAME)

os.makedirs("figures_segmentation", exist_ok=True)
os.makedirs("models", exist_ok=True)

# =========================
# CHARGEMENT & FEATURE ENGINEERING
# =========================
df_raw = pd.read_excel("data.xlsx")
print(f"Dataset chargé : {df_raw.shape[0]} lignes, {df_raw.shape[1]} colonnes")

def feature_engineering(df):
    df = df.copy()
    df["Formations"]         = df["Formations"].map({"oui":1,"non":0,"Oui":1,"Non":0}).fillna(0)
    df["Aide_auFinancement"] = df["Aide_auFinancement"].map({"oui":1,"non":0,"Oui":1,"Non":0}).fillna(0)
    df["surface_m2"]         = df["Surface_moyenne"].str.extract(r"(\d+)").astype(float)
    df["anciennete"]         = 2026 - df["date_creation"]
    df["est_france"]         = df["adresses"].str.contains("France", case=False, na=False).astype(int)
    df["ratio_apport_investissement"] = df["apport_personnel"] / df["investissement_global"]
    df["investissement_par_m2"]       = df["investissement_global"] / df["surface_m2"]
    def parse_h(h):
        try:
            hh, mm = map(int, str(h).strip().split(":"))
            return hh + mm / 60
        except:
            return np.nan
    if "heure_ouverture" in df.columns and "heure_fermeture" in df.columns:
        df["nb_heures_ouverture"] = (
            df["heure_fermeture"].apply(parse_h) - df["heure_ouverture"].apply(parse_h)
        ).apply(lambda x: x if x and x > 0 else np.nan)
    return df

def clean_numeric(series):
    if pd.api.types.is_numeric_dtype(series):
        return series
    return series.astype(str).str.extract(r"([\d\.]+)")[0].astype(float)

df         = feature_engineering(df_raw)
target_col = "CA_potentielApres2ans"

SEG_FEATURES = [c for c in [
    "CA_potentielApres2ans","investissement_global","apport_personnel",
    "ratio_apport_investissement","anciennete","Implantations","rating",
    "Redevance_fonctionnement","surface_m2","nb_heures_ouverture",
    "Formations","Aide_auFinancement","YOY_Sales","investissement_par_m2"
] if c in df.columns]

df_seg = df[SEG_FEATURES].copy()
for col in df_seg.columns:
    try:    df_seg[col] = clean_numeric(df_seg[col])
    except: df_seg[col] = pd.to_numeric(df_seg[col], errors="coerce")

imputer   = SimpleImputer(strategy="median")
scaler    = StandardScaler()
X_imputed = imputer.fit_transform(df_seg)
X_scaled  = scaler.fit_transform(X_imputed)

# =========================
# RUN MLFLOW — KMEANS
# =========================
print("\n" + "="*55)
print("MLflow — KMeans Clustering")

with mlflow.start_run(run_name="KMeans_Clustering") as kmeans_run:

    mlflow.set_tags({"project":"Franchise_Invest_AI","task":"clustering","algorithm":"KMeans"})
    mlflow.log_params({
        "K": 3, "n_init": 20, "max_iter": 500, "random_state": 42,
        "n_seg_features": len(SEG_FEATURES), "n_samples": len(df)
    })

    # Elbow + Silhouette
    inertias, silhouettes = [], []
    for k in range(2, 9):
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, lbl, sample_size=2000, random_state=42)
        silhouettes.append(sil)
        mlflow.log_metric("inertia",    km.inertia_, step=k)
        mlflow.log_metric("silhouette", sil,          step=k)
        print(f"  K={k} | Inertie={km.inertia_:,.0f} | Silhouette={sil:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(range(2,9), inertias,    "o-", color="#4C72B0", lw=2)
    axes[0].axvline(3, color="#E24B4A", linestyle="--", alpha=0.7, label="K=3")
    axes[0].set_title("Elbow"); axes[0].set_xlabel("K"); axes[0].legend()
    axes[1].plot(range(2,9), silhouettes, "s-", color="#55A868", lw=2)
    axes[1].axvline(3, color="#E24B4A", linestyle="--", alpha=0.7, label="K=3")
    axes[1].set_title("Silhouette"); axes[1].set_xlabel("K"); axes[1].legend()
    plt.tight_layout()
    ep = "figures_segmentation/mlflow_01_elbow_silhouette.png"
    plt.savefig(ep, dpi=150); plt.close()
    mlflow.log_artifact(ep)

    # KMeans K=3 final
    kmeans         = KMeans(n_clusters=3, random_state=42, n_init=20, max_iter=500)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df["cluster_id"] = cluster_labels

    sil_final = silhouette_score(X_scaled, cluster_labels, sample_size=3000, random_state=42)
    mlflow.log_metric("final_silhouette_score", round(sil_final, 4))
    mlflow.log_metric("final_inertia",          round(float(kmeans.inertia_), 2))
    print(f"\n  Silhouette final (K=3) : {sil_final:.3f}")

    # Nommage clusters
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    cluster_stats  = (df.groupby("cluster_id")[target_col]
                        .agg(["median","count"])
                        .reset_index()
                        .sort_values("median", ascending=False)
                        .reset_index(drop=True))

    LABEL_MAP_RAW = {
        cluster_stats.iloc[0]["cluster_id"]: "Haut Rendement",
        cluster_stats.iloc[1]["cluster_id"]: "Valeur Sûre",
        cluster_stats.iloc[2]["cluster_id"]: "Risque Élevé",
    }
    df["segment"] = df["cluster_id"].map(LABEL_MAP_RAW)
    COLORS = {"Haut Rendement":"#639922","Valeur Sûre":"#378ADD","Risque Élevé":"#E24B4A"}

    for seg, count in df["segment"].value_counts().items():
        mlflow.log_metric(f"segment_{seg.replace(' ','_')}_count", int(count))
        mlflow.log_metric(f"segment_{seg.replace(' ','_')}_pct",   round(count/len(df)*100, 2))
        med = df[df["segment"]==seg][target_col].median()
        mlflow.log_metric(f"ca_median_{seg.replace(' ','_')}", round(float(med), 2))

    # PCA 2D
    pca     = PCA(n_components=2, random_state=42)
    X_pca   = pca.fit_transform(X_scaled)
    df["pca1"] = X_pca[:,0]; df["pca2"] = X_pca[:,1]
    explained  = pca.explained_variance_ratio_
    mlflow.log_metric("pca_variance_total", round(float(sum(explained)), 4))

    fig, ax = plt.subplots(figsize=(10, 7))
    for seg, color in COLORS.items():
        mask = df["segment"] == seg
        ax.scatter(df.loc[mask,"pca1"], df.loc[mask,"pca2"],
                   c=color, label=seg, alpha=0.55, s=18, edgecolors="none")
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    for i,(cx,cy) in enumerate(centroids_pca):
        seg = LABEL_MAP_RAW[i]
        ax.scatter(cx,cy,c=COLORS[seg],s=220,marker="*",edgecolors="white",zorder=5)
        ax.annotate(seg,(cx,cy),textcoords="offset points",xytext=(10,8),
                    fontsize=10,fontweight="bold",color=COLORS[seg])
    ax.set_title(f"KMeans K=3 — PCA 2D | Silhouette={sil_final:.3f}")
    patches=[mpatches.Patch(color=c,label=l) for l,c in COLORS.items()]
    ax.legend(handles=patches); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    pp = "figures_segmentation/mlflow_02_clusters_pca2d.png"
    plt.savefig(pp, dpi=150); plt.close()
    mlflow.log_artifact(pp)

    joblib.dump(kmeans,  "models/kmeans_segmentation.pkl")
    joblib.dump(imputer, "models/seg_imputer.pkl")
    joblib.dump(scaler,  "models/seg_scaler.pkl")
    mlflow.log_artifact("models/kmeans_segmentation.pkl")

    kmeans_run_id = kmeans_run.info.run_id

# =========================
# RUNS MLFLOW — CLASSIFICATEURS
# =========================
print("\n" + "="*55)
print("MLflow — Classification supervisée")

CLF_FEATURES = [f for f in SEG_FEATURES if f != target_col]
X_clf        = df[CLF_FEATURES].copy()

for col in X_clf.columns:
    try:    X_clf[col] = clean_numeric(X_clf[col])
    except: X_clf[col] = pd.to_numeric(X_clf[col], errors="coerce")

le          = LabelEncoder()
y_encoded   = le.fit_transform(df["segment"])
imp_clf     = SimpleImputer(strategy="median")
X_clf_imp   = imp_clf.fit_transform(X_clf)

X_train, X_test, y_train, y_test = train_test_split(
    X_clf_imp, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

classifiers = {
    "RandomForest_Clf": {
        "model":  RandomForestClassifier(n_estimators=300, max_depth=12,
                                         random_state=42, n_jobs=-1, class_weight="balanced"),
        "params": {"n_estimators":300,"max_depth":12,"class_weight":"balanced"}
    },
    "GradientBoosting_Clf": {
        "model":  GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                              max_depth=5, random_state=42),
        "params": {"n_estimators":300,"learning_rate":0.05,"max_depth":5}
    },
}

best_clf=None; best_clf_acc=-1; best_clf_name=""

for name, cfg in classifiers.items():
    print(f"\nTraining {name} ...")
    with mlflow.start_run(run_name=name):

        mlflow.set_tags({"project":"Franchise_Invest_AI","task":"classification",
                         "linked_kmeans_run": kmeans_run_id})
        mlflow.log_params(cfg["params"])
        mlflow.log_params({"n_clf_features":len(CLF_FEATURES),"n_train":len(X_train),"cv_folds":5})

        cv_scores = cross_val_score(cfg["model"], X_clf_imp, y_encoded,
                                    cv=5, scoring="accuracy", n_jobs=-1)
        mlflow.log_metrics({
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std":  round(float(cv_scores.std()),  4),
        })

        cfg["model"].fit(X_train, y_train)
        y_pred   = cfg["model"].predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("test_accuracy", round(test_acc, 4))

        report = classification_report(y_test, y_pred, target_names=le.classes_,
                                       output_dict=True, zero_division=0)
        for cls_name, cls_m in report.items():
            if isinstance(cls_m, dict):
                sn = cls_name.replace(" ","_")
                mlflow.log_metrics({
                    f"{sn}_precision": round(cls_m.get("precision",0),4),
                    f"{sn}_recall":    round(cls_m.get("recall",0),4),
                    f"{sn}_f1":        round(cls_m.get("f1-score",0),4),
                })

        print(f"  CV Acc: {cv_scores.mean():.3f} ± {cv_scores.std():.3f} | Test: {test_acc:.3f}")

        # Matrice de confusion
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay(confusion_matrix(y_test,y_pred),
                               display_labels=le.classes_).plot(ax=ax,colorbar=False,cmap="Blues")
        ax.set_title(f"{name}  Accuracy={test_acc:.3f}"); ax.tick_params(axis="x",rotation=20)
        plt.tight_layout()
        cm_path = f"figures_segmentation/mlflow_{name}_confusion_matrix.png"
        plt.savefig(cm_path, dpi=120); plt.close()
        mlflow.log_artifact(cm_path)

        # Feature importance
# Feature importance — CORRIGÉ
# Feature importance — CORRECTION LIGNE 283
        if hasattr(cfg["model"], "feature_importances_"):
            # On crée un DataFrame avec les colonnes RÉELLES utilisées par le modèle
            # On utilise une liste de compréhension pour s'assurer que la longueur match
            feature_names = [f"Feature_{i}" for i in range(len(cfg["model"].feature_importances_))]
            
            # reconstruire les noms ou simplement utiliser des indices.
            # Le plus sûr pour éviter le crash :
            fi = pd.DataFrame({
                "importance": cfg["model"].feature_importances_
            })
            # On ajoute les noms de colonnes seulement si la taille correspond
            if len(CLF_FEATURES) == len(fi):
                fi["feature"] = CLF_FEATURES
            else:
                fi["feature"] = [f"Col_{i}" for i in range(len(fi))]

            fi = fi.sort_values("importance", ascending=False)

        sig = infer_signature(X_train, cfg["model"].predict(X_train))
        mlflow.sklearn.log_model(cfg["model"], artifact_path="model", signature=sig)

        if cv_scores.mean() > best_clf_acc:
            best_clf_acc=cv_scores.mean(); best_clf=cfg["model"]; best_clf_name=name

print(f"\nMeilleur classifieur : {best_clf_name} (CV Acc = {best_clf_acc:.3f})")

# Sauvegarde
joblib.dump(best_clf,"models/clf_segment.pkl")
joblib.dump(imp_clf, "models/clf_imputer.pkl")
joblib.dump(le,      "models/label_encoder.pkl")
with open("models/segmentation_config.json","w") as f:
    json.dump({"label_map":{int(k):v for k,v in LABEL_MAP_RAW.items()},
               "clf_features":CLF_FEATURES,"seg_features":SEG_FEATURES,
               "silhouette_score":round(sil_final,4),
               "best_classifier":best_clf_name,"cv_accuracy":round(best_clf_acc,4)},f,indent=2)

print("Segmentation DONE ✔  — mlflow.db mis à jour")
