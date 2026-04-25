"""
FRANCHISE INVEST AI — SEGMENTATION INTELLIGENTE
KMeans Clustering + Classification supervisée
Labels : "Haut Rendement" | "Valeur Sûre" | "Risque Élevé"
Fichier indépendant — ne modifie pas main.py
"""

# =========================
# 1. IMPORTS
# =========================
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # supprime le warning Windows subprocess
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, silhouette_score)
from sklearn.pipeline import Pipeline


# =========================
# 2. DOSSIERS
# =========================
os.makedirs("figures_segmentation", exist_ok=True)
os.makedirs("models", exist_ok=True)


# =========================
# 3. CHARGEMENT & FEATURE ENGINEERING
# (copie légère depuis main.py — aucune modification de main.py)
# =========================
df_raw = pd.read_excel("data.xlsx")
print(f"Dataset chargé : {df_raw.shape[0]} lignes, {df_raw.shape[1]} colonnes")

def feature_engineering(df):
    df = df.copy()
    df["Formations"] = df["Formations"].map(
        {"oui":1,"non":0,"Oui":1,"Non":0}).fillna(0)
    df["Aide_auFinancement"] = df["Aide_auFinancement"].map(
        {"oui":1,"non":0,"Oui":1,"Non":0}).fillna(0)
    df["surface_m2"] = df["Surface_moyenne"].str.extract(r"(\d+)").astype(float)
    df["anciennete"] = 2026 - df["date_creation"]
    df["est_france"] = df["adresses"].str.contains(
        "France", case=False, na=False).astype(int)
    df["ratio_apport_investissement"] = (
        df["apport_personnel"] / df["investissement_global"])
    df["investissement_par_m2"] = (
        df["investissement_global"] / df["surface_m2"])

    def parse_h(h):
        try:
            hh, mm = map(int, str(h).strip().split(":"))
            return hh + mm / 60
        except:
            return np.nan

    if "heure_ouverture" in df.columns and "heure_fermeture" in df.columns:
        df["nb_heures_ouverture"] = (
            df["heure_fermeture"].apply(parse_h) -
            df["heure_ouverture"].apply(parse_h)
        ).apply(lambda x: x if x and x > 0 else np.nan)

    return df

df = feature_engineering(df_raw)
target_col = "CA_potentielApres2ans"
print(f"Feature engineering OK — shape : {df.shape}")


# =========================
# 4. FEATURES DE SEGMENTATION
# On sélectionne les features les plus pertinentes pour distinguer
# la rentabilité et le risque d'une franchise
# =========================
SEG_FEATURES = [
    "CA_potentielApres2ans",        # performance principale
    "investissement_global",         # mise de départ
    "apport_personnel",              # solidité financière
    "ratio_apport_investissement",   # levier financier
    "anciennete",                    # maturité réseau
    "Implantations",                 # taille réseau
    "rating",                        # satisfaction client
    "Redevance_fonctionnement",      # coût récurrent
    "surface_m2",                    # taille opération
    "nb_heures_ouverture",           # intensité opérationnelle
    "Formations",                    # support franchiseur
    "Aide_auFinancement",            # accompagnement financier
    "YOY_Sales",                     # croissance des ventes
    "investissement_par_m2",         # efficacité surface
]

# Garder uniquement les colonnes disponibles
SEG_FEATURES = [c for c in SEG_FEATURES if c in df.columns]
print(f"Features de segmentation utilisées ({len(SEG_FEATURES)}) :", SEG_FEATURES)

df_seg = df[SEG_FEATURES].copy()


# =========================
# 5. NETTOYAGE NUMÉRIQUE + PREPROCESSING POUR CLUSTERING
# Certaines colonnes contiennent des strings comme "5%", "8 %", "10.5"
# → on extrait la valeur numérique avant d'imputer
# =========================
def clean_numeric(series):
    """Extrait le premier nombre flottant d'une colonne mixte (ex: '5%' → 5.0)."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    return (series.astype(str)
                  .str.extract(r"([\d\.]+)")[0]
                  .astype(float))

for col in df_seg.columns:
    try:
        df_seg[col] = clean_numeric(df_seg[col])
    except Exception:
        df_seg[col] = pd.to_numeric(df_seg[col], errors="coerce")

# Vérification
non_numeric = [c for c in df_seg.columns
               if not pd.api.types.is_numeric_dtype(df_seg[c])]
if non_numeric:
    print(f"  Colonnes non-numériques supprimées : {non_numeric}")
    df_seg.drop(columns=non_numeric, inplace=True)
    SEG_FEATURES = list(df_seg.columns)
else:
    print(f"  Toutes les colonnes sont numériques ✔")

imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()

X_imputed = imputer.fit_transform(df_seg)
X_scaled  = scaler.fit_transform(X_imputed)

print(f"Préprocessing clustering OK — shape : {X_scaled.shape}")


# =========================
# 6. CHOIX DU K — MÉTHODE ELBOW + SILHOUETTE
# =========================
print("\nRecherche du K optimal (Elbow + Silhouette)...")
inertias   = []
silhouettes = []
K_range    = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels, sample_size=2000,
                                        random_state=42))
    print(f"  K={k} | Inertie={km.inertia_:,.0f} | Silhouette={silhouettes[-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(list(K_range), inertias, "o-", color="#4C72B0", linewidth=2)
axes[0].axvline(3, color="#E24B4A", linestyle="--", alpha=0.7, label="K=3 choisi")
axes[0].set_xlabel("Nombre de clusters K")
axes[0].set_ylabel("Inertie")
axes[0].set_title("Méthode Elbow — choix de K")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(K_range), silhouettes, "s-", color="#55A868", linewidth=2)
axes[1].axvline(3, color="#E24B4A", linestyle="--", alpha=0.7, label="K=3 choisi")
axes[1].set_xlabel("Nombre de clusters K")
axes[1].set_ylabel("Score Silhouette")
axes[1].set_title("Score Silhouette — qualité des clusters")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures_segmentation/01_elbow_silhouette.png", dpi=150)
plt.close()
print("Figure 01 — Elbow + Silhouette sauvegardée ✔")


# =========================
# 7. KMEANS FINAL — K=3
# =========================
print("\nEntraînement KMeans K=3...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=20, max_iter=500)
cluster_labels = kmeans.fit_predict(X_scaled)
df["cluster_id"] = cluster_labels

sil_final = silhouette_score(X_scaled, cluster_labels, sample_size=3000,
                              random_state=42)
print(f"Score Silhouette final (K=3) : {sil_final:.3f}")


# =========================
# 8. INTERPRÉTATION & NOMMAGE DES CLUSTERS
# Nettoyage numérique de la target sur df complet (pour groupby fiables)
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
# On analyse le CA médian de chaque cluster pour attribuer
# automatiquement les labels métier
# =========================
cluster_stats = df.groupby("cluster_id")[target_col].agg(
    ["median", "mean", "std", "count"]).reset_index()
cluster_stats.columns = ["cluster_id", "CA_median", "CA_mean", "CA_std", "count"]
cluster_stats = cluster_stats.sort_values("CA_median", ascending=False).reset_index(drop=True)

print("\nStatistiques par cluster (trié par CA médian décroissant) :")
print(cluster_stats.to_string(index=False))

# Mapping automatique : rang 0 = plus haut CA, rang 2 = plus bas
LABEL_MAP_RAW = {
    cluster_stats.iloc[0]["cluster_id"]: "Haut Rendement",
    cluster_stats.iloc[1]["cluster_id"]: "Valeur Sûre",
    cluster_stats.iloc[2]["cluster_id"]: "Risque Élevé",
}

df["segment"] = df["cluster_id"].map(LABEL_MAP_RAW)

COLORS = {
    "Haut Rendement": "#639922",
    "Valeur Sûre":    "#378ADD",
    "Risque Élevé":   "#E24B4A",
}

print("\nMapping clusters → labels :")
for cid, label in LABEL_MAP_RAW.items():
    row = cluster_stats[cluster_stats["cluster_id"] == cid].iloc[0]
    print(f"  Cluster {int(cid)} → {label:15s} | "
          f"CA médian={row['CA_median']:,.0f}€ | n={int(row['count'])}")


# =========================
# 9. VISUALISATION — PCA 2D DES CLUSTERS
# =========================
print("\nPCA 2D pour visualisation...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df["pca1"] = X_pca[:, 0]
df["pca2"] = X_pca[:, 1]

explained = pca.explained_variance_ratio_
print(f"Variance expliquée par PCA : PC1={explained[0]:.1%}, PC2={explained[1]:.1%}, "
      f"Total={sum(explained):.1%}")

fig, ax = plt.subplots(figsize=(10, 7))
for seg, color in COLORS.items():
    mask = df["segment"] == seg
    ax.scatter(df.loc[mask, "pca1"], df.loc[mask, "pca2"],
               c=color, label=seg, alpha=0.55, s=18, edgecolors="none")

centroids_pca = pca.transform(kmeans.cluster_centers_)
for i, (cx, cy) in enumerate(centroids_pca):
    seg = LABEL_MAP_RAW[i]
    ax.scatter(cx, cy, c=COLORS[seg], s=220, marker="*",
               edgecolors="white", linewidths=1.5, zorder=5)
    ax.annotate(seg, (cx, cy), textcoords="offset points",
                xytext=(10, 8), fontsize=10, fontweight="bold",
                color=COLORS[seg])

ax.set_xlabel(f"Composante principale 1 ({explained[0]:.1%} variance)")
ax.set_ylabel(f"Composante principale 2 ({explained[1]:.1%} variance)")
ax.set_title("Segmentation des franchises — KMeans K=3 (PCA 2D)")
patches = [mpatches.Patch(color=c, label=l) for l, c in COLORS.items()]
ax.legend(handles=patches, loc="upper right")
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("figures_segmentation/02_clusters_pca2d.png", dpi=150)
plt.close()
print("Figure 02 — PCA 2D clusters sauvegardée ✔")


# =========================
# 10. PROFIL DE CHAQUE SEGMENT — RADAR/BARPLOT
# =========================
profile_features = [
    "CA_potentielApres2ans", "investissement_global", "apport_personnel",
    "anciennete", "Implantations", "rating", "ratio_apport_investissement",
    "YOY_Sales"
]
profile_features = [c for c in profile_features if c in df.columns]

# Forcer la conversion numérique sur toutes les colonnes du df
# (certaines colonnes comme YOY_Sales, rating peuvent être des strings)
for col in profile_features:
    df[col] = clean_numeric(df[col])

profile = df.groupby("segment")[profile_features].median().T
profile.columns.name = None

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, feat in enumerate(profile_features):
    vals  = [profile.loc[feat, seg] for seg in COLORS.keys()
             if seg in profile.columns]
    segs  = [seg for seg in COLORS.keys() if seg in profile.columns]
    cols  = [COLORS[s] for s in segs]
    bars  = axes[i].bar(segs, vals, color=cols, alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        axes[i].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() * 1.02,
                     f"{v:,.0f}" if v > 100 else f"{v:.2f}",
                     ha="center", fontsize=8, color="gray")
    axes[i].set_title(feat.replace("_", " "), fontsize=10)
    axes[i].tick_params(axis="x", labelsize=8, rotation=15)
    axes[i].grid(True, alpha=0.2, axis="y")
    axes[i].set_ylabel("Médiane", fontsize=8)

plt.suptitle("Profil médian par segment", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("figures_segmentation/03_profil_segments.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure 03 — Profil segments sauvegardée ✔")


# =========================
# 11. DISTRIBUTION DU CA PAR SEGMENT
# =========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

order = list(COLORS.keys())
palette = [COLORS[s] for s in order]

sns.boxplot(data=df, x="segment", y=target_col, order=order,
            palette=palette, ax=axes[0], width=0.5)
axes[0].set_title("Distribution CA par segment")
axes[0].set_xlabel("")
axes[0].set_ylabel("CA potentiel à 2 ans (€)")
axes[0].yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
axes[0].grid(True, alpha=0.2, axis="y")

counts = df["segment"].value_counts().reindex(order)
bars = axes[1].bar(order, counts.values, color=palette, alpha=0.85, width=0.5)
for bar, v in zip(bars, counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 15,
                 f"{v}\n({v/len(df)*100:.1f}%)",
                 ha="center", fontsize=10, color="gray")
axes[1].set_title("Répartition des franchises par segment")
axes[1].set_xlabel("")
axes[1].set_ylabel("Nombre de franchises")
axes[1].grid(True, alpha=0.2, axis="y")

plt.tight_layout()
plt.savefig("figures_segmentation/04_distribution_ca_segments.png", dpi=150)
plt.close()
print("Figure 04 — Distribution CA par segment sauvegardée ✔")


# =========================
# 12. CLASSIFICATION SUPERVISÉE
# Objectif : prédire le segment d'une NOUVELLE franchise
# sans avoir besoin de relancer KMeans
# =========================
print("\n" + "="*60)
print("CLASSIFICATION SUPERVISÉE — apprentissage des labels KMeans")
print("="*60)

# Features pour la classification (sans la target CA pour éviter leakage)
CLF_FEATURES = [f for f in SEG_FEATURES if f != target_col]
print(f"Features classification ({len(CLF_FEATURES)}) :", CLF_FEATURES)

X_clf = df[CLF_FEATURES].copy()
y_clf = df["segment"].copy()

# Nettoyage numérique (même logique que pour le clustering)
for col in X_clf.columns:
    try:
        X_clf[col] = clean_numeric(X_clf[col])
    except Exception:
        X_clf[col] = pd.to_numeric(X_clf[col], errors="coerce")

non_num_clf = [c for c in X_clf.columns
               if not pd.api.types.is_numeric_dtype(X_clf[c])]
if non_num_clf:
    print(f"  Colonnes CLF non-numériques supprimées : {non_num_clf}")
    X_clf.drop(columns=non_num_clf, inplace=True)
    CLF_FEATURES = list(X_clf.columns)

le = LabelEncoder()
y_encoded = le.fit_transform(y_clf)

imp_clf   = SimpleImputer(strategy="median")
X_clf_imp = imp_clf.fit_transform(X_clf)

X_train, X_test, y_train, y_test = train_test_split(
    X_clf_imp, y_encoded, test_size=0.2, random_state=42,
    stratify=y_encoded)

print(f"\nTrain : {len(X_train)} | Test : {len(X_test)}")
print("Distribution des segments dans le train :")
for label, count in zip(*np.unique(y_train, return_counts=True)):
    print(f"  {le.classes_[label]:15s} : {count} ({count/len(y_train)*100:.1f}%)")


# =========================
# 13. ENTRAÎNEMENT CLASSIFIEURS
# =========================
classifiers = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=12,
        random_state=42, n_jobs=-1, class_weight="balanced"),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05,
        max_depth=5, random_state=42),
}

clf_results  = {}
best_clf     = None
best_clf_acc = -1
best_clf_name = ""

for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    cv_scores = cross_val_score(clf, X_clf_imp, y_encoded,
                                cv=5, scoring="accuracy", n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred    = clf.predict(X_test)
    test_acc  = (y_pred == y_test).mean()

    clf_results[name] = {
        "cv_mean": cv_scores.mean(),
        "cv_std":  cv_scores.std(),
        "test_acc": test_acc,
        "y_pred": y_pred,
    }
    print(f"  CV Accuracy : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Test Accuracy : {test_acc:.3f}")
    print(classification_report(y_test, y_pred,
                                target_names=le.classes_,
                                zero_division=0))

    if cv_scores.mean() > best_clf_acc:
        best_clf_acc  = cv_scores.mean()
        best_clf      = clf
        best_clf_name = name

print(f"\nMeilleur classifieur : {best_clf_name} "
      f"(CV Accuracy = {best_clf_acc:.3f})")


# =========================
# 14. MATRICE DE CONFUSION
# =========================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (name, res) in zip(axes, clf_results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matrice de confusion — {name}\nAccuracy test : {res['test_acc']:.3f}")
    ax.tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig("figures_segmentation/05_confusion_matrix.png", dpi=150)
plt.close()
print("Figure 05 — Matrices de confusion sauvegardées ✔")


# =========================
# 15. FEATURE IMPORTANCE DU CLASSIFIEUR
# =========================
if hasattr(best_clf, "feature_importances_"):
    importances = best_clf.feature_importances_
    min_len = min(len(CLF_FEATURES), len(importances))
    fi = pd.DataFrame({
        "feature":    CLF_FEATURES[:min_len],
        "importance": importances[:min_len]
    }).sort_values("importance", ascending=False)
    print(f"  Features: {len(CLF_FEATURES)} | Importances: {len(importances)} | Utilisées: {min_len}")

    fig, ax = plt.subplots(figsize=(10, 6))
    top_fi = fi.head(12)
    bar_colors = ["#C44E52" if i < 3 else "#4C72B0"
                  for i in range(len(top_fi))]
    ax.barh(top_fi["feature"][::-1],
            top_fi["importance"][::-1],
            color=bar_colors[::-1], alpha=0.85)
    ax.axvline(fi["importance"].mean(), color="gray",
               linestyle="--", alpha=0.6, label="Importance moyenne")
    ax.set_title(f"Feature importance — {best_clf_name} (classification segment)")
    ax.set_xlabel("Importance")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()
    plt.savefig("figures_segmentation/06_feature_importance_clf.png", dpi=150)
    plt.close()
    print("Figure 06 — Feature importance classification sauvegardée ✔")

    print("\nTop 8 features pour prédire le segment :")
    print(fi.head(8).to_string(index=False))


# =========================
# 16. SAUVEGARDE DES MODÈLES ET ARTEFACTS
# =========================
joblib.dump(kmeans,    "models/kmeans_segmentation.pkl")
joblib.dump(imputer,   "models/seg_imputer.pkl")
joblib.dump(scaler,    "models/seg_scaler.pkl")
joblib.dump(best_clf,  "models/clf_segment.pkl")
joblib.dump(imp_clf,   "models/clf_imputer.pkl")
joblib.dump(le,        "models/label_encoder.pkl")

# Sauvegarder le mapping cluster → segment pour référence
import json
mapping_info = {
    "label_map": {int(k): v for k, v in LABEL_MAP_RAW.items()},
    "clf_features": CLF_FEATURES,
    "seg_features": SEG_FEATURES,
    "silhouette_score": round(sil_final, 4),
    "best_classifier": best_clf_name,
    "cv_accuracy": round(best_clf_acc, 4),
}
with open("models/segmentation_config.json", "w") as f:
    json.dump(mapping_info, f, indent=2)

print("\nModèles sauvegardés ✔")
print("  models/kmeans_segmentation.pkl")
print("  models/clf_segment.pkl")
print("  models/label_encoder.pkl")
print("  models/segmentation_config.json")


# =========================
# 17. FONCTION DE PRÉDICTION SEGMENT
# Usage depuis l'application : donner un dict de features
# → retourne "Haut Rendement" / "Valeur Sûre" / "Risque Élevé"
# =========================
def predict_segment(input_dict: dict) -> dict:
    """
    Prédit le segment d'une franchise à partir de ses caractéristiques.

    Paramètres
    ----------
    input_dict : dict
        Données brutes de la franchise (même format que data.xlsx).
        Les features manquantes sont imputées par la médiane du dataset.

    Retourne
    --------
    dict avec :
        - segment       : label prédit ("Haut Rendement", etc.)
        - probabilities : dict {segment: probabilité}
        - confidence    : probabilité de la classe prédite
    """
    df_in = pd.DataFrame([input_dict])
    df_in = feature_engineering(df_in)

    X_in = df_in[CLF_FEATURES].reindex(columns=CLF_FEATURES)
    X_in_imp = imp_clf.transform(X_in)

    pred_encoded = best_clf.predict(X_in_imp)[0]
    probas       = best_clf.predict_proba(X_in_imp)[0]

    segment   = le.inverse_transform([pred_encoded])[0]
    proba_map = {le.classes_[i]: round(float(p), 3)
                 for i, p in enumerate(probas)}

    return {
        "segment":       segment,
        "probabilities": proba_map,
        "confidence":    round(float(probas.max()), 3),
    }


# =========================
# 18. TEST DE LA FONCTION
# =========================
print("\n" + "="*60)
print("TEST — predict_segment()")
print("="*60)

examples = [
    {
        "nom": "franchise_A_test",
        "apport_personnel": 80000,
        "investissement_global": 250000,
        "Implantations": 400,
        "anciennete": 30,
        "rating": 4.5,
        "Redevance_fonctionnement": 5,
        "YOY_Sales": 8,
        "Surface_moyenne": "150 m2",
        "Formations": "Oui",
        "Aide_auFinancement": "Oui",
        "date_creation": 1996,
        "adresses": "Paris, France",
        "heure_ouverture": "08:00",
        "heure_fermeture": "22:00",
    },
    {
        "nom": "franchise_B_test",
        "apport_personnel": 20000,
        "investissement_global": 300000,
        "Implantations": 15,
        "anciennete": 3,
        "rating": 3.1,
        "Redevance_fonctionnement": 12,
        "YOY_Sales": -2,
        "Surface_moyenne": "60 m2",
        "Formations": "Non",
        "Aide_auFinancement": "Non",
        "date_creation": 2023,
        "adresses": "Lyon, France",
        "heure_ouverture": "11:00",
        "heure_fermeture": "15:00",
    },
]

for ex in examples:
    result = predict_segment(ex)
    print(f"\n  Franchise : {ex['nom']}")
    print(f"  Segment prédit  : {result['segment']}")
    print(f"  Confiance       : {result['confidence']*100:.1f}%")
    print(f"  Probabilités    : {result['probabilities']}")


# =========================
# 19. RÉSUMÉ FINAL
# =========================
print("\n" + "="*60)
print("SEGMENTATION COMPLÈTE ✔")
print("="*60)
print(f"\nKMeans K=3 — Silhouette : {sil_final:.3f}")
print(f"Classifieur : {best_clf_name} — CV Accuracy : {best_clf_acc:.3f}")
print("\nRépartition finale des segments :")
for seg, count in df["segment"].value_counts().items():
    pct = count / len(df) * 100
    print(f"  {seg:16s} : {count:5d} franchises ({pct:.1f}%)")

print("\nFigures générées dans figures_segmentation/ :")
for f in sorted(os.listdir("figures_segmentation")):
    print(f"  {f}")