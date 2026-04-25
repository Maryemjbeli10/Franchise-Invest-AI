"""
FRANCHISE INVEST AI - VERSION AMELIOREE
EDA enrichi + Traitement Outliers + Cross-Validation + 4 Modèles (+ XGBoost) + Feature Importance + Meilleur Modèle + Prédiction
"""

# =========================
# 1. IMPORTS
# =========================
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor


# =========================
# 2. CREATE FOLDERS
# =========================
os.makedirs("figures", exist_ok=True)
os.makedirs("models", exist_ok=True)


# =========================
# 3. LOAD DATA
# =========================
df = pd.read_excel("data.xlsx")
print("Dataset shape:", df.shape)


# =========================
# 4. FEATURE ENGINEERING
# =========================
def feature_engineering(df):

    df = df.copy()

    df["Formations"] = df["Formations"].map({"oui":1,"non":0,"Oui":1,"Non":0}).fillna(0)
    df["Aide_auFinancement"] = df["Aide_auFinancement"].map({"oui":1,"non":0,"Oui":1,"Non":0}).fillna(0)

    df["surface_m2"] = df["Surface_moyenne"].str.extract(r"(\d+)").astype(float)

    df["anciennete"] = 2026 - df["date_creation"]

    df["est_france"] = df["adresses"].str.contains("France", case=False, na=False).astype(int)

    df["ratio_apport_investissement"] = df["apport_personnel"] / df["investissement_global"]

    df["investissement_par_m2"] = df["investissement_global"] / df["surface_m2"]

    # NOUVEAU: Durée d'ouverture journalière
    def parse_hours(h):
        try:
            h = str(h).strip()
            hh, mm = map(int, h.split(":"))
            return hh + mm / 60
        except:
            return np.nan

    if "heure_ouverture" in df.columns and "heure_fermeture" in df.columns:
        df["heure_ouv_num"] = df["heure_ouverture"].apply(parse_hours)
        df["heure_ferm_num"] = df["heure_fermeture"].apply(parse_hours)
        df["nb_heures_ouverture"] = df["heure_ferm_num"] - df["heure_ouv_num"]
        df["nb_heures_ouverture"] = df["nb_heures_ouverture"].apply(lambda x: x if x > 0 else np.nan)
        df.drop(columns=["heure_ouverture", "heure_fermeture", "heure_ouv_num", "heure_ferm_num"], errors="ignore", inplace=True)

    return df


df = feature_engineering(df)


# =========================
# 5. TRAITEMENT DES OUTLIERS (NOUVEAU)
# =========================
target = "CA_potentielApres2ans"

def remove_outliers_iqr(df, col, factor=3.0):
    """Supprime les outliers extrêmes via IQR avec facteur large pour ne pas trop filtrer."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    before = len(df)
    df_clean = df[(df[col] >= lower) & (df[col] <= upper)]
    after = len(df_clean)
    print(f"Outliers supprimés sur '{col}': {before - after} lignes ({before} → {after})")
    return df_clean

df = remove_outliers_iqr(df, target, factor=3.0)

# Log-transformation de la target (réduit l'asymétrie)
df[f"{target}_log"] = np.log1p(df[target])
print(f"Log-transform appliqué sur la target. Asymétrie avant: {df[target].skew():.2f} | après: {df[f'{target}_log'].skew():.2f}")


# =========================
# 6. EDA ENRICHI
# =========================
sns.set_style("whitegrid")
sns.set_palette("husl")

# --- Graphiques ORIGINAUX ---

# 1. Distribution target (avant + après log)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df[target], bins=30, kde=True, ax=axes[0], color="#4C72B0")
axes[0].set_title("Distribution CA potentiel (original)")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
sns.histplot(df[f"{target}_log"], bins=30, kde=True, ax=axes[1], color="#55A868")
axes[1].set_title("Distribution CA potentiel (log-transformé)")
plt.tight_layout()
plt.savefig("figures/01_target_distribution.png", dpi=150)
plt.close()

# 2. Matrice de corrélation
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0, linewidths=0.3)
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.savefig("figures/02_correlation_matrix.png", dpi=150)
plt.close()

# 3. Valeurs manquantes
plt.figure(figsize=(10, 5))
missing = df.isnull().sum().sort_values(ascending=False)
missing[missing > 0].plot(kind="bar", color="#DD8452") if missing[missing > 0].any() else missing.head(15).plot(kind="bar", color="#4C72B0")
plt.title("Valeurs manquantes par colonne")
plt.tight_layout()
plt.savefig("figures/03_missing_values.png", dpi=150)
plt.close()

# 4. Boxplot outliers (avant nettoyage visible via log)
plt.figure(figsize=(10, 5))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].boxplot(df[target].dropna(), vert=False, patch_artist=True, boxprops=dict(facecolor="#4C72B0", alpha=0.6))
axes[0].set_title("Outliers CA (après nettoyage IQR x3)")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k€"))
axes[1].boxplot(df[f"{target}_log"].dropna(), vert=False, patch_artist=True, boxprops=dict(facecolor="#55A868", alpha=0.6))
axes[1].set_title("Outliers CA (log-transformé)")
plt.tight_layout()
plt.savefig("figures/04_outliers.png", dpi=150)
plt.close()

# --- NOUVEAUX GRAPHIQUES EDA ---

# 5. Scatter plot CA vs investissement_global
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df["investissement_global"],
    df[target],
    alpha=0.4,
    c=df["anciennete"],
    cmap="RdYlGn",
    s=20,
    edgecolors="none"
)
plt.colorbar(scatter, label="Ancienneté (années)")
plt.xlabel("Investissement global (€)")
plt.ylabel("CA potentiel à 2 ans (€)")
plt.title("CA potentiel vs Investissement global\n(couleur = ancienneté de la franchise)")
plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))

# Droite de tendance
mask = df[["investissement_global", target]].dropna()
z = np.polyfit(mask["investissement_global"], mask[target], 1)
p = np.poly1d(z)
x_line = np.linspace(mask["investissement_global"].min(), mask["investissement_global"].max(), 100)
plt.plot(x_line, p(x_line), "r--", linewidth=1.5, alpha=0.7, label="Tendance")
plt.legend()
plt.tight_layout()
plt.savefig("figures/05_scatter_CA_vs_investissement.png", dpi=150)
plt.close()

# 6. Analyse CA par secteur / Merchant_Category
if "Merchant_Category" in df.columns:
    plt.figure(figsize=(12, 6))
    cat_stats = df.groupby("Merchant_Category")[target].agg(["median", "mean", "count"]).reset_index()
    cat_stats = cat_stats[cat_stats["count"] >= 5].sort_values("median", ascending=False).head(15)

    bars = plt.bar(cat_stats["Merchant_Category"], cat_stats["median"] / 1000, color="#4C72B0", alpha=0.8, label="Médiane")
    plt.bar(cat_stats["Merchant_Category"], cat_stats["mean"] / 1000, color="#DD8452", alpha=0.5, label="Moyenne", width=0.4)

    for bar, count in zip(bars, cat_stats["count"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"n={count}", ha="center", fontsize=8, color="gray")

    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Catégorie (Merchant Category)")
    plt.ylabel("CA potentiel médian (k€)")
    plt.title("CA potentiel médian par catégorie de franchise")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/06_CA_par_secteur.png", dpi=150)
    plt.close()
    print("EDA 06 — analyse par secteur ✔")
else:
    print("EDA 06 — colonne 'Merchant_Category' absente, graphique ignoré")

# 7. Analyse CA par type de cuisine
if "cuisines" in df.columns:
    plt.figure(figsize=(12, 6))
    cuisine_stats = df.groupby("cuisines")[target].agg(["median", "count"]).reset_index()
    cuisine_stats = cuisine_stats[cuisine_stats["count"] >= 5].sort_values("median", ascending=False).head(15)

    bars = plt.bar(cuisine_stats["cuisines"], cuisine_stats["median"] / 1000, color="#55A868", alpha=0.85)
    for bar, count in zip(bars, cuisine_stats["count"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"n={count}", ha="center", fontsize=8, color="gray")

    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Type de cuisine")
    plt.ylabel("CA potentiel médian (k€)")
    plt.title("CA potentiel médian par type de cuisine")
    plt.tight_layout()
    plt.savefig("figures/07_CA_par_cuisine.png", dpi=150)
    plt.close()
    print("EDA 07 — analyse par cuisine ✔")
else:
    print("EDA 07 — colonne 'cuisines' absente, graphique ignoré")

# 8. Scatter CA vs apport_personnel (ratio)
plt.figure(figsize=(10, 6))
plt.scatter(
    df["ratio_apport_investissement"],
    df[target],
    alpha=0.4,
    s=18,
    color="#C44E52",
    edgecolors="none"
)
plt.xlabel("Ratio apport / investissement global")
plt.ylabel("CA potentiel à 2 ans (€)")
plt.title("CA potentiel vs Ratio apport personnel / investissement")
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
plt.tight_layout()
plt.savefig("figures/08_scatter_CA_vs_ratio_apport.png", dpi=150)
plt.close()

print("EDA ENRICHI SAVED ✔")


# =========================
# 7. SPLIT
# =========================
target_log = f"{target}_log"

X = df.drop(columns=[target, target_log], errors="ignore")
y = df[target_log]          # on entraîne sur la version log

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()


# =========================
# 8. PREPROCESSING
# =========================
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])


# =========================
# 9. MODÈLES — 4 modèles dont XGBoost (NOUVEAU)
# =========================
models = {
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(               # NOUVEAU
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
}


# =========================
# 10. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 11. TRAIN LOOP + CROSS-VALIDATION (NOUVEAU)
# =========================
results = {}
cv_results = {}
best_pipeline = None
best_score = -999
best_name = ""

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():

    print(f"\nTraining {name} ...")

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    # --- Cross-validation (NOUVEAU) ---
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
    cv_results[name] = cv_scores
    print(f"  CV R² : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # --- Entraînement final sur tout le train set ---
    pipeline.fit(X_train, y_train)
    preds_log = pipeline.predict(X_test)

    # Retransformer en échelle originale pour les métriques
    preds = np.expm1(preds_log)
    y_test_orig = np.expm1(y_test)

    r2 = r2_score(y_test_orig, preds)
    rmse = np.sqrt(mean_squared_error(y_test_orig, preds))
    mae = mean_absolute_error(y_test_orig, preds)

    results[name] = {"r2": r2, "rmse": rmse, "mae": mae, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()}
    print(f"  Test  → R²:{r2:.3f} | RMSE:{rmse:,.0f} | MAE:{mae:,.0f}")

    if cv_scores.mean() > best_score:
        best_score = cv_scores.mean()
        best_pipeline = pipeline
        best_name = name


# =========================
# 12. RÉSUMÉ DES RÉSULTATS
# =========================
print("\n" + "="*60)
print("RESULTS SUMMARY (classé par CV R² moyen):")
print("="*60)
sorted_results = sorted(results.items(), key=lambda x: x[1]["cv_mean"], reverse=True)
for k, v in sorted_results:
    flag = " ← BEST" if k == best_name else ""
    print(f"{k:20s} | CV R²: {v['cv_mean']:.3f} ± {v['cv_std']:.3f} | Test R²: {v['r2']:.3f} | RMSE: {v['rmse']:,.0f} | MAE: {v['mae']:,.0f}{flag}")

print(f"\nBEST MODEL (selon CV): {best_name}")


# =========================
# 13. GRAPHIQUE COMPARAISON CV (NOUVEAU)
# =========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot des scores CV
cv_data = [cv_results[name] for name in models.keys()]
bp = axes[0].boxplot(cv_data, labels=list(models.keys()), patch_artist=True, notch=True)
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_title("Cross-Validation R² (5 folds) — comparaison des modèles")
axes[0].set_ylabel("R² score")
axes[0].tick_params(axis='x', rotation=20)
axes[0].grid(True, alpha=0.3)

# Barplot RMSE sur test set
names = list(results.keys())
rmses = [results[n]["rmse"] for n in names]
bars = axes[1].bar(names, [r/1000 for r in rmses], color=colors, alpha=0.8)
for bar, val in zip(bars, rmses):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val/1000:.1f}k", ha="center", fontsize=10)
best_idx = names.index(best_name)
bars[best_idx].set_edgecolor("gold")
bars[best_idx].set_linewidth(3)
axes[1].set_title("RMSE sur test set (k€) — plus bas = meilleur")
axes[1].set_ylabel("RMSE (k€)")
axes[1].tick_params(axis='x', rotation=20)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("figures/09_model_comparison_cv.png", dpi=150)
plt.close()
print("Graphique comparaison modèles ✔")


# =========================
# 14. FEATURE IMPORTANCE (NOUVEAU)
# =========================
try:
    # Récupérer le nom de toutes les features après preprocessing
    ohe_feature_names = best_pipeline.named_steps["prep"].named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(cat_cols)
    all_feature_names = np.concatenate([num_cols, ohe_feature_names])

    # Récupérer les importances selon le type de modèle
    final_model = best_pipeline.named_steps["model"]
    if hasattr(final_model, "feature_importances_"):
        importances = final_model.feature_importances_
    else:
        importances = None

    if importances is not None:
        fi_df = pd.DataFrame({
            "feature": all_feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False).head(20)

        plt.figure(figsize=(10, 8))
        colors_fi = ["#C44E52" if i < 3 else "#4C72B0" for i in range(len(fi_df))]
        bars = plt.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color=colors_fi[::-1], alpha=0.85)
        plt.xlabel("Importance (gain moyen)")
        plt.title(f"Top 20 features importantes — {best_name}")
        plt.axvline(fi_df["importance"].mean(), color="gray", linestyle="--", alpha=0.7, label="Moyenne")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/10_feature_importance.png", dpi=150)
        plt.close()
        print(f"Feature importance sauvegardée ✔")
        print("\nTop 10 features:")
        print(fi_df.head(10).to_string(index=False))
    else:
        print("Feature importance non disponible pour ce modèle.")

except Exception as e:
    print(f"Feature importance skippée: {e}")


# =========================
# 15. SAVE MODEL
# =========================
joblib.dump(best_pipeline, "models/best_model.pkl")
joblib.dump(results, "models/results_summary.pkl")
print("\nMODÈLE SAUVEGARDÉ ✔")


# =========================
# 16. PREDICTION FUNCTION (mise à jour avec log)
# =========================
def predict(model, input_dict):
    """
    Prédit le CA potentiel à 2 ans.
    Retourne le CA estimé, un intervalle de confiance basé sur les CV scores,
    et le taux de rentabilité.
    """
    df_input = pd.DataFrame([input_dict])
    df_input = feature_engineering(df_input)

    pred_log = model.predict(df_input)[0]
    pred = np.expm1(pred_log)

    # Intervalle basé sur la variabilité du CV (±1 std du modèle)
    cv_std = results[best_name]["cv_std"]
    margin = cv_std * 0.5
    lower = np.expm1(pred_log * (1 - margin))
    upper = np.expm1(pred_log * (1 + margin))

    return {
        "CA_potentiel_2ans": round(pred, 2),
        "intervalle_bas": round(lower, 2),
        "intervalle_haut": round(upper, 2),
        "rentabilite_%": round((pred / input_dict["investissement_global"]) * 100, 2),
        "best_model": best_name,
        "cv_r2": round(results[best_name]["cv_mean"], 3)
    }


# =========================
# 17. TEST
# =========================
example = {
    "nom": "test",
    "apport_personnel": 50000,
    "droites_entree": 20000,
    "investissement_global": 200000,
    "Implantations": 10,
    "Formations": 1,
    "Aide_auFinancement": 1,
    "Redevance_fonctionnement": 5,
    "Redevance_publicitaire": 2,
    "Surface_moyenne": "100 m2",
    "Type_de_contrat": "franchise",
    "pickup_enabled": 1,
    "delivery_enabled": 1,
    "avis": 100,
    "date_creation": 2010,
    "sales_in_millions_2019": 10,
    "company_owned_units_2019": 50,
    "total_units_2019": 100,
    "unit_change_from_2018": 5,
    "CompanyStores": 20,
    "TotalUnits2021": 120,
    "TotalChange_in_Units_from2020": 10,
    "Merchant_Category": "food",
    "franchise_fee": 30000,
    "YOY_Sales": 5,
    "YOY_Units": 3,
    "rating": 4,
    "food_rating": 4,
    "service_rating": 4,
    "adresses": "France",
    "cuisines": "fast food",
    "performance_CA": "moyenne",
    "duree_fonctionnement": 10,
    "heure_ouverture": "09:00",
    "heure_fermeture": "22:00"
}

print("\nPREDICTION TEST:")
result = predict(best_pipeline, example)
for k, v in result.items():
    print(f"  {k}: {v}")

print("\n" + "="*60)
print("PROJECT COMPLETE ✔")
print("="*60)
print("\nFigures générées:")
for f in sorted(os.listdir("figures")):
    print(f"  figures/{f}")