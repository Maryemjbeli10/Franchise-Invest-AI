"""
MLFLOW — RECOMMANDATION (recommandation.py)
============================================
Stockage local SQLite — pas besoin de serveur séparé.

IMPORTANT : ce fichier doit être dans le même dossier que recommandation.py
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn

from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIGURATION MLFLOW
# =========================
MLFLOW_DB       = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Franchise_Recommandation"

mlflow.set_tracking_uri(MLFLOW_DB)
mlflow.set_experiment(EXPERIMENT_NAME)

os.makedirs("figures_recommandation", exist_ok=True)

# =========================
# IMPORTS DEPUIS recommandation.py
# =========================
from recommandation import (
    load_and_clean,
    build_feature_matrix,
    InteractionStore,
    FranchiseRecommender,
    NUMERIC_FEATURES,
    CAT_FEATURES,
    INTERACTION_WEIGHTS
)

# =========================
# CHARGEMENT DES DONNÉES
# =========================
print("Chargement des données...")
df = load_and_clean("data.xlsx")
print(f"  {len(df)} franchises | {(df['has_rating']==0).sum()} sans rating")

# =========================
# RUN MLFLOW — BUILD DU MODÈLE
# =========================
print("\n" + "="*55)
print("MLflow — Build Recommender")

with mlflow.start_run(run_name="Build_Recommender") as build_run:

    mlflow.set_tags({"project":"Franchise_Invest_AI","task":"recommendation",
                     "model_type":"Hybrid_ContentBased_SVD","version":"v3"})

    mlflow.log_params({
        "n_franchises":        len(df),
        "n_numeric_features":  len(NUMERIC_FEATURES),
        "n_cat_features":      len(CAT_FEATURES),
        "nn_neighbors":        20,
        "nn_metric":           "cosine",
        "alpha_default":       0.65,
        "roi_min_default":     1.2,
        "no_rating_penalty":   0.90,
    })

    mlflow.log_metrics({
        "n_franchises":        len(df),
        "n_with_rating":       int(df["has_rating"].sum()),
        "n_without_rating":    int((df["has_rating"]==0).sum()),
        "rating_coverage_pct": round(float(df["has_rating"].mean())*100, 2),
        "mean_composite_score":round(float(df["composite_score"].mean()), 4),
        "n_cuisines_uniques":  int(df["cuisines"].nunique()),
    })

    # Interactions simulées
    store = InteractionStore("interactions.json")
    store.log("investor_001","TUTTI PIZZA","view")
    store.log("investor_001","TUTTI PIZZA","favorite")
    store.log("investor_001","SUBWAY","view")
    store.log("investor_001","WENDY'S","view")
    store.log("investor_001","WENDY'S","contact")
    store.log("investor_002","TUTTI PIZZA","dossier")
    store.log("investor_002","WENDY'S","favorite")
    store.log("investor_002","SUBWAY","contact")
    store.log("investor_003","TUTTI PIZZA","contact")
    store.log("investor_003","SUBWAY","favorite")
    store.log("investor_003","WENDY'S","view")

    mlflow.log_metric("total_interactions",    store.total_interactions())
    mlflow.log_metric("n_users_with_history",  len(store.get_all_users()))

    # Fit
    t0    = time.time()
    model = FranchiseRecommender(interaction_store=store)
    model.fit(df)
    fit_time = time.time() - t0
    mlflow.log_metric("fit_time_seconds", round(fit_time, 3))
    print(f"  Modèle entraîné en {fit_time:.2f}s")

    # Similarité cosinus (sample 200)
    fm          = model.feature_matrix
    sample_idx  = np.random.choice(len(fm), min(200, len(fm)), replace=False)
    sim_matrix  = cosine_similarity(fm[sample_idx])
    np.fill_diagonal(sim_matrix, 0)

    mlflow.log_metrics({
        "mean_cosine_similarity":   round(float(sim_matrix.mean()), 4),
        "max_cosine_similarity":    round(float(sim_matrix.max()),  4),
        "median_cosine_similarity": round(float(np.median(sim_matrix)), 4),
    })

    # Heatmap similarité
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(sim_matrix[:30,:30], ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
                xticklabels=False, yticklabels=False)
    ax.set_title(f"Matrice Cosinus (30×30 sample)  Moyenne={sim_matrix.mean():.3f}")
    plt.tight_layout()
    sim_path = "figures_recommandation/mlflow_01_similarity_heatmap.png"
    plt.savefig(sim_path, dpi=120); plt.close()
    mlflow.log_artifact(sim_path)

    # Distribution composite score
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(df["composite_score"], bins=30, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax2.set_xlabel("Composite Score"); ax2.set_ylabel("Nb franchises")
    ax2.set_title("Distribution du Composite Score"); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    sc_path = "figures_recommandation/mlflow_02_composite_score_dist.png"
    plt.savefig(sc_path, dpi=120); plt.close()
    mlflow.log_artifact(sc_path)

    model.save("franchise_recommender.pkl")
    mlflow.log_artifact("franchise_recommender.pkl")
    build_run_id = build_run.info.run_id

# =========================
# RUNS MLFLOW — ÉVALUATION SCÉNARIOS
# =========================
print("\n" + "="*55)
print("MLflow — Evaluation Scénarios")

TEST_SCENARIOS = [
    {"name":"Budget_Moyen_Italienne",
     "budget":80000, "top_n":10, "user_id":None,
     "prefs":{"cuisine":"italienne","performance_min":1,"roi_min":1.2}},
    {"name":"Budget_Eleve_Delivery",
     "budget":150000, "top_n":10, "user_id":None,
     "prefs":{"delivery_required":True,"prefer_high_rating":True,"roi_min":1.2}},
    {"name":"Budget_Faible_Formation",
     "budget":30000, "top_n":10, "user_id":None,
     "prefs":{"formations_required":True,"aide_financement_required":True,"roi_min":1.0}},
    {"name":"User_Connecte_CF_Reel",
     "budget":80000, "top_n":10, "user_id":"investor_001",
     "prefs":{"cuisine":"italienne","performance_min":1,"roi_min":1.2}},
]

all_metrics = {}

for sc in TEST_SCENARIOS:
    sname = sc["name"]
    print(f"\n  Scénario : {sname}")

    with mlflow.start_run(run_name=f"Eval_{sname}"):

        mlflow.set_tags({"project":"Franchise_Invest_AI","task":"recommendation_eval",
                         "scenario":sname,"linked_build_run":build_run_id,
                         "user_mode":"connected" if sc["user_id"] else "anonymous"})

        mlflow.log_params({
            "budget":   sc["budget"], "top_n": sc["top_n"],
            "roi_min":  sc["prefs"].get("roi_min",1.2),
            "cuisine":  sc["prefs"].get("cuisine","all"),
            "user_id":  sc["user_id"] or "anonymous",
        })

        t0   = time.time()
        recs = model.recommend(budget=sc["budget"], preferences=sc["prefs"],
                               top_n=sc["top_n"], user_id=sc["user_id"])
        latency = time.time() - t0

        if recs.empty:
            mlflow.log_metric("n_results", 0)
            continue

        roi_values = recs["CA_potentielApres2ans"] / recs["investissement_global"]
        roi_min    = sc["prefs"].get("roi_min", 1.2)
        metrics    = {
            "n_results":                 int(len(recs)),
            "latency_seconds":           round(latency, 4),
            "mean_recommendation_score": round(float(recs["recommendation_score"].mean()), 4),
            "mean_roi":                  round(float(roi_values.mean()), 4),
            "min_roi":                   round(float(roi_values.min()),  4),
            "pct_roi_ok":                round(float((roi_values >= roi_min).sum() / len(recs) * 100), 2),
            "mean_ca_potentiel":         round(float(recs["CA_potentielApres2ans"].mean()), 0),
            "coverage_pct":              round(float(len(recs) / len(df) * 100), 2),
            "n_with_rating":             int(recs["has_rating"].sum()),
        }
        if "rating_composite" in recs.columns:
            metrics["mean_rating"] = round(float(recs["rating_composite"].mean()), 4)

        mlflow.log_metrics(metrics)
        all_metrics[sname] = metrics

        # Figure scores / ROI / CA
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].barh(recs["nom"].str[:25][::-1], recs["recommendation_score"][::-1],
                     color="#4C72B0", alpha=0.8)
        axes[0].set_xlabel("Score"); axes[0].set_title("Score de recommandation")
        axes[0].grid(True, alpha=0.3, axis="x")

        bar_colors = ["#639922" if r>=roi_min else "#E24B4A" for r in roi_values.values[::-1]]
        axes[1].barh(recs["nom"].str[:25][::-1], roi_values.values[::-1], color=bar_colors, alpha=0.8)
        axes[1].axvline(roi_min, color="red", linestyle="--", alpha=0.7, label=f"ROI min {roi_min}x")
        axes[1].set_xlabel("ROI"); axes[1].set_title("ROI (CA / Invest)"); axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="x")

        axes[2].barh(recs["nom"].str[:25][::-1],
                     (recs["CA_potentielApres2ans"]/1000).values[::-1],
                     color="#55A868", alpha=0.8)
        axes[2].set_xlabel("CA potentiel (k€)"); axes[2].set_title("CA potentiel à 2 ans")
        axes[2].grid(True, alpha=0.3, axis="x")

        plt.suptitle(f"{sname} | Budget={sc['budget']:,}€ | n={len(recs)}", fontsize=12)
        plt.tight_layout()
        rp = f"figures_recommandation/mlflow_{sname}_recs.png"
        plt.savefig(rp, dpi=120); plt.close()
        mlflow.log_artifact(rp)

        # CSV des recs
        cp = f"figures_recommandation/mlflow_{sname}_recs.csv"
        recs.to_csv(cp, index=False)
        mlflow.log_artifact(cp)

        print(f"    n={len(recs)} | ROI moy={metrics['mean_roi']:.2f}x | "
              f"Score moy={metrics['mean_recommendation_score']:.4f} | {latency:.3f}s")

# =========================
# RUN MLFLOW — COMPARAISON SCÉNARIOS
# =========================
if all_metrics:
    with mlflow.start_run(run_name="Scenarios_Comparison"):
        mlflow.set_tags({"project":"Franchise_Invest_AI","task":"summary",
                         "linked_build_run":build_run_id})

        for sname, metrics in all_metrics.items():
            for k, v in metrics.items():
                mlflow.log_metric(f"{sname}_{k}", v)

        scenario_names = list(all_metrics.keys())
        colors = ["#4C72B0","#55A868","#DD8452","#C44E52"]
        short  = [s.replace("_","\n") for s in scenario_names]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0][0].bar(short,[all_metrics[s]["mean_roi"] for s in scenario_names],color=colors,alpha=0.8)
        axes[0][0].axhline(1.2,color="red",linestyle="--",alpha=0.7,label="Seuil 1.2x")
        axes[0][0].set_title("ROI moyen par scénario"); axes[0][0].legend()

        axes[0][1].bar(short,[all_metrics[s]["mean_recommendation_score"] for s in scenario_names],
                       color=colors,alpha=0.8)
        axes[0][1].set_title("Score moyen de recommandation")

        axes[1][0].bar(short,[all_metrics[s]["pct_roi_ok"] for s in scenario_names],color=colors,alpha=0.8)
        axes[1][0].set_title("% franchises ROI ≥ seuil")

        axes[1][1].bar(short,[all_metrics[s]["latency_seconds"] for s in scenario_names],color=colors,alpha=0.8)
        axes[1][1].set_title("Latence (secondes)")

        for ax_row in axes:
            for ax in ax_row:
                ax.tick_params(axis="x",labelsize=8); ax.grid(True,alpha=0.3,axis="y")

        plt.suptitle("Comparaison des scénarios", fontsize=13, fontweight="bold")
        plt.tight_layout()
        comp_path = "figures_recommandation/mlflow_scenarios_comparison.png"
        plt.savefig(comp_path, dpi=150); plt.close()
        mlflow.log_artifact(comp_path)

print("\nRecommandation DONE ✔  — mlflow.db mis à jour")
