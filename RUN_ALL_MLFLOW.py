"""
RUN_ALL_MLFLOW.py
==================
Lance les 3 expériences MLflow dans l'ordre correct.
Stockage local SQLite — pas besoin de serveur séparé avant de lancer.

Usage :
    1. python run_all_mlflow.py          ← lance les 3 expériences
    2. mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
    3. Ouvrir http://127.0.0.1:5001 dans le navigateur
"""

import subprocess
import sys
import os


def run_script(script_name: str):
    print(f"\n{'='*60}")
    print(f"  LANCEMENT : {script_name}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script_name], text=True)
    if result.returncode != 0:
        print(f"  [ERREUR] {script_name} a échoué (code {result.returncode})")
    else:
        print(f"  [OK] {script_name} terminé avec succès")
    return result.returncode


if __name__ == "__main__":
    print("=" * 60)
    print("  FRANCHISE INVEST AI — MLflow Pipeline Complet")
    print("=" * 60)
    print("\nStockage local SQLite — aucun serveur requis avant le lancement.")
    print("Les résultats seront dans mlflow.db\n")

    scripts = [
        "mlflow_regression.py",
        "mlflow_segmentation.py",
        "mlflow_recommandation.py",
    ]

    errors = []
    for script in scripts:
        if not os.path.exists(script):
            print(f"  [SKIP] {script} introuvable dans le dossier courant")
            errors.append(script)
            continue
        code = run_script(script)
        if code != 0:
            errors.append(script)

    print("\n" + "=" * 60)
    if errors:
        print(f"  [ATTENTION] Scripts avec erreurs : {errors}")
    else:
        print("  TOUS LES RUNS MLFLOW SONT TERMINÉS ✔")
    print("=" * 60)
    print("\nPour voir le dashboard MLflow, lance dans le terminal :")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001")
    print("  Puis ouvrir : http://127.0.0.1:5001")
