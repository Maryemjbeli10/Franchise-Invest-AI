"""
ml_utils.py
Pont entre Django et vos modèles ML (ai.py, segmentation.py, recommandation.py)
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from django.conf import settings


# Chemin vers les modèles sauvegardés
# Correction : utilise BASE_DIR directement, pas le dossier parent
BASE_ML_DIR = settings.BASE_DIR

class RegressionModelWrapper:
    """
    Wrapper autour du pipeline sklearn sauvegardé par ai.py
    Gère la feature engineering et la prédiction
    """
    def __init__(self, pipeline_path):
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Modèle regression non trouvé: {pipeline_path}")
        self.pipeline = joblib.load(pipeline_path)
        self.target_col = "CA_potentielApres2ans"
        
    def predict(self, input_dict):
        """
        input_dict: dictionnaire avec les features brutes (comme dans ai.py)
        Retourne: dict avec CA_potentiel_2ans, intervalle, rentabilite, etc.
        """
        # Feature engineering (copié de ai.py)
        df_input = pd.DataFrame([input_dict])
        df_input = self._feature_engineering(df_input)
        
        # Préparer X (mêmes colonnes que lors de l'entraînement)
        # Le pipeline contient déjà le preprocessor
        X = df_input.drop(columns=[self.target_col, f"{self.target_col}_log"], errors="ignore")
        
        # Prédiction en log-space
        pred_log = self.pipeline.predict(X)[0]
        
        # Retransformer
        pred = np.expm1(pred_log)
        
        # Intervalle de confiance approximatif (±10%)
        margin = 0.1
        lower = np.expm1(pred_log * (1 - margin))
        upper = np.expm1(pred_log * (1 + margin))
        
        # Rentabilité
        investissement = input_dict.get('investissement_global', 1)
        rentabilite = (pred / investissement) * 100 if investissement > 0 else 0
        
        return {
            "CA_potentiel_2ans": round(float(pred), 2),
            "intervalle_bas": round(float(lower), 2),
            "intervalle_haut": round(float(upper), 2),
            "rentabilite_%": round(float(rentabilite), 2),
            "input_used": input_dict
        }
    
    def _feature_engineering(self, df):
        """Copie de la fonction de ai.py"""
        df = df.copy()
        df["Formations"] = df["Formations"].map({"oui":1,"non":0,"Oui":1,"Non":0}).fillna(0)
        df["Aide_auFinancement"] = df["Aide_auFinancement"].map({"oui":1,"non":0,"Oui":1,"Non":0}).fillna(0)
        df["surface_m2"] = df["Surface_moyenne"].astype(str).str.extract(r"(\d+)").astype(float)
        df["anciennete"] = 2026 - df["date_creation"]
        df["est_france"] = df["adresses"].astype(str).str.contains("France", case=False, na=False).astype(int)
        df["ratio_apport_investissement"] = df["apport_personnel"] / df["investissement_global"].replace(0, np.nan)
        df["investissement_par_m2"] = df["investissement_global"] / df["surface_m2"].replace(0, np.nan)
        
        def parse_hours(h):
            try:
                hh, mm = map(int, str(h).strip().split(":"))
                return hh + mm / 60
            except:
                return np.nan
        
        if "heure_ouverture" in df.columns and "heure_fermeture" in df.columns:
            df["heure_ouv_num"] = df["heure_ouverture"].apply(parse_hours)
            df["heure_ferm_num"] = df["heure_fermeture"].apply(parse_hours)
            df["nb_heures_ouverture"] = df["heure_ferm_num"] - df["heure_ouv_num"]
            df["nb_heures_ouverture"] = df["nb_heures_ouverture"].apply(lambda x: x if x > 0 else np.nan)
            df.drop(columns=["heure_ouverture","heure_fermeture","heure_ouv_num","heure_ferm_num"],
                    errors="ignore", inplace=True)
        return df


class MLProvider:
    """
    Singleton pour charger et fournir les modèles ML
    """
    _regression = None
    _recommender = None
    _segmentation = None
    _clf_segment = None
    _imputer = None
    _le = None

    @classmethod
    def load_models(cls):
        if cls._clf_segment is None:
            # Maintenant que 'models' est à côté de manage.py, le chemin est direct
            model_path = os.path.join(settings.BASE_DIR, 'models') 
            
            cls._clf_segment = joblib.load(os.path.join(model_path, 'clf_segment.pkl'))
            cls._imputer = joblib.load(os.path.join(model_path, 'seg_imputer.pkl'))
            cls._le = joblib.load(os.path.join(model_path, 'label_encoder.pkl'))

    @classmethod
    def predict_franchise_segment(cls, franchise_obj):
        cls.load_models()

        # On prépare les 14 colonnes dans l'ordre de SEG_FEATURES
        data = {
            "CA_potentielApres2ans": 500000, # Colonne fantôme pour l'imputer
            "investissement_global": franchise_obj.investissement_global or 0,
            "apport_personnel": franchise_obj.apport_personnel or 0,
            "ratio_apport_investissement": (franchise_obj.apport_personnel / franchise_obj.investissement_global) if franchise_obj.investissement_global else 0.3,
            "anciennete": 2026 - getattr(franchise_obj, 'annee_creation', 2020),
            "Implantations": franchise_obj.implantations or 0,
            "rating": 4.0,
            "Redevance_fonctionnement": 5.0,
            "surface_m2": 100.0,
            "nb_heures_ouverture": 12.0,
            "Formations": 1,
            "Aide_auFinancement": 1,
            "YOY_Sales": 0.05,
            "investissement_par_m2": 1500.0 # 14ème colonne
        }

        # L'ordre doit être strictement celui de SEG_FEATURES dans ton script
        cols_ordre = [
            "CA_potentielApres2ans", "investissement_global", "apport_personnel",
            "ratio_apport_investissement", "anciennete", "Implantations",
            "rating", "Redevance_fonctionnement", "surface_m2",
            "nb_heures_ouverture", "Formations", "Aide_auFinancement",
            "YOY_Sales", "investissement_par_m2"
        ]
        
        df = pd.DataFrame([data])[cols_ordre]

        # 1. On passe dans l'imputer (qui attend 14 colonnes)
        X_imp = cls._imputer.transform(df.values)
        
        # 2. Le classifieur, lui, n'en attend que 13 (sans le CA)
        # On retire la première colonne (CA) avant de prédire
        X_for_clf = X_imp[:, 1:] 
        
        try:
            pred_idx = cls._clf_segment.predict(X_for_clf)[0]
            label_txt = cls._le.inverse_transform([pred_idx])[0]
            
            mapping = {
                "Haut Rendement": "haut_rendement",
                "Valeur Sûre": "valeur_sure",
                "Risque Élevé": "risque_eleve"
            }
            return mapping.get(label_txt, 'valeur_sure')
        except Exception as e:
            # Si le classifieur râle encore sur le nombre, on saura combien il en veut
            raise Exception(f"Classifieur : {e}. Il attend {cls._clf_segment.n_features_in_} features.")
    
    @classmethod
    def get_regression(cls):
        if cls._regression is None:
            # Chercher dans plusieurs emplacements possibles
            possible_paths = [
                os.path.join(BASE_ML_DIR, "models", "best_model.pkl"),
                os.path.join(BASE_ML_DIR, "best_model.pkl"),
                os.path.join(os.path.dirname(BASE_ML_DIR), "models", "best_model.pkl"),
            ]
            
            model_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    model_path = p
                    break
            
            if model_path is None:
                raise FileNotFoundError(
                    f"Modèle regression non trouvé. Chemins cherchés: {possible_paths}"
                )
            
            print(f"[MLProvider] Chargement regression depuis: {model_path}")
            cls._regression = RegressionModelWrapper(model_path)
        return cls._regression
    
    @classmethod
    def get_recommender(cls):
        if cls._recommender is None:
            # Ajouter le dossier parent au path pour les imports
            parent_dir = os.path.dirname(BASE_ML_DIR)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            try:
                from recommandation import FranchiseRecommender, load_and_clean
            except ImportError as e:
                print(f"[MLProvider] ERREUR: Impossible d'importer recommandation.py: {e}")
                # Fallback : créer un recommender basique
                cls._recommender = DummyRecommender()
                return cls._recommender

            # Chercher le modèle dans plusieurs emplacements
            possible_paths = [
                os.path.join(BASE_ML_DIR, "models", "franchise_recommender.pkl"),
                os.path.join(BASE_ML_DIR, "franchise_recommender.pkl"),
                os.path.join(parent_dir, "franchise_recommender.pkl"),
                os.path.join(parent_dir, "models", "franchise_recommender.pkl"),
            ]

            recommender_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    recommender_path = p
                    break

            # CAS 1 : modèle trouvé
            if recommender_path:
                print(f"[MLProvider] Chargement recommender depuis: {recommender_path}")
                try:
                    cls._recommender = FranchiseRecommender.load(recommender_path)
                except Exception as e:
                    print(f"[MLProvider] ERREUR chargement: {e}, recréation...")
                    cls._recommender = cls._create_recommender()
            else:
                # CAS 2 : pas trouvé → recréer
                print("[MLProvider] Aucun .pkl trouvé → création du modèle")
                cls._recommender = cls._create_recommender()

        return cls._recommender
    
    @classmethod
    def _create_recommender(cls):
        """Crée un nouveau recommender à partir des données"""
        parent_dir = os.path.dirname(BASE_ML_DIR)
        
        try:
            from recommandation import FranchiseRecommender, load_and_clean
        except ImportError as e:
            print(f"[MLProvider] ERREUR import: {e}")
            return DummyRecommender()

        # Chercher le fichier de données
        data_paths = [
            os.path.join(parent_dir, "data.xlsx"),
            os.path.join(BASE_ML_DIR, "data.xlsx"),
            os.path.join(BASE_ML_DIR, "data", "data.xlsx"),
        ]
        
        data_path = None
        for p in data_paths:
            if os.path.exists(p):
                data_path = p
                break
        
        if data_path is None:
            print(f"[MLProvider] ERREUR: data.xlsx non trouvé")
            return DummyRecommender()

        try:
            df = load_and_clean(data_path)
            recommender = FranchiseRecommender()
            recommender.fit(df)
            
            # Sauvegarde auto pour prochaine fois
            save_path = os.path.join(BASE_ML_DIR, "franchise_recommender.pkl")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            recommender.save(save_path)
            print(f"[MLProvider] Modèle sauvegardé: {save_path}")
            
            return recommender
        except Exception as e:
            print(f"[MLProvider] ERREUR création: {e}")
            return DummyRecommender()
    
    @classmethod
    def get_segmentation(cls):
        """Charge le modèle de segmentation (à implémenter selon ton segmentation.py)"""
        if cls._segmentation is None:
            # Implémentation similaire à get_recommender
            # Adapte selon ton fichier segmentation.py
            pass
        return cls._segmentation


class DummyRecommender:
    """
    Recommender de fallback si le vrai modèle ne charge pas
    Retourne des recommandations vides ou basiques
    """
    def recommend(self, user_profile, top_n=5):
        print("[DummyRecommender] Utilisation du fallback")
        return []
    
    def get_similar_franchises(self, franchise_id, top_n=5):
        return []
    
    def explain_recommendation(self, franchise_id, user_profile):
        return {"reason": "Modèle non disponible"}


# Fonction utilitaire pour tester
def test_ml_provider():
    """Teste le chargement des modèles"""
    print("=== Test ML Provider ===")
    
    try:
        reg = MLProvider.get_regression()
        print(f"✅ Regression chargée: {type(reg)}")
    except Exception as e:
        print(f"❌ Regression: {e}")
    
    try:
        rec = MLProvider.get_recommender()
        print(f"✅ Recommender chargé: {type(rec)}")
    except Exception as e:
        print(f"❌ Recommender: {e}")


if __name__ == "__main__":
    # Configuration minimale Django pour test standalone
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'franchise_invest_ai.settings')
    try:
        import django
        django.setup()
    except:
        pass
    
    test_ml_provider()


