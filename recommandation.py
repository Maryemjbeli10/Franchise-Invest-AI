"""
Franchise Invest AI - Système de Recommandation
================================================
Hybride RÉEL :
  - Content-Based Filtering  (cosinus similarity)
  - Collaborative Filtering  (SVD sur vraies interactions utilisateurs)
  - Fallback intelligent si pas assez d'historique

CORRECTIONS v3 :
  [FIX 1] Filtre ROI minimum — CA >= investissement * roi_min
  [FIX 2] Normalisation des scores hybrides (0-1) pour CAS 1 et CAS 2
  [FIX 3] Filtre cuisine strict avec gestion NaN + log debug
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD


# ─────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────

NUMERIC_FEATURES = [
    'apport_personnel', 'investissement_global', 'CA_potentielApres2ans',
    'Implantations', 'rating_composite', 'food_rating', 'service_rating',
    'Redevance_fonctionnement', 'Redevance_publicitaire', 'franchise_fee',
    'YOY_Sales', 'YOY_Units', 'Surface_m2', 'duree_fonctionnement',
    'duree_ouverture_h', 'performance_score', 'composite_score',
    'pickup_enabled', 'delivery_enabled', 'Formations_bin',
    'Aide_auFinancement_bin', 'has_rating'
]

CAT_FEATURES = ['cuisines', 'Merchant_Category', 'budget_tier']

# Poids des interactions utilisateur pour le CF réel
INTERACTION_WEIGHTS = {
    'view':      1,   # l'utilisateur a consulté la fiche
    'favorite':  3,   # ajouté en favori
    'contact':   5,   # demande de contact / devis
    'dossier':   8,   # création dossier investissement
}


# ─────────────────────────────────────────────
# 1. CHARGEMENT & NETTOYAGE
# ─────────────────────────────────────────────

def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath)

    # Aide_auFinancement
    df['Aide_auFinancement'] = df['Aide_auFinancement'].fillna('non')

    # Pickup / delivery → binaire propre
    df['pickup_enabled']   = (df['pickup_enabled']  > 0.5).astype(int)
    df['delivery_enabled'] = (df['delivery_enabled'] > 0.5).astype(int)

    # YOY et redevances → float
    for col in ['YOY_Sales', 'YOY_Units', 'Redevance_fonctionnement', 'Redevance_publicitaire']:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace('%', '').str.strip(),
            errors='coerce'
        ).fillna(0.0)

    # Surface → numérique
    df['Surface_m2'] = pd.to_numeric(
        df['Surface_moyenne'].astype(str).str.extract(r'(\d+)')[0],
        errors='coerce'
    )
    df['Surface_m2'] = df['Surface_m2'].fillna(df['Surface_m2'].median())

    # Durée ouverture
    def hours_diff(row):
        try:
            h_open  = int(row['heure_ouverture'].replace('h', '').split(':')[0])
            h_close = int(row['heure_fermeture'].replace('h', '').split(':')[0])
            diff = h_close - h_open
            return diff if diff > 0 else diff + 24
        except:
            return 10
    df['duree_ouverture_h'] = df.apply(hours_diff, axis=1)

    # Binaires
    df['Formations_bin']         = (df['Formations'].str.lower() == 'oui').astype(int)
    df['Aide_auFinancement_bin'] = (df['Aide_auFinancement'].str.lower() == 'oui').astype(int)

    # ── FIX RATINGS : 0 = donnée manquante, pas vrai zéro ──
    for col in ['rating', 'food_rating', 'service_rating']:
        df[col] = df[col].replace(0, np.nan)

    df['has_rating'] = df['rating'].notna().astype(int)

    # Rating composite = moyenne des 3 ratings disponibles
    df['rating_composite'] = df[['rating', 'food_rating', 'service_rating']].mean(axis=1)
    median_rating = df['rating_composite'].median()
    df['rating_composite'] = df['rating_composite'].fillna(median_rating)
    df['food_rating']      = df['food_rating'].fillna(df['food_rating'].median())
    df['service_rating']   = df['service_rating'].fillna(df['service_rating'].median())

    # Performance → numérique
    perf_map = {'faible': 0, 'moyenne': 1, 'élevée': 2}
    df['performance_score'] = df['performance_CA'].map(perf_map).fillna(1)

    # Budget tier
    df['budget_tier'] = pd.qcut(
        df['investissement_global'], q=3, labels=['low', 'medium', 'high']
    )

    # Score composite (signal principal)
    s = MinMaxScaler()
    df['composite_score'] = (
        0.30 * s.fit_transform(df[['CA_potentielApres2ans']]).flatten() +
        0.20 * s.fit_transform(df[['rating_composite']]).flatten() +
        0.15 * s.fit_transform(df[['Implantations']]).flatten() +
        0.15 * df['performance_score'] / 2 +
        0.10 * s.fit_transform(df[['YOY_Sales']]).flatten() +
        0.10 * s.fit_transform(df[['duree_fonctionnement']]).flatten()
    )

    # Pénalité franchises sans rating (-10% du score)
    df.loc[df['has_rating'] == 0, 'composite_score'] *= 0.90

    # ── [FIX 3] Normaliser la colonne cuisines : strip + lowercase ──
    # Évite les problèmes de correspondance dus aux espaces ou casse
    if 'cuisines' in df.columns:
        df['cuisines'] = (
            df['cuisines']
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({'nan': np.nan, 'none': np.nan, '': np.nan})
        )

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# 2. MATRICE DE FEATURES
# ─────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame):
    num_df = df[NUMERIC_FEATURES].fillna(0)
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(num_df)

    cat_df = pd.get_dummies(df[CAT_FEATURES], prefix=CAT_FEATURES)
    feature_matrix = np.hstack([num_scaled, cat_df.values.astype(float)])

    return feature_matrix, scaler, cat_df.columns.tolist()


# ─────────────────────────────────────────────
# 3. GESTIONNAIRE D'INTERACTIONS (CF Réel)
# ─────────────────────────────────────────────

class InteractionStore:
    """
    Stocke les interactions utilisateur → franchise.
    Chaque interaction a un poids selon son type.

    Utilisation dans l'app :
        store = InteractionStore('interactions.json')
        store.log('user_42', 'McDonald s', 'favorite')
        store.log('user_42', 'SUBWAY',     'contact')
    """

    def __init__(self, path: str = 'interactions.json'):
        self.path = path
        # {user_id: {franchise_nom: score_cumulé}}
        self.data: dict = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)

    def _save(self):
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def log(self, user_id: str, franchise_name: str, action: str):
        """
        Enregistre une interaction.

        Args:
            user_id       : identifiant unique de l'utilisateur
            franchise_name: nom exact de la franchise
            action        : 'view' | 'favorite' | 'contact' | 'dossier'
        """
        weight = INTERACTION_WEIGHTS.get(action, 1)
        if user_id not in self.data:
            self.data[user_id] = {}
        prev = self.data[user_id].get(franchise_name, 0)
        self.data[user_id][franchise_name] = prev + weight
        self._save()
        print(f"[Interaction] {user_id} → {franchise_name} ({action}, +{weight}pts)")

    def get_user_scores(self, user_id: str) -> dict:
        """Retourne {franchise_name: score} pour un utilisateur."""
        return self.data.get(user_id, {})

    def get_all_users(self) -> list:
        return list(self.data.keys())

    def total_interactions(self) -> int:
        return sum(len(v) for v in self.data.values())

    def build_matrix(self, franchise_names: list):
        """
        Construit la matrice utilisateur × franchise pour le SVD.
        Retourne (matrix, user_ids)
        """
        users = self.get_all_users()
        if not users:
            return None, []

        mat = np.zeros((len(users), len(franchise_names)))
        fname_to_idx = {n: i for i, n in enumerate(franchise_names)}

        for u_idx, user_id in enumerate(users):
            for fname, score in self.data[user_id].items():
                if fname in fname_to_idx:
                    mat[u_idx, fname_to_idx[fname]] = score

        return mat, users


# ─────────────────────────────────────────────
# 4. MODÈLE DE RECOMMANDATION
# ─────────────────────────────────────────────

class FranchiseRecommender:
    """
    Système de recommandation hybride RÉEL.

    - Content-Based  : cosinus similarity sur profil franchise
    - Collaborative  : SVD sur vraies interactions si disponibles,
                       sinon fallback sur scoring par segment budget

    CORRECTIONS v3 :
      [FIX 1] Filtre ROI minimum
      [FIX 2] Normalisation scores hybrides 0-1
      [FIX 3] Filtre cuisine strict avec gestion NaN
    """

    def __init__(self, interaction_store: InteractionStore = None):
        self.df              = None
        self.feature_matrix  = None
        self.scaler          = None
        self.cat_cols        = None
        self.nn_model        = None
        self.is_fitted       = False
        self.store           = interaction_store or InteractionStore()
        # Fallback CF (segment budget)
        self._svd_fallback        = None
        self._svd_matrix_fallback = None
        self._cf_item_fallback    = None

    # ── Entraînement ──────────────────────────

    def fit(self, df: pd.DataFrame):
        self.df = df.copy()
        self.feature_matrix, self.scaler, self.cat_cols = build_feature_matrix(df)

        self.nn_model = NearestNeighbors(
            n_neighbors=20, metric='cosine', algorithm='brute', n_jobs=-1
        )
        self.nn_model.fit(self.feature_matrix)

        self._build_fallback_cf()
        self.is_fitted = True
        print(f"[OK] Modèle entraîné sur {len(df)} franchises.")
        return self

    def _build_fallback_cf(self):
        """CF par segment budget — utilisé tant qu'il n'y a pas assez d'interactions."""
        df = self.df.copy()
        df['_tier_num'] = df['budget_tier'].map({'low': 0, 'medium': 1, 'high': 2})
        mat = np.zeros((3, len(df)))
        for t in range(3):
            mask = df['_tier_num'] == t
            for idx, score in zip(df.index[mask], df.loc[mask, 'composite_score'].values):
                mat[t, idx] = score
        self._svd_fallback        = TruncatedSVD(n_components=2)
        self._svd_matrix_fallback = self._svd_fallback.fit_transform(mat)
        self._cf_item_fallback    = self._svd_fallback.components_

    def _build_real_cf(self, user_id: str):
        """
        CF réel basé sur les interactions de TOUS les utilisateurs.
        Retourne le vecteur de scores pour user_id, ou None si pas assez de données.
        """
        franchise_names = self.df['nom'].tolist()
        mat, users = self.store.build_matrix(franchise_names)

        # Minimum 3 utilisateurs avec interactions pour activer le vrai CF
        if mat is None or len(users) < 3 or user_id not in users:
            return None

        # SVD sur la matrice complète
        n_components = min(10, len(users) - 1, mat.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        user_factors = svd.fit_transform(mat)   # (n_users, k)
        item_factors = svd.components_           # (k, n_franchises)

        # Score pour cet utilisateur
        u_idx = users.index(user_id)
        scores = user_factors[u_idx] @ item_factors  # (n_franchises,)

        # Normalise 0-1
        mn, mx = scores.min(), scores.max()
        if mx > mn:
            scores = (scores - mn) / (mx - mn)

        return scores

    # ── Scores internes ───────────────────────

    def _get_budget_tier(self, budget: float) -> str:
        q1 = self.df['investissement_global'].quantile(0.33)
        q2 = self.df['investissement_global'].quantile(0.66)
        if budget <= q1:   return 'low'
        elif budget <= q2: return 'medium'
        else:              return 'high'

    def _content_score(self, user_vector: np.ndarray) -> np.ndarray:
        return cosine_similarity(
            user_vector.reshape(1, -1), self.feature_matrix
        ).flatten()

    def _fallback_cf_score(self, budget_tier: str) -> np.ndarray:
        tier_idx = {'low': 0, 'medium': 1, 'high': 2}.get(budget_tier, 1)
        scores   = self._svd_matrix_fallback[tier_idx] @ self._cf_item_fallback
        mn, mx   = scores.min(), scores.max()
        return (scores - mn) / (mx - mn) if mx > mn else scores

    def _build_user_vector(self, budget: float, preferences: dict) -> np.ndarray:
        df      = self.df
        tier    = self._get_budget_tier(budget)
        # [FIX 3] cuisine normalisée avant comparaison
        cuisine = preferences.get('cuisine', '').strip().lower()

        mask = df['budget_tier'] == tier
        if cuisine:
            sub = df[mask & (df['cuisines'] == cuisine)]
        else:
            sub = df[mask]
        if len(sub) == 0:
            sub = df[mask]
        if len(sub) == 0:
            sub = df

        num_vals   = sub[NUMERIC_FEATURES].fillna(0).mean().values
        num_scaled = self.scaler.transform(num_vals.reshape(1, -1)).flatten()

        cuisine_val = cuisine if cuisine else (
            sub['cuisines'].mode()[0] if len(sub) > 0 else 'non spécifié'
        )
        merchant = sub['Merchant_Category'].mode()[0] if len(sub) > 0 else 'Non spécifié'

        fake_row    = pd.DataFrame([{
            'cuisines': cuisine_val,
            'Merchant_Category': merchant,
            'budget_tier': tier
        }])
        cat_dummies = pd.get_dummies(fake_row, prefix=CAT_FEATURES)
        for col in self.cat_cols:
            if col not in cat_dummies.columns:
                cat_dummies[col] = 0
        cat_vec = cat_dummies[self.cat_cols].values.astype(float).flatten()

        return np.hstack([num_scaled, cat_vec])

    def _apply_hard_filters(self, budget: float, preferences: dict) -> np.ndarray:
        df   = self.df
        mask = df['apport_personnel'] <= budget

        # ── [FIX 3] Filtre cuisine strict avec gestion NaN ──────────────
        cuisine = preferences.get('cuisine', '').strip().lower()
        if cuisine and cuisine not in ['all', 'tous', '']:
            # Exclut les lignes où cuisines est NaN
            cuisine_mask = df['cuisines'].notna() & (df['cuisines'] == cuisine)
            n_cuisine = cuisine_mask.sum()
            print(f"[Debug] Cuisine '{cuisine}' : {n_cuisine} franchises trouvées")
            if n_cuisine > 0:
                mask = mask & cuisine_mask
            else:
                print(f"[Warning] Aucune franchise pour la cuisine '{cuisine}' — filtre ignoré")

        # Ville
        ville = preferences.get('ville', '').strip().lower()
        if ville and ville not in ['all', 'tous', '']:
            mask = mask & (
                df['adresses'].str.lower().str.contains(ville, na=False)
            )

        if preferences.get('delivery_required'):
            mask = mask & (df['delivery_enabled'] == 1)

        if preferences.get('pickup_required'):
            mask = mask & (df['pickup_enabled'] == 1)

        invest_max = preferences.get('investissement_max')
        if invest_max:
            mask = mask & (df['investissement_global'] <= invest_max)

        perf_min = preferences.get('performance_min')
        if perf_min is not None:
            mask = mask & (df['performance_score'] >= perf_min)

        if preferences.get('formations_required'):
            mask = mask & (df['Formations_bin'] == 1)

        if preferences.get('aide_financement_required'):
            mask = mask & (df['Aide_auFinancement_bin'] == 1)

        # ── [FIX 1] Filtre ROI minimum ───────────────────────────────────
        # CA potentiel >= investissement * roi_min
        # Par défaut roi_min=1.2 → CA doit être au moins 20% supérieur à l'investissement
        roi_min = preferences.get('roi_min', 1.2)
        roi_mask = df['CA_potentielApres2ans'] >= (df['investissement_global'] * roi_min)
        n_before_roi = mask.sum()
        mask = mask & roi_mask
        n_after_roi  = mask.sum()
        print(f"[FIX 1 ROI] Filtre ROI x{roi_min} : {n_before_roi} → {n_after_roi} franchises")

        return mask.values

    # ── Interface publique ────────────────────

    def recommend(
        self,
        budget: float,
        preferences: dict = None,
        top_n: int = 10,
        alpha: float = 0.65,
        user_id: str = None
    ) -> pd.DataFrame:
        """
        Recommande les top_n franchises.

        Args:
            budget      : apport personnel disponible (€)
            preferences : filtres (voir ci-dessous)
            top_n       : nombre de résultats
            alpha       : poids content-based (0-1). 1-alpha = poids CF
            user_id     : identifiant utilisateur pour CF réel (optionnel)

        Preferences disponibles :
            cuisine              str    ex. 'italienne'
            ville                str    ex. 'Paris'
            delivery_required    bool
            pickup_required      bool
            investissement_max   float
            performance_min      int    0=faible 1=moyenne 2=élevée
            formations_required  bool
            aide_financement_required bool
            prefer_high_rating   bool
            roi_min              float  défaut=1.2 (CA >= invest * 1.2)
        """
        if preferences is None:
            preferences = {}

        # 1. Filtres durs (inclut FIX 1 ROI + FIX 3 cuisine)
        hard_mask = self._apply_hard_filters(budget, preferences)

        if hard_mask.sum() < top_n:
            print(f"[Warning] {hard_mask.sum()} résultats après filtres — relâchement des filtres optionnels.")
            # Relâchement : on garde budget + ROI mais on retire les autres filtres
            relaxed_mask = self.df['apport_personnel'].values <= budget
            roi_min = preferences.get('roi_min', 1.2)
            relaxed_mask = relaxed_mask & (
                self.df['CA_potentielApres2ans'].values >=
                self.df['investissement_global'].values * roi_min
            )
            hard_mask = relaxed_mask

        eligible_idx = np.where(hard_mask)[0]

        # 2. Content-based score
        user_vector = self._build_user_vector(budget, preferences)
        cb_scores   = self._content_score(user_vector)

        # 3. Collaborative score : réel si possible, sinon fallback
        cf_real = self._build_real_cf(user_id) if user_id else None

        if cf_real is not None:
            cf_scores = cf_real
            cf_mode   = 'CF Réel (interactions utilisateurs)'
        else:
            cf_scores = self._fallback_cf_score(self._get_budget_tier(budget))
            cf_mode   = 'CF Fallback (segment budget)'

        # 4. Score hybride brut
        hybrid = alpha * cb_scores + (1 - alpha) * cf_scores

        # 5. Bonus soft preferences
        cuisine_pref = preferences.get('cuisine', '').strip().lower()
        if cuisine_pref:
            hybrid += (
                self.df['cuisines'] == cuisine_pref
            ).astype(float).values * 0.05

        if preferences.get('prefer_high_rating'):
            hybrid += MinMaxScaler().fit_transform(
                self.df[['rating_composite']]
            ).flatten() * 0.05

        # 6. Pénalité franchises sans rating
        no_rating = (self.df['has_rating'] == 0).values
        hybrid[no_rating] *= 0.90

        # ── [FIX 2] Normalisation globale 0-1 ──────────────────────────
        # Garantit que CAS 1 (fallback) et CAS 2 (CF réel) ont la même échelle
        # On normalise uniquement sur les éligibles pour ne pas biaiser
        eligible_scores = hybrid[eligible_idx]
        mn, mx = eligible_scores.min(), eligible_scores.max()
        if mx > mn:
            hybrid[eligible_idx] = (eligible_scores - mn) / (mx - mn)

        # 7. Appliquer masque et trier
        masked = np.full(len(self.df), -np.inf)
        masked[eligible_idx] = hybrid[eligible_idx]

        top_idx = np.argsort(masked)[::-1][:top_n]
        result  = self.df.iloc[top_idx].copy()
        result['recommendation_score'] = np.round(hybrid[top_idx], 4)
        result['rank'] = range(1, len(result) + 1)

        print(f"[Mode CF] {cf_mode}")
        print(f"[Résultats] {len(result)} franchises recommandées")

        # Vérification ROI pour chaque résultat (log de contrôle)
        for _, row in result.iterrows():
            roi = row['CA_potentielApres2ans'] / row['investissement_global']
            if roi < 1.0:
                print(f"  [Alerte ROI] {row['nom']} — CA/Invest = {roi:.2f}x ← inférieur à 1")

        output_cols = [
            'rank', 'nom', 'cuisines', 'adresses', 'apport_personnel',
            'investissement_global', 'CA_potentielApres2ans',
            'rating_composite', 'food_rating', 'service_rating',
            'performance_CA', 'Implantations', 'pickup_enabled',
            'delivery_enabled', 'Formations', 'Aide_auFinancement',
            'YOY_Sales', 'YOY_Units', 'Merchant_Category',
            'has_rating', 'composite_score', 'recommendation_score'
        ]
        cols = [c for c in output_cols if c in result.columns]
        return result[cols].reset_index(drop=True)

    def similar(self, franchise_name: str, top_n: int = 5) -> pd.DataFrame:
        """Franchises les plus similaires à une franchise donnée."""
        matches = self.df[
            self.df['nom'].str.lower().str.contains(franchise_name.lower(), na=False)
        ]
        if matches.empty:
            print(f"Franchise '{franchise_name}' non trouvée.")
            return pd.DataFrame()

        idx = matches.index[0]
        distances, indices = self.nn_model.kneighbors(
            self.feature_matrix[idx].reshape(1, -1), n_neighbors=top_n + 1
        )
        similar_idx = indices.flatten()[1:]
        result = self.df.iloc[similar_idx].copy()
        result['similarity'] = (1 - distances.flatten()[1:]).round(4)
        result['rank'] = range(1, len(result) + 1)

        return result[[
            'rank', 'nom', 'cuisines', 'investissement_global',
            'CA_potentielApres2ans', 'rating_composite',
            'performance_CA', 'similarity'
        ]].reset_index(drop=True)

    def explain(self, franchise_name: str, budget: float) -> None:
        """Explique pourquoi une franchise est recommandée."""
        matches = self.df[
            self.df['nom'].str.lower().str.contains(franchise_name.lower(), na=False)
        ]
        if matches.empty:
            print("Franchise non trouvée.")
            return
        row = matches.iloc[0]

        roi = row['CA_potentielApres2ans'] / row['investissement_global']

        print(f"\n📋 Explication — {row['nom']}")
        print("-" * 45)
        if row['apport_personnel'] <= budget:
            print(f"  ✅ Apport requis ({row['apport_personnel']:,.0f}€) dans votre budget")
        if row['performance_CA'] == 'élevée':
            print("  🚀 Performance CA élevée")
        if row['has_rating'] == 1 and row['rating_composite'] > 1.5:
            print(f"  ⭐ Très bien noté ({row['rating_composite']:.2f}/2.2)")
        elif row['has_rating'] == 0:
            print("  ⚠️  Pas encore de note disponible")
        if row['YOY_Sales'] > 20:
            print(f"  📈 Croissance ventes +{row['YOY_Sales']:.1f}%")
        if row['delivery_enabled'] == 1:
            print("  🛵 Livraison disponible")
        if row['Formations_bin'] == 1:
            print("  🎓 Formation incluse")
        if row['Aide_auFinancement_bin'] == 1:
            print("  💰 Aide au financement disponible")
        if row['Implantations'] > self.df['Implantations'].median():
            print(f"  🌍 Large réseau ({row['Implantations']:.0f} implantations)")
        print(f"\n  CA potentiel 2 ans : {row['CA_potentielApres2ans']:,.0f}€")
        print(f"  Investissement     : {row['investissement_global']:,.0f}€")
        print(f"  ROI estimé         : {roi:.2f}x ({'✅ rentable' if roi >= 1.2 else '⚠️ faible rentabilité'})")

    def get_filter_options(self) -> dict:
        """Options pour alimenter les dropdowns du frontend."""
        df = self.df
        return {
            'cuisines'            : sorted(df['cuisines'].dropna().unique().tolist()),
            'villes'              : sorted(df['adresses'].dropna().unique().tolist()[:150]),
            'merchant_categories' : sorted(df['Merchant_Category'].dropna().unique().tolist()),
            'budget_min'          : round(float(df['apport_personnel'].min()), 0),
            'budget_max'          : round(float(df['apport_personnel'].max()), 0),
            'investissement_min'  : round(float(df['investissement_global'].min()), 0),
            'investissement_max'  : round(float(df['investissement_global'].max()), 0),
            'performance_options' : ['faible', 'moyenne', 'élevée'],
            'roi_min_default'     : 1.2,
        }

    # ── Sauvegarde / Chargement ───────────────

    def save(self, path: str = 'franchise_recommender.pkl'):
        joblib.dump(self, path)
        print(f"[OK] Modèle sauvegardé → {path}")

    @classmethod
    def load(cls, path: str = 'franchise_recommender.pkl'):
        model = joblib.load(path)
        print(f"[OK] Modèle chargé depuis {path}")
        return model


# ─────────────────────────────────────────────
# 5. EXÉCUTION & DÉMONSTRATION
# ─────────────────────────────────────────────

if __name__ == '__main__':

    print("=" * 55)
    print("  FRANCHISE INVEST AI — Recommandation v3 (corrigée)")
    print("=" * 55)

    # ── Chargement & entraînement ──────────────
    df = load_and_clean('data.xlsx')
    print(f"\nDonnées : {len(df)} franchises chargées et nettoyées")
    print(f"Ratings manquants corrigés : {(df['has_rating']==0).sum()} franchises sans note")

    # Contrôle FIX 3 : vérifier les cuisines normalisées
    print(f"\nCuisines disponibles après normalisation :")
    print(sorted(df['cuisines'].dropna().unique().tolist()))

    store = InteractionStore('interactions.json')
    model = FranchiseRecommender(interaction_store=store)
    model.fit(df)
    model.save('franchise_recommender.pkl')

    # ══════════════════════════════════════════
    # SIMULATION : ce que fait l'app en pratique
    # ══════════════════════════════════════════

    print("\n" + "=" * 55)
    print("  SIMULATION UTILISATEUR")
    print("=" * 55)

    user_id = "investor_001"

    print(f"\n[Étape 1] L'investisseur '{user_id}' consulte et interagit :")
    store.log(user_id, "TUTTI PIZZA",   "view")
    store.log(user_id, "TUTTI PIZZA",   "favorite")
    store.log(user_id, "SUBWAY",        "view")
    store.log(user_id, "WENDY'S",       "view")
    store.log(user_id, "WENDY'S",       "contact")

    # Autres utilisateurs (simulation plateforme)
    store.log("investor_002", "TUTTI PIZZA",  "dossier")
    store.log("investor_002", "WENDY'S",      "favorite")
    store.log("investor_002", "SUBWAY",       "contact")
    store.log("investor_003", "TUTTI PIZZA",  "contact")
    store.log("investor_003", "SUBWAY",       "favorite")
    store.log("investor_003", "WENDY'S",      "view")

    print(f"\nTotal interactions enregistrées : {store.total_interactions()}")

    # ── CAS 1 : Visiteur sans compte ──────────
    print("\n" + "─" * 55)
    print("  CAS 1 : Visiteur sans compte")
    print("  Budget=80 000€ | Cuisine=italienne | Perf≥moyenne")
    print("─" * 55)

    recs_anon = model.recommend(
        budget=80000,
        preferences={
            'cuisine': 'italienne',
            'performance_min': 1,
            'roi_min': 1.2,         # [FIX 1] CA >= invest * 1.2
        },
        top_n=5,
        user_id=None
    )
    print(recs_anon[[
        'rank', 'nom', 'investissement_global', 'CA_potentielApres2ans',
        'rating_composite', 'has_rating', 'performance_CA', 'recommendation_score'
    ]].to_string(index=False))

    # Contrôle ROI
    print("\n  Contrôle ROI des recommandations :")
    for _, row in recs_anon.iterrows():
        roi = row['CA_potentielApres2ans'] / row['investissement_global']
        print(f"    {row['nom'][:30]:30s} | ROI = {roi:.2f}x")

    # ── CAS 2 : Investisseur connecté ─────────
    print("\n" + "─" * 55)
    print(f"  CAS 2 : Investisseur connecté (user_id='{user_id}')")
    print("  Même formulaire + historique d'interactions")
    print("─" * 55)

    recs_user = model.recommend(
        budget=80000,
        preferences={
            'cuisine': 'italienne',
            'performance_min': 1,
            'roi_min': 1.2,         # [FIX 1]
        },
        top_n=5,
        user_id=user_id
    )
    print(recs_user[[
        'rank', 'nom', 'investissement_global', 'CA_potentielApres2ans',
        'rating_composite', 'has_rating', 'performance_CA', 'recommendation_score'
    ]].to_string(index=False))

    # ── Franchises similaires ─────────────────
    print("\n" + "─" * 55)
    print("  Franchises similaires à WENDY'S")
    print("─" * 55)
    sims = model.similar("WENDY'S", top_n=5)
    print(sims.to_string(index=False))

    # ── Explication (avec ROI affiché) ────────
    print("\n" + "─" * 55)
    print("  Explication — TUTTI PIZZA")
    print("─" * 55)
    model.explain("TUTTI PIZZA", budget=100000)

    # ── Options filtres frontend ──────────────
    print("\n" + "─" * 55)
    print("  Options disponibles pour le frontend")
    print("─" * 55)
    opts = model.get_filter_options()
    print(f"Cuisines       : {opts['cuisines']}")
    print(f"Budget range   : {opts['budget_min']:,.0f}€ → {opts['budget_max']:,.0f}€")
    print(f"Invest. range  : {opts['investissement_min']:,.0f}€ → {opts['investissement_max']:,.0f}€")
    print(f"ROI min défaut : {opts['roi_min_default']}x")

    print("\n✅ Système v3 opérationnel.")
    print("   franchise_recommender.pkl  → modèle sauvegardé")
    print("   interactions.json          → historique interactions")
    print("\nCorrections appliquées :")
    print("   [FIX 1] ✅ Filtre ROI minimum (CA >= invest * roi_min)")
    print("   [FIX 2] ✅ Normalisation scores 0-1 (CAS 1 = CAS 2 même échelle)")
    print("   [FIX 3] ✅ Filtre cuisine strict + gestion NaN + debug log")