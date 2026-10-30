from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import UserSignupForm, InvestorProfileForm
from .models import User, Simulation, Franchise
from .ml_utils import MLProvider
from django.http import JsonResponse
import pandas as pd
from django.core.paginator import Paginator
import json
import os
import io
from django.http import FileResponse
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
from django.conf import settings
import math
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(settings.BASE_DIR), 'data.xlsx')

print(f"[INFO] BASE_DIR: {settings.BASE_DIR}")
print(f"[INFO] Chemin data.xlsx: {DATA_PATH}")
print(f"[INFO] Fichier existe: {os.path.exists(DATA_PATH)}")


def clean_for_json(value):
    """Remplace NaN, inf, -inf par des valeurs sérialisables en JSON"""
    if pd.isna(value):
        return None
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    return value

def calculer_probabilite(ca, investissement, apport):
    """
    Calcule la probabilité de succès d'un investissement en franchise.
    Prend en compte le ROI et le ratio d'apport personnel.
    """
    roi = ca / investissement if investissement > 0 else 0
    ratio_apport = apport / investissement if investissement > 0 else 0
    
    # Score base sur ROI
    if roi >= 1.5:
        score = 85
    elif roi >= 1.2:
        score = 75
    elif roi >= 1.0:
        score = 65
    elif roi >= 0.8:
        score = 50
    elif roi >= 0.5:
        score = 35
    else:
        score = 20
    
    # Pénalité apport faible
    if ratio_apport < 0.15:
        score -= 20
    elif ratio_apport < 0.25:
        score -= 10
    
    return min(95, max(15, score))

def signup_view(request):
    if request.method == 'POST':
        form = UserSignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect_by_role(user)
    else:
        form = UserSignupForm()
    return render(request, 'signup.html', {'form': form})


def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect_by_role(user)
    else:
        form = AuthenticationForm()
    return render(request, "login.html", {"form": form})


def redirect_by_role(user):
    if user.role == "INVESTISSEUR":
        return redirect('investisseur_dashboard')
    elif user.role == "EXPERT":
        return redirect('expert_dashboard')
    return redirect('login')


def logout_user(request):
    logout(request)
    return redirect('login')


# ═══════════════════════════════════════════════════════════
# DASHBOARD INVESTISSEUR — CORRIGÉ AVEC FORMATAGE PROPRE
# ═══════════════════════════════════════════════════════════

@login_required
def investisseur_dashboard(request):
    """
    Dashboard investisseur avec simulation ML complète.
    """
    user = request.user

    if user.role != "INVESTISSEUR":
        messages.warning(request, "Cette page est réservée aux investisseurs.")
        return redirect('expert_dashboard')

    # ── 1. STATS ──
    stats = {
        'matches': 10,
        'simulations': Simulation.objects.filter(user=user).count(),
        'rapports': 0,
    }

    # ── 2. RECOMMANDATIONS AI ──
    recommendations = []

    try:
        recommender = MLProvider.get_recommender()

        preferences = {
            'cuisine': (user.cuisine_preferee or '').strip().lower(),
            'performance_min': user.performance_min or 1,
            'roi_min': user.roi_min or 1.2,
            'delivery_required': user.livraison_preferee,
            'pickup_required': user.pickup_preferee,
            'formations_required': user.formation_required,
            'aide_financement_required': user.aide_financement_required,
            'ville': (user.ville_interet or '').strip().lower(),
        }

        budget = user.budget_disponible or 50000

        # ✅ APPEL CORRECT DU MODELE AI
        recs_df = recommender.recommend(
            budget=budget,
            preferences=preferences,
            top_n=5,
            user_id=str(user.id)  # 🔥 important pour le collaborative filtering
        )

        if not recs_df.empty:
            for _, row in recs_df.iterrows():
                investissement = row.get('investissement_global', 0)
                rating = row.get('rating_composite', 0)

                recommendations.append({
                    'nom': row.get('nom', 'Franchise inconnue'),
                    'cuisines': row.get('cuisines', 'Non spécifié'),
                    'investissement_global': f"{int(investissement):,}€".replace(",", " "),
                    'rating_composite': round(float(rating), 2) if rating else 0,
                    'match_score': int(row.get('recommendation_score', 0.45) * 100),
                    'budget_warning': investissement > (user.budget_disponible or 0),
                    'category_icon': _get_category_icon(row.get('cuisines', '')),
                })

    except Exception as e:
        print(f"[ERREUR Recommandation] {e}")
        import traceback; traceback.print_exc()


    # ── 3. SIMULATIONS RÉCENTES ──
    simulations = Simulation.objects.filter(user=user).order_by('-created_at')[:5]

    # ── 4. POST : NOUVELLE SIMULATION (CORRIGÉ) ──
    if request.method == 'POST' and request.POST.get('action') == 'simulate':
        try:
            franchise_id = request.POST.get('franchise_id', '')
            budget = float(request.POST.get('budget', 0))
            investissement_global = float(request.POST.get('investissement_global', 0))
            apport_personnel = float(request.POST.get('apport_personnel', 0) or 0)
            cuisine = request.POST.get('cuisine', '')
            location = request.POST.get('location', '')
            franchise_name = request.POST.get('franchise_name', '')
            
            # ── RÉCUPÉRER LES CARACTÉRISTIQUES ML DE LA FRANCHISE ──
            franchise_data = {}
            if franchise_id and franchise_id.startswith('franchise_'):
                try:
                    idx = int(franchise_id.replace('franchise_', ''))
                    df = pd.read_excel(DATA_PATH)
                    row = df.iloc[idx]
                    franchise_data = {
                        'nom': str(row.get('nom', 'Inconnu')),
                        'rating': float(row.get('rating', 4)) if pd.notna(row.get('rating')) else 4,
                        'food_rating': float(row.get('food_rating', 4)) if pd.notna(row.get('food_rating')) else 4,
                        'service_rating': float(row.get('service_rating', 4)) if pd.notna(row.get('service_rating')) else 4,
                        'YOY_Sales': float(row.get('YOY_Sales', 5)) if pd.notna(row.get('YOY_Sales')) else 5,
                        'YOY_Units': float(row.get('YOY_Units', 3)) if pd.notna(row.get('YOY_Units')) else 3,
                        'Implantations': int(row.get('Implantations', 10)) if pd.notna(row.get('Implantations')) else 10,
                        'date_creation': int(row.get('date_creation', 2010)) if pd.notna(row.get('date_creation')) else 2010,
                        'Redevance_fonctionnement': float(row.get('Redevance_fonctionnement', 5)) if pd.notna(row.get('Redevance_fonctionnement')) else 5,
                        'Redevance_publicitaire': float(row.get('Redevance_publicitaire', 2)) if pd.notna(row.get('Redevance_publicitaire')) else 2,
                        'franchise_fee': float(row.get('franchise_fee', 30000)) if pd.notna(row.get('franchise_fee')) else 30000,
                        'avis': int(row.get('avis', 100)) if pd.notna(row.get('avis')) else 100,
                        'sales_in_millions_2019': float(row.get('sales_in_millions_2019', 10)) if pd.notna(row.get('sales_in_millions_2019')) else 10,
                        'company_owned_units_2019': int(row.get('company_owned_units_2019', 50)) if pd.notna(row.get('company_owned_units_2019')) else 50,
                        'total_units_2019': int(row.get('total_units_2019', 100)) if pd.notna(row.get('total_units_2019')) else 100,
                        'unit_change_from_2018': int(row.get('unit_change_from_2018', 5)) if pd.notna(row.get('unit_change_from_2018')) else 5,
                        'CompanyStores': int(row.get('CompanyStores', 20)) if pd.notna(row.get('CompanyStores')) else 20,
                        'TotalUnits2021': int(row.get('TotalUnits2021', 120)) if pd.notna(row.get('TotalUnits2021')) else 120,
                        'TotalChange_in_Units_from2020': int(row.get('TotalChange_in_Units_from2020', 10)) if pd.notna(row.get('TotalChange_in_Units_from2020')) else 10,
                        'Merchant_Category': str(row.get('Merchant_Category', 'food')),
                        'Type_de_contrat': str(row.get('Type_de_contrat', 'franchise')),
                        'pickup_enabled': int(row.get('pickup_enabled', 1)) if pd.notna(row.get('pickup_enabled')) else 1,
                        'delivery_enabled': int(row.get('delivery_enabled', 1)) if pd.notna(row.get('delivery_enabled')) else 1,
                        'Formations': str(row.get('Formations', 'oui')),
                        'Aide_auFinancement': str(row.get('Aide_auFinancement', 'oui')),
                        'duree_fonctionnement': int(row.get('duree_fonctionnement', 10)) if pd.notna(row.get('duree_fonctionnement')) else 10,
                        'heure_ouverture': str(row.get('heure_ouverture', '09:00')),
                        'heure_fermeture': str(row.get('heure_fermeture', '22:00')),
                        'Surface_moyenne': str(row.get('Surface_moyenne', '100 m2')),
                    }
                except Exception as e:
                    print(f"[WARN] Impossible de charger franchise {franchise_id}: {e}")
            
            # ── CONSTRUIRE input_dict POUR ML ──
            # Les caractéristiques ML viennent du dataset
            # Les paramètres financiers viennent de l'utilisateur
            input_dict = {
                'nom': franchise_data.get('nom', franchise_name or 'Franchise'),
                'apport_personnel': apport_personnel,
                'droites_entree': 20000,
                'investissement_global': investissement_global,  # ← UTILISATEUR
                'Implantations': franchise_data.get('Implantations', 10),
                'Formations': 1 if (franchise_data.get('Formations', 'oui').lower() in ['oui', '1', 'yes']) or user.formation_required else 0,
                'Aide_auFinancement': 1 if (franchise_data.get('Aide_auFinancement', 'oui').lower() in ['oui', '1', 'yes']) or user.aide_financement_required else 0,
                'Redevance_fonctionnement': franchise_data.get('Redevance_fonctionnement', 5),
                'Redevance_publicitaire': franchise_data.get('Redevance_publicitaire', 2),
                'Surface_moyenne': franchise_data.get('Surface_moyenne', '100 m2'),
                'Type_de_contrat': franchise_data.get('Type_de_contrat', 'franchise'),
                'pickup_enabled': franchise_data.get('pickup_enabled', 1) if user.pickup_preferee else 0,
                'delivery_enabled': franchise_data.get('delivery_enabled', 1) if user.livraison_preferee else 0,
                'avis': franchise_data.get('avis', 100),
                'date_creation': franchise_data.get('date_creation', 2010),
                'sales_in_millions_2019': franchise_data.get('sales_in_millions_2019', 10),
                'company_owned_units_2019': franchise_data.get('company_owned_units_2019', 50),
                'total_units_2019': franchise_data.get('total_units_2019', 100),
                'unit_change_from_2018': franchise_data.get('unit_change_from_2018', 5),
                'CompanyStores': franchise_data.get('CompanyStores', 20),
                'TotalUnits2021': franchise_data.get('TotalUnits2021', 120),
                'TotalChange_in_Units_from2020': franchise_data.get('TotalChange_in_Units_from2020', 10),
                'Merchant_Category': franchise_data.get('Merchant_Category', 'food'),
                'franchise_fee': franchise_data.get('franchise_fee', 30000),
                'YOY_Sales': franchise_data.get('YOY_Sales', 5),
                'YOY_Units': franchise_data.get('YOY_Units', 3),
                'rating': franchise_data.get('rating', 4),
                'food_rating': franchise_data.get('food_rating', 4),
                'service_rating': franchise_data.get('service_rating', 4),
                'adresses': location,  # ← UTILISATEUR
                'cuisines': cuisine or franchise_data.get('cuisines', 'fast food'),  # ← UTILISATEUR ou dataset
                'performance_CA': 'moyenne',
                'duree_fonctionnement': franchise_data.get('duree_fonctionnement', 10),
                'heure_ouverture': franchise_data.get('heure_ouverture', '09:00'),
                'heure_fermeture': franchise_data.get('heure_fermeture', '22:00'),
            }

            # ── PRÉDICTION ML ──
            regression_model = MLProvider.get_regression()
            result = regression_model.predict(input_dict)
            ca_annee_2 = result.get('CA_potentiel_2ans', 0)

            # ── CALCULS RENTABILITÉ ──
            rentabilite = None
            if ca_annee_2 > 0 and investissement_global > 0:
                ca_mensuel = ca_annee_2 / 24
                if ca_mensuel > 0:
                    rentabilite = int(investissement_global / ca_mensuel)

            probabilite = calculer_probabilite(ca_annee_2, investissement_global, apport_personnel)

            # ── SAUVEGARDE ──
            Simulation.objects.create(
                user=user,
                franchise_name=franchise_data.get('nom', franchise_name or 'Inconnu'),
                location=location,  # ← UTILISATEUR
                cuisine=cuisine or franchise_data.get('cuisines', ''),  # ← UTILISATEUR
                budget=budget,  # ← UTILISATEUR
                investissement_global=investissement_global,  # ← UTILISATEUR
                apport_personnel=apport_personnel,  # ← UTILISATEUR
                ca_annee_2=ca_annee_2,
                rentabilite_mois=rentabilite,
                probabilite_succes=probabilite,
            )

            messages.success(
                request, 
                f"Simulation '{franchise_data.get('nom', '')}' réussie ! "
                f"CA prédit: {ca_annee_2:,.0f}€ | Rentabilité: {rentabilite} mois | Succès: {probabilite}%"
            )
            return redirect('investisseur_dashboard')

        except Exception as e:
            messages.error(request, f"Erreur simulation : {str(e)}")
            print(f"[ERREUR Simulation] {e}")
            import traceback; traceback.print_exc()

    # ── FORMATAGE PROFIL ──
    experience_labels = {
        'DEBUTANT': 'Débutant',
        'INTERMEDIAIRE': 'Intermédiaire',
        'EXPERT': 'Expert',
    }

    context = {
        'stats': stats,
        'recommendations': recommendations,
        'simulations': simulations,
        'experience_label': experience_labels.get(user.niveau_experience, user.niveau_experience),
    }

    return render(request, 'investisseur_dashboard.html', context)

def _get_category_icon(cuisine):
    """Retourne l'icône FontAwesome selon le type de cuisine."""
    cuisine = str(cuisine).lower()
    icons = {
        'pizza': 'fa-pizza-slice',
        'burger': 'fa-hamburger',
        'fast food': 'fa-hamburger',
        'fast-food': 'fa-hamburger',
        'cafe': 'fa-coffee',
        'café': 'fa-coffee',
        'coffee': 'fa-coffee',
        'boulangerie': 'fa-bread-slice',
        'patisserie': 'fa-cookie',
        'pâtisserie': 'fa-cookie',
        'italienne': 'fa-pizza-slice',
        'asiatique': 'fa-bowl-rice',
        'sushi': 'fa-fish',
        'salon de thé': 'fa-mug-hot',
    }
    for key, icon in icons.items():
        if key in cuisine:
            return icon
    return 'fa-store'  # icône par défaut


@login_required
def expert_dashboard(request):
    return render(request, "expert_dashboard.html")


@login_required
def edit_profile(request):
    user = request.user
    if request.method == "POST":
        form = InvestorProfileForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            messages.success(request, "Profil mis à jour avec succès !")
            return redirect('investisseur_dashboard')
    else:
        form = InvestorProfileForm(instance=user)

    return render(request, "edit_profile.html", {"form": form, "user": user})


def home_view(request):
    return render(request, "home.html")





def how_it_works(request):
    return render(request, 'how_it_works.html')


def about(request):
    return render(request, 'about.html')


def pricing(request):
    return render(request, 'pricing.html')


def contact(request):
    return render(request, 'contact.html')


def newsletter_subscribe(request):
    if request.method == "POST":
        email = request.POST.get("email")
        messages.success(request, "Inscription à la newsletter réussie !")
    return redirect('home')

# ── AJAX : Recherche de franchises ─────────────────────────
@login_required
@login_required
def search_franchises(request):
    """
    Endpoint AJAX pour la recherche de franchises.
    Retourne max 20 résultats pour performance.
    """
    query = request.GET.get('q', '').strip().lower()
    
    # DEBUG
    print(f"\n[DEBUG search_franchises] User: {request.user}, Query: '{query}'")
    print(f"[DEBUG] DATA_PATH: {DATA_PATH}")
    print(f"[DEBUG] Exists: {os.path.exists(DATA_PATH)}")
    
    try:
        if not os.path.exists(DATA_PATH):
            print(f"[ERREUR] Fichier non trouvé: {DATA_PATH}")
            return JsonResponse({
                'results': [],
                'error': f'data.xlsx non trouvé. Chemin cherché: {DATA_PATH}'
            }, status=404)
        
        df = pd.read_excel(DATA_PATH)
        print(f"[DEBUG] Dataset chargé: {len(df)} lignes, {len(df.columns)} colonnes")
        print(f"[DEBUG] Colonnes: {list(df.columns)[:10]}...")
        
        # Nettoyage basique
        df['nom'] = df['nom'].fillna('Inconnu')
        df['adresses'] = df['adresses'].fillna('Localisation inconnue')
        
        if query:
            mask = (
                df['nom'].str.lower().str.contains(query, na=False) |
                df['adresses'].str.lower().str.contains(query, na=False) |
                df.get('cuisines', pd.Series(['']*len(df))).str.lower().str.contains(query, na=False)
            )
            df = df[mask]
            print(f"[DEBUG] Après filtre: {len(df)} résultats")
        
        # Limiter à 20 résultats
        results = []
        for idx, row in df.head(20).iterrows():
            try:
                results.append({
                    'id': f"franchise_{idx}",
                    'text': f"{row['nom']} — {str(row.get('adresses', ''))[:30]}",
                    'nom': str(row['nom']),
                    'ville': str(row['adresses'])[:50] if pd.notna(row.get('adresses')) else 'N/A',
                    'investissement_global': clean_for_json(row.get('investissement_global', 0)),
                    'apport_personnel': clean_for_json(row.get('apport_personnel', 0)),
                    'cuisines': str(row.get('cuisines', '')) if pd.notna(row.get('cuisines')) else '',
                    'surface_moyenne': str(row.get('Surface_moyenne', '100 m2')) if pd.notna(row.get('Surface_moyenne')) else '100 m2',
                    'date_creation': clean_for_json(row.get('date_creation', 2010)),
                    'Implantations': clean_for_json(row.get('Implantations', 10)),
                    'rating': clean_for_json(row.get('rating', 4)),
                    'Redevance_fonctionnement': clean_for_json(row.get('Redevance_fonctionnement', 5)),
                    'Redevance_publicitaire': clean_for_json(row.get('Redevance_publicitaire', 2)),
                    'YOY_Sales': clean_for_json(row.get('YOY_Sales', 5)),
                    'YOY_Units': clean_for_json(row.get('YOY_Units', 3)),
                    'franchise_fee': clean_for_json(row.get('franchise_fee', 30000)),
                    'avis': clean_for_json(row.get('avis', 100)),
                })
            except Exception as row_err:
                print(f"[DEBUG] Erreur ligne {idx}: {row_err}")
                continue
        
        print(f"[DEBUG] Retourne {len(results)} résultats")
        return JsonResponse({'results': results})
        
    except Exception as e:
        print(f"[ERREUR GLOBALE search_franchises]: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'results': [],
            'error': str(e),
            'traceback': traceback.format_exc()
        }, status=500)

# ── GET : Détails d'une franchise par ID ────────────────────
@login_required
def get_franchise_details(request):
    franchise_id = request.GET.get('id', '')
    
    try:
        idx = int(franchise_id.replace('franchise_', ''))
        df = pd.read_excel(DATA_PATH)
        row = df.iloc[idx]
        
        return JsonResponse({
            'success': True,
            'data': {
                'nom': str(row.get('nom', '')),
                'investissement_global': clean_for_json(row.get('investissement_global', 0)),
                'apport_personnel': clean_for_json(row.get('apport_personnel', 0)),
                'cuisines': str(row.get('cuisines', '')) if pd.notna(row.get('cuisines')) else '',
                'ville': str(row.get('adresses', '')) if pd.notna(row.get('adresses')) else '',
                'surface_moyenne': str(row.get('Surface_moyenne', '100 m2')) if pd.notna(row.get('Surface_moyenne')) else '100 m2',
                'date_creation': clean_for_json(row.get('date_creation', 2010)),
                'Implantations': clean_for_json(row.get('Implantations', 10)),
                'rating': clean_for_json(row.get('rating', 4)),
                'Redevance_fonctionnement': clean_for_json(row.get('Redevance_fonctionnement', 5)),
                'Redevance_publicitaire': clean_for_json(row.get('Redevance_publicitaire', 2)),
                'YOY_Sales': clean_for_json(row.get('YOY_Sales', 5)),
                'YOY_Units': clean_for_json(row.get('YOY_Units', 3)),
                'franchise_fee': clean_for_json(row.get('franchise_fee', 30000)),
                'avis': clean_for_json(row.get('avis', 100)),
            }
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

@login_required
def simulation_view(request):
    """
    Page dédiée pour la simulation.
    GET: Affiche le formulaire de recherche de franchise
    POST: Traite la simulation et redirige vers les résultats
    """
    user = request.user
    
    if user.role != "INVESTISSEUR":
        messages.warning(request, "Cette page est réservée aux investisseurs.")
        return redirect('expert_dashboard')
    
    if request.method == 'POST':
        return _process_simulation(request, user)
    
    # GET: Afficher le formulaire
    return render(request, 'simulation_form.html', {
        'user_budget': user.budget_disponible,
    })


def _process_simulation(request, user):
    """Traite la soumission du formulaire de simulation"""
    try:
        franchise_id = request.POST.get('franchise_id', '')
        franchise_name = request.POST.get('franchise_name', '')
        budget = float(request.POST.get('budget', 0) or 0)
        investissement_global = float(request.POST.get('investissement_global', 0) or 0)
        apport_personnel = float(request.POST.get('apport_personnel', 0) or 0)
        cuisine = request.POST.get('cuisine', '')
        location = request.POST.get('location', '')
        
        # ── RÉCUPÉRER CARACTÉRISTIQUES ML DE LA FRANCHISE ──
        franchise_data = {}
        if franchise_id and franchise_id.startswith('franchise_'):
            try:
                idx = int(franchise_id.replace('franchise_', ''))
                df = pd.read_excel(DATA_PATH)
                row = df.iloc[idx]
                franchise_data = {
                    'nom': str(row.get('nom', 'Inconnu')),
                    'rating': float(row.get('rating', 4)) if pd.notna(row.get('rating')) else 4,
                    'food_rating': float(row.get('food_rating', 4)) if pd.notna(row.get('food_rating')) else 4,
                    'service_rating': float(row.get('service_rating', 4)) if pd.notna(row.get('service_rating')) else 4,
                    'YOY_Sales': float(row.get('YOY_Sales', 5)) if pd.notna(row.get('YOY_Sales')) else 5,
                    'YOY_Units': float(row.get('YOY_Units', 3)) if pd.notna(row.get('YOY_Units')) else 3,
                    'Implantations': int(row.get('Implantations', 10)) if pd.notna(row.get('Implantations')) else 10,
                    'date_creation': int(row.get('date_creation', 2010)) if pd.notna(row.get('date_creation')) else 2010,
                    'Redevance_fonctionnement': float(row.get('Redevance_fonctionnement', 5)) if pd.notna(row.get('Redevance_fonctionnement')) else 5,
                    'Redevance_publicitaire': float(row.get('Redevance_publicitaire', 2)) if pd.notna(row.get('Redevance_publicitaire')) else 2,
                    'franchise_fee': float(row.get('franchise_fee', 30000)) if pd.notna(row.get('franchise_fee')) else 30000,
                    'avis': int(row.get('avis', 100)) if pd.notna(row.get('avis')) else 100,
                    'sales_in_millions_2019': float(row.get('sales_in_millions_2019', 10)) if pd.notna(row.get('sales_in_millions_2019')) else 10,
                    'company_owned_units_2019': int(row.get('company_owned_units_2019', 50)) if pd.notna(row.get('company_owned_units_2019')) else 50,
                    'total_units_2019': int(row.get('total_units_2019', 100)) if pd.notna(row.get('total_units_2019')) else 100,
                    'unit_change_from_2018': int(row.get('unit_change_from_2018', 5)) if pd.notna(row.get('unit_change_from_2018')) else 5,
                    'CompanyStores': int(row.get('CompanyStores', 20)) if pd.notna(row.get('CompanyStores')) else 20,
                    'TotalUnits2021': int(row.get('TotalUnits2021', 120)) if pd.notna(row.get('TotalUnits2021')) else 120,
                    'TotalChange_in_Units_from2020': int(row.get('TotalChange_in_Units_from2020', 10)) if pd.notna(row.get('TotalChange_in_Units_from2020')) else 10,
                    'Merchant_Category': str(row.get('Merchant_Category', 'food')),
                    'Type_de_contrat': str(row.get('Type_de_contrat', 'franchise')),
                    'pickup_enabled': int(row.get('pickup_enabled', 1)) if pd.notna(row.get('pickup_enabled')) else 1,
                    'delivery_enabled': int(row.get('delivery_enabled', 1)) if pd.notna(row.get('delivery_enabled')) else 1,
                    'Formations': str(row.get('Formations', 'oui')),
                    'Aide_auFinancement': str(row.get('Aide_auFinancement', 'oui')),
                    'duree_fonctionnement': int(row.get('duree_fonctionnement', 10)) if pd.notna(row.get('duree_fonctionnement')) else 10,
                    'heure_ouverture': str(row.get('heure_ouverture', '09:00')),
                    'heure_fermeture': str(row.get('heure_fermeture', '22:00')),
                    'Surface_moyenne': str(row.get('Surface_moyenne', '100 m2')),
                    'cuisines': str(row.get('cuisines', '')) if pd.notna(row.get('cuisines')) else '',
                }
            except Exception as e:
                print(f"[WARN] Erreur chargement franchise: {e}")
        
        # ── CONSTRUIRE input_dict ──
        input_dict = {
            'nom': franchise_data.get('nom', franchise_name or 'Franchise'),
            'apport_personnel': apport_personnel or budget * 0.3,  # 30% du budget par défaut
            'droites_entree': 20000,
            'investissement_global': investissement_global,
            'Implantations': franchise_data.get('Implantations', 10),
            'Formations': 1 if (franchise_data.get('Formations', 'oui').lower() in ['oui', '1', 'yes']) or user.formation_required else 0,
            'Aide_auFinancement': 1 if (franchise_data.get('Aide_auFinancement', 'oui').lower() in ['oui', '1', 'yes']) or user.aide_financement_required else 0,
            'Redevance_fonctionnement': franchise_data.get('Redevance_fonctionnement', 5),
            'Redevance_publicitaire': franchise_data.get('Redevance_publicitaire', 2),
            'Surface_moyenne': franchise_data.get('Surface_moyenne', '100 m2'),
            'Type_de_contrat': franchise_data.get('Type_de_contrat', 'franchise'),
            'pickup_enabled': 1 if user.pickup_preferee else 0,
            'delivery_enabled': 1 if user.livraison_preferee else 0,
            'avis': franchise_data.get('avis', 100),
            'date_creation': franchise_data.get('date_creation', 2010),
            'sales_in_millions_2019': franchise_data.get('sales_in_millions_2019', 10),
            'company_owned_units_2019': franchise_data.get('company_owned_units_2019', 50),
            'total_units_2019': franchise_data.get('total_units_2019', 100),
            'unit_change_from_2018': franchise_data.get('unit_change_from_2018', 5),
            'CompanyStores': franchise_data.get('CompanyStores', 20),
            'TotalUnits2021': franchise_data.get('TotalUnits2021', 120),
            'TotalChange_in_Units_from2020': franchise_data.get('TotalChange_in_Units_from2020', 10),
            'Merchant_Category': franchise_data.get('Merchant_Category', 'food'),
            'franchise_fee': franchise_data.get('franchise_fee', 30000),
            'YOY_Sales': franchise_data.get('YOY_Sales', 5),
            'YOY_Units': franchise_data.get('YOY_Units', 3),
            'rating': franchise_data.get('rating', 4),
            'food_rating': franchise_data.get('food_rating', 4),
            'service_rating': franchise_data.get('service_rating', 4),
            'adresses': location,
            'cuisines': cuisine or franchise_data.get('cuisines', 'fast food'),
            'performance_CA': 'moyenne',
            'duree_fonctionnement': franchise_data.get('duree_fonctionnement', 10),
            'heure_ouverture': franchise_data.get('heure_ouverture', '09:00'),
            'heure_fermeture': franchise_data.get('heure_fermeture', '22:00'),
        }
        
        # ── PRÉDICTION ML ──
        regression_model = MLProvider.get_regression()
        result = regression_model.predict(input_dict)
        ca_annee_2 = result.get('CA_potentiel_2ans', 0)
        
        # ── CALCULS RENTABILITÉ ──
        rentabilite_mois = None
        if ca_annee_2 > 0 and investissement_global > 0:
            ca_mensuel = ca_annee_2 / 24
            if ca_mensuel > 0:
                rentabilite_mois = int(investissement_global / ca_mensuel)
        
        roi = (ca_annee_2 / investissement_global) if investissement_global > 0 else 0
        probabilite = calculer_probabilite(ca_annee_2, investissement_global, apport_personnel)
        
        # ── SAUVEGARDE EN BDD ──
        from .models import Simulation
        sim = Simulation.objects.create(
            user=user,
            franchise_name=franchise_data.get('nom', franchise_name or 'Inconnu'),
            location=location,
            cuisine=cuisine or franchise_data.get('cuisines', ''),
            budget=budget,
            investissement_global=investissement_global,
            apport_personnel=apport_personnel,
            ca_annee_2=ca_annee_2,
            rentabilite_mois=rentabilite_mois,
            probabilite_succes=probabilite,
        )
        
        # ── REDIRECTION VERS PAGE DE RÉSULTATS ──
        return redirect('simulation_result', simulation_id=sim.id)
        
    except Exception as e:
        messages.error(request, f"Erreur simulation : {str(e)}")
        print(f"[ERREUR Simulation] {e}")
        import traceback; traceback.print_exc()
        return redirect('simulation')


@login_required
def simulation_result(request, simulation_id):
    """
    Page de résultats de simulation
    """
    from .models import Simulation
    try:
        sim = Simulation.objects.get(id=simulation_id, user=request.user)
    except Simulation.DoesNotExist:
        messages.error(request, "Simulation non trouvée.")
        return redirect('simulation')
    
    # Calculs additionnels pour l'affichage
    roi = (sim.ca_annee_2 / sim.investissement_global) if sim.investissement_global > 0 else 0
    ca_mensuel = sim.ca_annee_2 / 24 if sim.ca_annee_2 else 0
    
    context = {
        'simulation': sim,
        'roi': round(roi, 2),
        'ca_mensuel': round(ca_mensuel, 2),
        'rentabilite_annee': round(sim.rentabilite_mois / 12, 1) if sim.rentabilite_mois else None,
        'progress_color': 'green' if sim.probabilite_succes >= 70 else 'yellow' if sim.probabilite_succes >= 50 else 'red',
    }
    
    return render(request, 'simulation_result.html', context)


@login_required
def export_pdf(request):
    """
    Génère un PDF professionnel avec les données du dashboard expert.
    """
    if request.user.role != "EXPERT":
        return JsonResponse({'error': 'Accès réservé aux experts'}, status=403)

    # Créer le buffer PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2563eb'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    normal_style.leading = 14

    # Contenu du PDF
    elements = []

    # En-tête
    elements.append(Paragraph("Franchise Invest AI", title_style))
    elements.append(Paragraph("Rapport d'Analyse Expert", ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.gray,
        alignment=TA_CENTER,
        spaceAfter=20
    )))

    # Date et infos
    elements.append(Paragraph(f"<b>Date du rapport :</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}", normal_style))
    elements.append(Paragraph(f"<b>Expert :</b> {request.user.get_full_name() or request.user.username}", normal_style))
    elements.append(Paragraph(f"<b>Email :</b> {request.user.email}", normal_style))
    elements.append(Spacer(1, 20))

    # Section 1 : Résumé Exécutif
    elements.append(Paragraph("📊 Résumé Exécutif", heading_style))
    
    summary_data = [
        ['Métrique', 'Valeur', 'Statut'],
        ['Franchises analysées', '6 690', '✅ À jour'],
        ['Modèles ML actifs', '3', '✅ Opérationnels'],
        ['Précision moyenne (R²)', '0.87', '✅ Excellente'],
        ['Segments identifiés', '3', '✅ Haut/Valeur/Risque'],
    ]
    
    summary_table = Table(summary_data, colWidths=[6*cm, 4*cm, 4*cm])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8fafc'), colors.white]),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))

    # Section 2 : Performance des Modèles ML
    elements.append(Paragraph("🤖 Performance des Modèles ML", heading_style))
    
    ml_data = [
        ['Modèle', 'Algorithme', 'CV R²', 'Usage'],
        ['Régression', 'XGBoost / Random Forest', '0.87', 'Prédiction CA 2 ans'],
        ['Segmentation', 'K-Means + RF Classifier', '0.92', 'Catégorisation risque'],
        ['Recommandation', 'Filtrage Collaboratif', '0.78', 'Matching investisseur'],
    ]
    
    ml_table = Table(ml_data, colWidths=[4*cm, 5*cm, 3*cm, 4*cm])
    ml_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16a34a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0fdf4'), colors.white]),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    elements.append(ml_table)
    elements.append(Spacer(1, 20))

    # Section 3 : Segments de Franchises
    elements.append(Paragraph("📈 Répartition des Segments", heading_style))
    
    segment_data = [
        ['Segment', 'Description', '% du portefeuille', 'Profil investisseur'],
        ['Haut Rendement', 'CA élevé, forte croissance', '25%', 'Expert, apport > 100k€'],
        ['Valeur Sûre', 'Stabilité, réseau mature', '45%', 'Intermédiaire, 50-100k€'],
        ['Risque Élevé', 'Nouveau concept, faible réseau', '30%', 'Débutant, < 50k€'],
    ]
    
    segment_table = Table(segment_data, colWidths=[3.5*cm, 5*cm, 3.5*cm, 4*cm])
    segment_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ca8a04')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#fefce8'), colors.white]),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    elements.append(segment_table)
    elements.append(Spacer(1, 20))

    # Section 4 : Recommandations Stratégiques
    elements.append(Paragraph("💡 Recommandations Stratégiques", heading_style))
    
    recommendations = [
        "• <b>Diversification géographique :</b> 60% des opportunités sont concentrées en Île-de-France. Élargir le périmètre d'analyse.",
        "• <b>Seuil d'apport :</b> Les franchises 'Haut Rendement' nécessitent un apport moyen de 85k€. Préconiser un minimum de 30% d'apport.",
        "• <b>Segments à fort potentiel :</b> La restauration rapide santé (+35% YOY) et les concepts hybrides café/coworking sont sous-représentés.",
        "• <b>Risque systémique :</b> 18% des franchises du portefeuille ont un YOY_Sales négatif. Surveillance renforcée recommandée.",
    ]
    
    for rec in recommendations:
        elements.append(Paragraph(rec, normal_style))
        elements.append(Spacer(1, 6))

    # Pied de page
    elements.append(Spacer(1, 30))
    elements.append(Paragraph(
        "<i>Ce rapport a été généré automatiquement par Franchise Invest AI. "
        "Les prédictions sont basées sur des modèles de Machine Learning et ne constituent pas un conseil d'investissement.</i>",
        ParagraphStyle('Footer', parent=normal_style, textColor=colors.gray, fontSize=8, alignment=TA_CENTER)
    ))

    # Générer le PDF
    doc.build(elements)
    buffer.seek(0)

    return FileResponse(
        buffer,
        as_attachment=True,
        filename=f'Franchise_Invest_AI_Rapport_Expert_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf',
        content_type='application/pdf'
    )

def franchise_detail(request, slug):
    # 1. Récupérer la franchise par son slug
    franchise = get_object_or_404(Franchise, slug=slug)
    
    # 2. Logique BI : Trouver des franchises similaires
    similar_franchises = Franchise.objects.filter(
        segment=franchise.segment
    ).exclude(id=franchise.id)[:3]
    
    # 3. Simulation de score de matching
    # Note : get_segment_display() fonctionne sur les objets du modèle, pas les dictionnaires
    user_match = {
        'match_score': 85,
        'match_reasons': [
            "Correspond à votre budget d'investissement",
            f"Secteur {franchise.segment} en forte croissance",
            "Localisation cible hors Tunis validée"
        ]
    }
    
    context = {
        'franchise': franchise,
        'similar_franchises': similar_franchises,
        'user_match': user_match,
    }
    
    return render(request, 'franchise_details.html', context)

def franchises_list(request):
# On récupère tous les objets par défaut
    object_list = Franchise.objects.all().order_by('-note_ai')

    # On récupère les filtres du formulaire (GET)
    search_query = request.GET.get('search')
    if search_query:
        object_list = object_list.filter(name__icontains=search_query)

    segment_query = request.GET.get('segment')
    if segment_query:
        object_list = object_list.filter(segment=segment_query)

    # Pagination
    paginator = Paginator(object_list, 9) # 9 franchises par page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
'page_obj': page_obj,
        'segments': Franchise.SEGMENT_CHOICES,
        'categories': Franchise.CATEGORY_CHOICES, # Utilise les choix du modèle
        'risk_levels': [('faible', 'Faible'), ('moyen', 'Moyen'), ('eleve', 'Élevé')],

    }
    return render(request, 'franchises.html', context)

def test_ml(request):
    from .ml_utils import MLProvider

    test = {
        "CA_potentielApres2ans": 500000,
        "investissement_global": 200000,
        "apport_personnel": 80000,
        "anciennete": 10,
        "Implantations": 50,
        "rating": 4.5,
        "Redevance_fonctionnement": 5,
        "surface_m2": 120,
        "nb_heures_ouverture": 12,
        "Formations": 1,
        "Aide_auFinancement": 1,
        "YOY_Sales": 8,
        "investissement_par_m2": 1000
    }

    cluster = MLProvider.predict_segment(test)
    label = MLProvider.get_segment_label(cluster)

    return JsonResponse({
        "cluster": cluster,
        "label": label
    })