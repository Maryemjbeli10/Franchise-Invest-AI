import pandas as pd
from core.models import Franchise
from django.utils.text import slugify
from core.ml_utils import MLProvider



def sync_segments():
    franchises = Franchise.objects.all()
    print(f"Mise à jour de la segmentation pour {franchises.count()} franchises...")
    
    for f in franchises:
        try:
            label = MLProvider.predict_franchise_segment(f)
            f.segment = label
            f.save()
        except Exception as e:
            print(f"Erreur pour {f.nom}: {e}")
    
    print("Segmentation terminée ! ✔")

def run():
    # 1. Chargement
    try:
        df = pd.read_excel("../data.xlsx").fillna(0)
    except:
        df = pd.read_excel("data.xlsx").fillna(0)

    def clean_float(value):
        if isinstance(value, str):
            value = value.replace('%', '').strip()
        try:
            return float(value)
        except:
            return 0.0

    print(f"Début de l'importation : {len(df)} lignes...")

    from django.db import transaction

    with transaction.atomic():
        for index, row in df.iterrows():
            nom_val = str(row.get('nom', f"Franchise {index}"))
            
            # On génère un slug unique en ajoutant l'index pour éviter les doublons
            unique_slug = f"{slugify(nom_val)}-{index}"
            
            # Utilisation de update_or_create pour être plus flexible
            Franchise.objects.update_or_create(
                nom=nom_val,
                # On utilise l'index dans le slug pour garantir l'unicité réclamée par ton models.py
                slug=unique_slug, 
                defaults={
                    'apport_personnel': clean_float(row.get('apport_personnel', 0)),
                    'investissement_global': clean_float(row.get('investissement_global', 0)),
                    'ca_potentiel': clean_float(row.get('CA_potentielApres2ans', 0)),
                    'implantations': clean_float(row.get('Implantations', 0)),
                    'note_ai': int(clean_float(row.get('rating', 0)) * 20),
                    'segment': 'restauration',
                    'taux_reussite': clean_float(row.get('YOY_Sales', 0)),
                    'description': f"Cuisine: {row.get('cuisines', 'N/A')}. Localisation: {row.get('adresses', 'Tunisie')}.",
                }
            )
            
            if index % 500 == 0:
                print(f"Progression : {index}/{len(df)}...")

    print(f"TERMINÉ ! Total : {Franchise.objects.count()} franchises en base.")

if __name__ == "__main__":
    run()