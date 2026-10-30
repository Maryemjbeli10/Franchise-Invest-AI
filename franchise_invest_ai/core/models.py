from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils.text import slugify

class User(AbstractUser):

    class Role(models.TextChoices):
        INVESTISSEUR = "INVESTISSEUR", "Investisseur"
        EXPERT = "EXPERT", "Banque/Consultant"

    role = models.CharField(max_length=20, choices=Role.choices, default=Role.INVESTISSEUR)

    # ───────────────
    # 💰 PROFIL INVESTISSEUR
    # ───────────────
    budget_disponible = models.FloatField(default=0.0, null=True, blank=True)
    ville_interet = models.CharField(max_length=100, blank=True)

    niveau_experience = models.CharField(
        max_length=50,
        choices=[
            ("DEBUTANT", "Débutant"),
            ("INTERMEDIAIRE", "Intermédiaire"),
            ("EXPERT", "Expert"),
        ],
        default="DEBUTANT"
    )

    apport_personnel = models.FloatField(default=0.0, null=True, blank=True)

    # ───────────────
    # 🍽️ PRÉFÉRENCES IA (TRÈS IMPORTANT POUR TON RECO SYSTEM)
    # ───────────────

    cuisine_preferee = models.CharField(max_length=100, blank=True)

    livraison_preferee = models.BooleanField(default=False)
    pickup_preferee = models.BooleanField(default=False)

    formation_required = models.BooleanField(default=False)
    aide_financement_required = models.BooleanField(default=False)

    roi_min = models.FloatField(default=1.2)

    performance_min = models.IntegerField(default=1)

    def __str__(self):
        return f"{self.username} - {self.role}"
    

class Simulation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='simulations')
    franchise_name = models.CharField(max_length=200)
    location = models.CharField(max_length=200, blank=True)
    cuisine = models.CharField(max_length=100, blank=True)
    budget = models.FloatField()
    investissement_global = models.FloatField()
    apport_personnel = models.FloatField(null=True, blank=True)
    ca_annee_2 = models.FloatField(null=True, blank=True)
    rentabilite_mois = models.IntegerField(null=True, blank=True)
    probabilite_succes = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

class Franchise(models.Model):
    CATEGORY_CHOICES = [
        ('restauration', 'Restauration'),
        ('services', 'Services'),
        ('sante', 'Santé'),
        # Ajoutez les autres catégories ici...
    ]
    
    SEGMENT_CHOICES = [
        ('haut_rendement', 'Haut Rendement'),
        ('valeur_sure', 'Valeur Sûre'),
        ('risque_eleve', 'Risque Élevé'),
    ]

    nom = models.CharField(max_length=255)
    slug = models.SlugField(unique=True, blank=True)
    apport_personnel = models.FloatField(default=0)
    
    investissement_global = models.FloatField(default=0)
    ca_potentiel = models.FloatField(default=0)  # CA_potentielApres2ans
    implantations = models.FloatField(default=0)
    note_ai = models.IntegerField(default=0) # Score calculé par ai.py
    segment = models.CharField(max_length=100, choices=SEGMENT_CHOICES, blank=True, null=True)
    taux_reussite = models.FloatField(default=0)
    description = models.TextField(blank=True)
    RISK_CHOICES = [('faible', 'Faible'), ('moyen', 'Moyen'), ('eleve', 'Élevé')]
    risk_level = models.CharField(max_length=20, choices=RISK_CHOICES, default='moyen')
    logo = models.ImageField(upload_to='franchises/', null=True, blank=True)

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.nom)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.nom