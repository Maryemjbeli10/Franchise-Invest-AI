from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    class Role(models.TextChoices):
        INVESTISSEUR = "INVESTISSEUR", "Investisseur"
        EXPERT = "EXPERT", "Banque/Consultant"

    role = models.CharField(
        max_length=20, 
        choices=Role.choices, 
        default=Role.INVESTISSEUR
    )
    
    # Champs spécifiques
    budget_disponible = models.FloatField(default=0.0, null=True, blank=True)
    ville_interet = models.CharField(max_length=100, blank=True)

    def __str__(self):
        return f"{self.username} - {self.role}"