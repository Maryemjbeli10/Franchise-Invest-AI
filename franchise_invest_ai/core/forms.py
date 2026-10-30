from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import User

class UserSignupForm(UserCreationForm):
    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("username", "email", "role") # Le champ role vient de votre modèle User

class InvestorProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = [
            "budget_disponible",
            "ville_interet",
            "niveau_experience",
            "apport_personnel",
            "cuisine_preferee",
            "livraison_preferee",
            "pickup_preferee",
            "formation_required",
            "aide_financement_required",
            "roi_min",
            "performance_min",
        ]
