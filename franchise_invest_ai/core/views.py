from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from .forms import UserSignupForm


# VUE D'INSCRIPTION
def signup_view(request):
    if request.method == 'POST':
        form = UserSignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user) # Connecter automatiquement après inscription
            return redirect_by_role(user)
    else:
        form = UserSignupForm()
    return render(request, 'signup.html', {'form': form})

# VUE DE CONNEXION (Mise à jour)
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

# Fonction utilitaire pour la redirection
def redirect_by_role(user):
    if user.role == "INVESTISSEUR":
        return redirect('investisseur_dashboard')
    elif user.role == "EXPERT":
        return redirect('expert_dashboard')
    return redirect('login')

@login_required
def investisseur_dashboard(request):
    # Logique ML à intégrer ici (XGBoost / Recommandation)
    return render(request, "investisseur_dashboard.html")

@login_required
def expert_dashboard(request):
    # Logique Power BI à intégrer ici
    return render(request, "expert_dashboard.html")

def logout_user(request):
    logout(request)
    return redirect('login')