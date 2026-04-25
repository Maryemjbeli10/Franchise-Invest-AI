from django.contrib import admin
from django.urls import path
from core import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('investisseur/', views.investisseur_dashboard, name='investisseur_dashboard'),
    path('expert/', views.expert_dashboard, name='expert_dashboard'),
    path('logout/', views.logout_user, name='logout'),
]