from django.contrib import admin
from django.urls import path
from core import views

urlpatterns = [
    path('admin/', admin.site.urls),

    # Auth
    path('', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('register/', views.signup_view, name='register'),  # AJOUT IMPORTANT
    path('logout/', views.logout_user, name='logout'),

    # Home
    path('home/', views.home_view, name='home'),

    # Dashboards
    path('investisseur/', views.investisseur_dashboard, name='investisseur_dashboard'),
    path('expert/', views.expert_dashboard, name='expert_dashboard'),
    path('dashboard/', views.investisseur_dashboard, name='dashboard'),  # AJOUT IMPORTANT

    # Profile
    path('edit-profile/', views.edit_profile, name='edit_profile'),

    # ── ENDPOINTS AJAX (NOUVEAUX) ──
    path('api/franchises/search/', views.search_franchises, name='search_franchises'),
    path('api/franchises/details/', views.get_franchise_details, name='get_franchise_details'),

    # Pages site
    path('how-it-works/', views.how_it_works, name='how_it_works'),
    path('about/', views.about, name='about'),
    path('pricing/', views.pricing, name='pricing'),
    path('contact/', views.contact, name='contact'),
    path('newsletter/', views.newsletter_subscribe, name='newsletter_subscribe'),
    path('simulation/', views.simulation_view, name='simulation'),
    path('simulation/result/<int:simulation_id>/', views.simulation_result, name='simulation_result'),
    path('api/export-pdf/', views.export_pdf, name='export_pdf'),

    path('franchise/<slug:slug>/', views.franchise_detail, name='franchise_detail'),
    path('nos-franchises/', views.franchises_list, name='franchises_list'),

]
