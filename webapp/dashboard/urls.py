from django.urls import path
from . import views, api

urlpatterns = [
    # Dashboard
    path("", views.index, name="index"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("logout/", views.logout, name="logout"),
    path("experiment/<int:experiment_id>/", views.experiment_detail, name="experiment_detail"),

    # HTMX partials
    path("htmx/experiments/", views.experiments_table, name="experiments_table"),

    # API
    path("api/experiments/", api.experiment_create, name="api_experiment_create"),
    path("api/keys/", api.key_create, name="api_key_create"),
]
