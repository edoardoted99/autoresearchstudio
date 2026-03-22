from django.urls import path
from . import views, api

urlpatterns = [
    # Dashboard
    path("", views.index, name="index"),
    path("logout/", views.logout, name="logout"),
    path("project/<int:project_id>/", views.project_detail, name="project_detail"),
    path("experiment/<int:experiment_id>/", views.experiment_detail, name="experiment_detail"),

    # HTMX partials
    path("htmx/experiments/<int:project_id>/", views.experiments_table, name="experiments_table"),

    # API
    path("api/experiments/", api.experiment_create, name="api_experiment_create"),
    path("api/keys/", api.key_create, name="api_key_create"),
]
