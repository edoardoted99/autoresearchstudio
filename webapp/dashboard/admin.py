from django.contrib import admin
from .models import ApiKey, Project, Experiment


@admin.register(ApiKey)
class ApiKeyAdmin(admin.ModelAdmin):
    list_display = ("name", "key", "is_active", "created_at")
    readonly_fields = ("key",)


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ("name", "api_key", "created_at")


@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    list_display = (
        "experiment_number", "project", "run_tag", "status",
        "metric_value", "description", "created_at",
    )
    list_filter = ("status", "project", "run_tag")
