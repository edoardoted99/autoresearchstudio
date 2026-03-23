from django.contrib import admin
from .models import ApiKey, Experiment


@admin.register(ApiKey)
class ApiKeyAdmin(admin.ModelAdmin):
    list_display = ("name", "key", "is_active", "created_at")
    readonly_fields = ("key",)


@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    list_display = (
        "experiment_number", "api_key", "run_tag", "status",
        "metric_value", "description", "created_at",
    )
    list_filter = ("status", "api_key", "run_tag")
