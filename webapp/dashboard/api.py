"""API endpoints for the CLI to POST experiment data."""

import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.utils.dateparse import parse_datetime

from .models import ApiKey, Project, Experiment


def _authenticate(request):
    """Extract and validate API key from Authorization header."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        key = auth[7:]
    else:
        return None
    try:
        return ApiKey.objects.get(key=key, is_active=True)
    except ApiKey.DoesNotExist:
        return None


@csrf_exempt
@require_POST
def experiment_create(request):
    """Receive experiment data from the CLI.

    POST /api/experiments/
    Authorization: Bearer ars_xxxx
    Body: JSON with experiment fields
    """
    api_key = _authenticate(request)
    if api_key is None:
        return JsonResponse({"error": "Invalid or missing API key"}, status=401)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Get or create project
    project_name = data.get("project_name") or data.get("run_tag", "default")
    project, _ = Project.objects.get_or_create(
        api_key=api_key,
        name=project_name,
    )

    # Parse optional datetimes
    started_at = None
    if data.get("started_at"):
        started_at = parse_datetime(data["started_at"])
    finished_at = None
    if data.get("finished_at"):
        finished_at = parse_datetime(data["finished_at"])

    # Upsert: if experiment with same run_tag + experiment_number exists, update it
    exp, created = Experiment.objects.update_or_create(
        project=project,
        run_tag=data.get("run_tag", ""),
        experiment_number=data.get("experiment_number", 0),
        defaults={
            "commit_hash": data.get("commit_hash", ""),
            "parent_commit_hash": data.get("parent_commit_hash", ""),
            "metric_name": data.get("metric_name", ""),
            "metric_value": data.get("metric_value"),
            "secondary_metrics": data.get("secondary_metrics", {}),
            "status": data.get("status", "running"),
            "description": data.get("description", ""),
            "diff": data.get("diff", ""),
            "timeout_seconds": data.get("timeout_seconds"),
            "duration_seconds": data.get("duration_seconds"),
            "started_at": started_at,
            "finished_at": finished_at,
        },
    )

    return JsonResponse({
        "id": exp.id,
        "created": created,
        "project": project.name,
        "experiment_number": exp.experiment_number,
        "status": exp.status,
    }, status=201 if created else 200)


@csrf_exempt
@require_POST
def key_create(request):
    """Generate a new API key.

    POST /api/keys/
    Body: {"name": "my-project"}  (optional)
    """
    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        data = {}

    name = data.get("name", "cli")
    api_key = ApiKey.objects.create(name=name)

    return JsonResponse({
        "key": api_key.key,
    }, status=201)
