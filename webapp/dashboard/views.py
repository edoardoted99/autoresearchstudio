from django.shortcuts import render, get_object_or_404
from django.http import Http404

from .models import ApiKey, Project, Experiment


def _get_api_key(request):
    """Get API key from session or query param."""
    key = request.GET.get("key") or request.session.get("api_key")
    if not key:
        return None
    try:
        api_key = ApiKey.objects.get(key=key, is_active=True)
        request.session["api_key"] = key
        return api_key
    except ApiKey.DoesNotExist:
        return None


def index(request):
    """Landing page / project list."""
    api_key = _get_api_key(request)
    if api_key is None:
        return render(request, "dashboard/login.html")

    projects = Project.objects.filter(api_key=api_key).order_by("-created_at")

    project_data = []
    for p in projects:
        experiments = p.experiments.all()
        total = experiments.count()
        keeps = experiments.filter(status="keep").count()
        discards = experiments.filter(status="discard").count()
        crashes = experiments.filter(status="crash").count()
        best = experiments.filter(status="keep", metric_value__isnull=False).order_by("-id").first()
        latest = experiments.order_by("-id").first()
        run_tags = list(experiments.values_list("run_tag", flat=True).distinct())

        project_data.append({
            "project": p,
            "total": total,
            "keeps": keeps,
            "discards": discards,
            "crashes": crashes,
            "best": best,
            "latest": latest,
            "run_tags": run_tags,
        })

    return render(request, "dashboard/index.html", {
        "api_key": api_key,
        "projects": project_data,
    })


def project_detail(request, project_id):
    """Project detail: list of runs and experiments."""
    api_key = _get_api_key(request)
    if api_key is None:
        return render(request, "dashboard/login.html")

    project = get_object_or_404(Project, id=project_id, api_key=api_key)
    run_tag = request.GET.get("run")

    experiments = project.experiments.all()
    if run_tag:
        experiments = experiments.filter(run_tag=run_tag)

    run_tags = list(
        project.experiments.values_list("run_tag", flat=True).distinct().order_by("run_tag")
    )

    # Stats
    total = experiments.count()
    keeps = experiments.filter(status="keep").count()
    discards = experiments.filter(status="discard").count()
    crashes = experiments.filter(status="crash").count()
    # Latest keep is always the best (judge guarantees this)
    best = experiments.filter(status="keep", metric_value__isnull=False).order_by("-id").first()

    # Running best series for chart
    kept_experiments = list(experiments.filter(status="keep", metric_value__isnull=False).order_by("id"))
    running_best = []
    best_so_far = None
    for exp in kept_experiments:
        best_so_far = exp.metric_value
        running_best.append({"number": exp.experiment_number, "value": best_so_far})

    return render(request, "dashboard/project.html", {
        "api_key": api_key,
        "project": project,
        "experiments": experiments,
        "run_tags": run_tags,
        "current_run": run_tag,
        "total": total,
        "keeps": keeps,
        "discards": discards,
        "crashes": crashes,
        "best": best,
        "running_best": running_best,
    })


def experiment_detail(request, experiment_id):
    """Single experiment detail with diff."""
    api_key = _get_api_key(request)
    if api_key is None:
        return render(request, "dashboard/login.html")

    experiment = get_object_or_404(Experiment, id=experiment_id, project__api_key=api_key)

    return render(request, "dashboard/experiment.html", {
        "api_key": api_key,
        "experiment": experiment,
        "project": experiment.project,
    })


# --- HTMX partials ---

def experiments_table(request, project_id):
    """HTMX partial: refreshable experiments table."""
    api_key = _get_api_key(request)
    if api_key is None:
        raise Http404

    project = get_object_or_404(Project, id=project_id, api_key=api_key)
    run_tag = request.GET.get("run")

    experiments = project.experiments.all()
    if run_tag:
        experiments = experiments.filter(run_tag=run_tag)

    return render(request, "dashboard/partials/experiments_table.html", {
        "experiments": experiments,
        "project": project,
    })
