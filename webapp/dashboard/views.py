import json

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


def _build_chart_data(experiments):
    """Build Chart.js data for the progress chart (like Karpathy's progress.png)."""
    all_exps = list(experiments.exclude(status="crash").order_by("id"))

    # All experiment points (scatter)
    keep_points = []
    discard_points = []
    for exp in all_exps:
        if exp.metric_value is None:
            continue
        point = {
            "x": exp.experiment_number,
            "y": exp.metric_value,
            "description": exp.description[:50],
        }
        if exp.status == "keep":
            keep_points.append(point)
        else:
            discard_points.append(point)

    # Running best line (step through keeps in order)
    kept = [e for e in all_exps if e.status == "keep" and e.metric_value is not None]
    running_best = []
    for exp in kept:
        running_best.append({
            "x": exp.experiment_number,
            "y": exp.metric_value,
        })

    return {
        "keep_points": keep_points,
        "discard_points": discard_points,
        "running_best": running_best,
    }


def _build_tree_data(experiments):
    """Build branch tree data for visualization.

    The tree shows:
    - Keep experiments as the main trunk (green)
    - Discard/crash experiments as branches off the trunk (gray/red)
    """
    all_exps = list(experiments.order_by("id"))
    if not all_exps:
        return []

    # Build parent->children mapping based on parent_commit_hash
    by_commit = {e.commit_hash: e for e in all_exps if e.commit_hash}
    by_parent = {}
    for e in all_exps:
        parent = e.parent_commit_hash
        if parent not in by_parent:
            by_parent[parent] = []
        by_parent[parent].append(e)

    # Build trunk: sequence of keep experiments
    keeps = [e for e in all_exps if e.status == "keep"]
    keep_commits = {e.commit_hash for e in keeps}

    # Build tree nodes
    nodes = []
    for i, exp in enumerate(all_exps):
        # Determine if this is on the trunk (keep chain)
        is_trunk = exp.status == "keep"
        # Find branches: discards/crashes whose parent is a keep
        parent_is_keep = exp.parent_commit_hash in keep_commits

        nodes.append({
            "id": exp.id,
            "number": exp.experiment_number,
            "commit": exp.commit_hash[:7] if exp.commit_hash else "---",
            "parent_commit": exp.parent_commit_hash[:7] if exp.parent_commit_hash else "",
            "status": exp.status,
            "metric": exp.metric_value,
            "description": exp.description[:40],
            "is_trunk": is_trunk,
        })

    return nodes


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
    best = experiments.filter(status="keep", metric_value__isnull=False).order_by("-id").first()

    # Chart and tree data
    chart_data = _build_chart_data(experiments)
    tree_nodes = _build_tree_data(experiments)

    # Metric name for labels
    first_with_metric = experiments.filter(metric_name__gt="").first()
    metric_name = first_with_metric.metric_name if first_with_metric else "metric"

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
        "chart_data_json": json.dumps(chart_data),
        "tree_nodes_json": json.dumps(tree_nodes),
        "metric_name": metric_name,
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
