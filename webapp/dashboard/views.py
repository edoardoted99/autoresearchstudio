import json
from datetime import timedelta

from django.shortcuts import render, get_object_or_404, redirect
from django.http import Http404
from django.utils import timezone

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


def logout(request):
    """Clear session and redirect to login."""
    request.session.flush()
    return redirect("index")


def index(request):
    """Landing page / project list."""
    api_key = _get_api_key(request)
    if api_key is None:
        return render(request, "dashboard/login.html")

    projects = Project.objects.filter(api_key=api_key).order_by("-created_at")

    # If only one project, go straight to it
    if projects.count() == 1:
        return redirect("project_detail", project_id=projects.first().id)

    project_data = []
    for p in projects:
        experiments = p.experiments.all()
        total = experiments.count()
        keeps = experiments.filter(status="keep").count()
        discards = experiments.filter(status="discard").count()
        crashes = experiments.filter(status="crash").count()
        best = experiments.filter(status="keep", metric_value__isnull=False).order_by("-metric_value").first()
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


def _build_chart_data(experiments, direction="maximize"):
    """Build Chart.js data for the progress chart."""
    # Deduplicate: keep only the latest record per experiment_number
    seen_nums = {}
    for exp in experiments.order_by("id"):
        seen_nums[exp.experiment_number] = exp
    all_exps = sorted(seen_nums.values(), key=lambda e: e.experiment_number)

    keep_points = []
    discard_points = []
    for exp in all_exps:
        if exp.metric_value is None:
            continue
        point = {
            "x": exp.experiment_number,
            "y": exp.metric_value,
            "description": exp.description[:50] if exp.description else "",
        }
        if exp.status == "keep":
            keep_points.append(point)
        else:
            discard_points.append(point)

    # Running best line: step through keeps ordered by experiment_number,
    # tracking the best value seen so far.
    is_better = (lambda a, b: a > b) if direction == "maximize" else (lambda a, b: a < b)
    kept_ordered = sorted(
        [e for e in all_exps if e.status == "keep" and e.metric_value is not None],
        key=lambda e: e.experiment_number,
    )
    running_best = []
    best_so_far = None
    best_at_num = {}  # experiment_number -> best value at that point
    for exp in kept_ordered:
        if best_so_far is None or is_better(exp.metric_value, best_so_far):
            best_so_far = exp.metric_value
        best_at_num[exp.experiment_number] = best_so_far

    # Build running best as an explicit staircase line (no Chart.js stepped).
    # For each keep, the line goes: horizontal at old best → vertical up to
    # new best at the keep's x. This ensures every keep point sits ON the line.
    if best_at_num:
        sorted_nums = sorted(best_at_num.keys())
        for i, num in enumerate(sorted_nums):
            val = best_at_num[num]
            if i > 0:
                prev_val = best_at_num[sorted_nums[i - 1]]
                # Horizontal segment at previous best up to this x
                running_best.append({"x": num, "y": prev_val})
            # Point at the (possibly new) best
            running_best.append({"x": num, "y": val})
        # Extend horizontally to the last experiment number
        if all_exps:
            last_num = max(e.experiment_number for e in all_exps)
            if running_best[-1]["x"] < last_num:
                running_best.append({"x": last_num, "y": running_best[-1]["y"]})

    return {
        "keep_points": keep_points,
        "discard_points": discard_points,
        "running_best": running_best,
        "total": len(all_exps),
        "keeps": len(kept_ordered),
    }


def _build_tree_data(experiments):
    """Build branch tree data for visualization.

    The tree shows:
    - Keep experiments as the main trunk (green, horizontal)
    - Discard/crash/running experiments as branches off the most recent keep
    """
    # Deduplicate: keep only the latest record per experiment_number
    seen_nums = {}
    for exp in experiments.order_by("id"):
        seen_nums[exp.experiment_number] = exp
    all_exps = sorted(seen_nums.values(), key=lambda e: e.experiment_number)
    if not all_exps:
        return []

    # Build trunk: ordered sequence of keeps
    keeps = [e for e in all_exps if e.status == "keep"]

    # Assign each non-keep experiment to the most recent keep before it.
    # This is more reliable than using parent_commit_hash, which can be
    # wrong when experiments share the same commit hash.
    keep_ids = set(e.id for e in keeps)
    last_keep_id = None  # track most recent keep as we iterate
    parent_map = {}  # exp.id -> parent keep id

    for exp in all_exps:
        if exp.id in keep_ids:
            last_keep_id = exp.id
        else:
            parent_map[exp.id] = last_keep_id

    nodes = []
    for exp in all_exps:
        is_trunk = exp.id in keep_ids
        parent_id = parent_map.get(exp.id)

        nodes.append({
            "id": exp.id,
            "number": exp.experiment_number,
            "commit": exp.commit_hash[:7] if exp.commit_hash else "---",
            "parent_id": parent_id,
            "status": exp.status,
            "metric": exp.metric_value,
            "description": exp.description[:40] if exp.description else "",
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
    best = experiments.filter(status="keep", metric_value__isnull=False).order_by("-metric_value").first()

    # Auto-close stale "running" experiments (timeout + 2min grace period)
    stale_cutoff = timezone.now() - timedelta(minutes=2)
    for exp in experiments.filter(status="running", started_at__isnull=False):
        deadline = exp.started_at + timedelta(seconds=(exp.timeout_seconds or 600))
        if deadline < stale_cutoff:
            exp.status = "crash"
            exp.save(update_fields=["status"])

    # Running experiment (for progress bar)
    running_exp = experiments.filter(status="running").order_by("-id").first()

    # Detect metric direction from first experiment's metric_name
    first_with_metric = experiments.filter(metric_name__gt="").first()
    metric_name = first_with_metric.metric_name if first_with_metric else "metric"
    # Infer direction: if metric name contains "loss" or "error", minimize
    direction = "minimize" if any(k in metric_name.lower() for k in ("loss", "error", "bpb")) else "maximize"

    # Chart and tree data
    chart_data = _build_chart_data(experiments, direction=direction)
    tree_nodes = _build_tree_data(experiments)

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
        "running_exp": running_exp,
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
