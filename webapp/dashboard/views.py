import json
from datetime import timedelta

from django.shortcuts import render, get_object_or_404, redirect
from django.http import Http404
from django.utils import timezone

from .models import ApiKey, Experiment

VERSION = "0.4.3"


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
    """Landing page — always shows the public homepage."""
    return render(request, "dashboard/login.html", {"version": VERSION})


def dashboard(request):
    """Dashboard for a single API key (= one project)."""
    api_key = _get_api_key(request)
    if api_key is None:
        return redirect("index")

    run_tag = request.GET.get("run")

    experiments = api_key.experiments.all()
    if run_tag:
        experiments = experiments.filter(run_tag=run_tag)

    run_tags = list(
        api_key.experiments.order_by("run_tag").values_list("run_tag", flat=True).distinct()
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
    direction = "minimize" if any(k in metric_name.lower() for k in ("loss", "error", "bpb")) else "maximize"

    # Chart and tree data
    chart_data = _build_chart_data(experiments, direction=direction)
    tree_nodes = _build_tree_data(experiments)

    return render(request, "dashboard/project.html", {
        "api_key": api_key,
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
        "version": VERSION,
    })


def _build_chart_data(experiments, direction="maximize"):
    """Build Chart.js data for the progress chart."""
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

    is_better = (lambda a, b: a > b) if direction == "maximize" else (lambda a, b: a < b)
    kept_ordered = sorted(
        [e for e in all_exps if e.status == "keep" and e.metric_value is not None],
        key=lambda e: e.experiment_number,
    )
    running_best = []
    best_so_far = None
    best_at_num = {}
    for exp in kept_ordered:
        if best_so_far is None or is_better(exp.metric_value, best_so_far):
            best_so_far = exp.metric_value
        best_at_num[exp.experiment_number] = best_so_far

    if best_at_num:
        sorted_nums = sorted(best_at_num.keys())
        for i, num in enumerate(sorted_nums):
            val = best_at_num[num]
            if i > 0:
                prev_val = best_at_num[sorted_nums[i - 1]]
                running_best.append({"x": num, "y": prev_val})
            running_best.append({"x": num, "y": val})
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
    """Build branch tree data for visualization."""
    seen_nums = {}
    for exp in experiments.order_by("id"):
        seen_nums[exp.experiment_number] = exp
    all_exps = sorted(seen_nums.values(), key=lambda e: e.experiment_number)
    if not all_exps:
        return []

    keeps = [e for e in all_exps if e.status == "keep"]
    keep_ids = set(e.id for e in keeps)
    last_keep_id = None
    parent_map = {}

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


def experiment_detail(request, experiment_id):
    """Single experiment detail with diff."""
    api_key = _get_api_key(request)
    if api_key is None:
        return redirect("index")

    experiment = get_object_or_404(Experiment, id=experiment_id, api_key=api_key)

    return render(request, "dashboard/experiment.html", {
        "api_key": api_key,
        "experiment": experiment,
        "version": VERSION,
    })


# --- HTMX partials ---

def experiments_table(request):
    """HTMX partial: refreshable experiments table."""
    api_key = _get_api_key(request)
    if api_key is None:
        raise Http404

    run_tag = request.GET.get("run")

    experiments = api_key.experiments.all()
    if run_tag:
        experiments = experiments.filter(run_tag=run_tag)

    return render(request, "dashboard/partials/experiments_table.html", {
        "api_key": api_key,
        "experiments": experiments,
    })


def progress_bar(request):
    """HTMX partial: progress bar for running experiment."""
    api_key = _get_api_key(request)
    if api_key is None:
        raise Http404

    run_tag = request.GET.get("run")
    experiments = api_key.experiments.all()
    if run_tag:
        experiments = experiments.filter(run_tag=run_tag)

    running_exp = experiments.filter(status="running").order_by("-id").first()

    return render(request, "dashboard/partials/progress_bar.html", {
        "running_exp": running_exp,
    })
