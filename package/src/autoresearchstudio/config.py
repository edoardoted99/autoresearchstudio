"""Project configuration: YAML loading, validation, defaults."""

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class SecondaryMetric:
    name: str
    pattern: str


@dataclass
class MetricConfig:
    name: str
    pattern: str
    direction: str = "minimize"  # "minimize" or "maximize"
    secondary: list[SecondaryMetric] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    run_command: str
    timeout: Optional[int] = 60
    setup_command: Optional[str] = None
    log_file: str = "run.log"
    constraints: Optional[str] = None


@dataclass
class JudgeConfig:
    threshold: float = 0.0
    simplicity_note: Optional[str] = None


@dataclass
class ApiConfig:
    key: Optional[str] = None
    endpoint: str = "https://autoresearch.studio/api"


@dataclass
class FilesConfig:
    editable: list[str] = field(default_factory=list)
    readonly: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)


@dataclass
class ProjectConfig:
    name: str = "my-project"
    description: str = ""
    goal: str = ""


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    files: FilesConfig = field(default_factory=FilesConfig)
    experiment: ExperimentConfig = field(default_factory=lambda: ExperimentConfig(run_command="python train.py"))
    metric: MetricConfig = field(default_factory=lambda: MetricConfig(name="loss", pattern=r"^loss:\s+([\d.]+)"))
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    api: ApiConfig = field(default_factory=ApiConfig)


CONFIG_FILENAME = "autoresearch.yaml"


def load_config(path: str = CONFIG_FILENAME) -> Config:
    """Load config from YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    project = ProjectConfig(**raw.get("project", {}))

    files_raw = raw.get("files", {})
    files = FilesConfig(
        editable=files_raw.get("editable", []),
        readonly=files_raw.get("readonly", []),
        context=files_raw.get("context", []),
    )

    exp_raw = raw.get("experiment", {})
    experiment = ExperimentConfig(
        run_command=exp_raw.get("run_command", "python train.py"),
        timeout=exp_raw.get("timeout", 60),
        setup_command=exp_raw.get("setup_command"),
        log_file=exp_raw.get("log_file", "run.log"),
        constraints=exp_raw.get("constraints"),
    )

    met_raw = raw.get("metric", {})
    secondary = [
        SecondaryMetric(name=s["name"], pattern=s["pattern"])
        for s in met_raw.get("secondary", [])
    ]
    metric = MetricConfig(
        name=met_raw.get("name", "loss"),
        pattern=met_raw.get("pattern", r"^loss:\s+([\d.]+)"),
        direction=met_raw.get("direction", "minimize"),
        secondary=secondary,
    )

    judge_raw = raw.get("judge", {})
    judge = JudgeConfig(
        threshold=judge_raw.get("threshold", 0.0),
        simplicity_note=judge_raw.get("simplicity_note"),
    )

    api_raw = raw.get("api", {})
    api = ApiConfig(
        key=api_raw.get("key") or os.environ.get("ARS_API_KEY"),
        endpoint=api_raw.get("endpoint", "https://autoresearch.studio/api"),
    )

    return Config(
        project=project,
        files=files,
        experiment=experiment,
        metric=metric,
        judge=judge,
        api=api,
    )


def validate_config(config: Config) -> list[str]:
    """Validate config, return list of error messages (empty = valid)."""
    errors = []

    if not config.files.editable:
        errors.append("files.editable must list at least one file")

    if not config.experiment.run_command:
        errors.append("experiment.run_command is required")

    if config.metric.direction not in ("minimize", "maximize"):
        errors.append(f"metric.direction must be 'minimize' or 'maximize', got '{config.metric.direction}'")

    if not config.metric.pattern:
        errors.append("metric.pattern is required")

    for f in config.files.editable:
        if not os.path.exists(f):
            errors.append(f"editable file not found: {f}")

    for f in config.files.readonly:
        if not os.path.exists(f):
            errors.append(f"readonly file not found: {f}")

    return errors


def config_to_yaml(config: Config) -> str:
    """Serialize config to YAML string."""
    data = {
        "project": {
            "name": config.project.name,
            "description": config.project.description,
            "goal": config.project.goal,
        },
        "files": {
            "editable": config.files.editable,
            "readonly": config.files.readonly,
        },
        "experiment": {
            "run_command": config.experiment.run_command,
            "timeout": config.experiment.timeout,
        },
        "metric": {
            "name": config.metric.name,
            "pattern": config.metric.pattern,
            "direction": config.metric.direction,
        },
        "judge": {
            "threshold": config.judge.threshold,
        },
        "api": {
            "key": config.api.key,
            "endpoint": config.api.endpoint,
        },
    }

    if config.experiment.setup_command:
        data["experiment"]["setup_command"] = config.experiment.setup_command
    if config.experiment.log_file != "run.log":
        data["experiment"]["log_file"] = config.experiment.log_file
    if config.experiment.constraints:
        data["experiment"]["constraints"] = config.experiment.constraints
    if config.files.context:
        data["files"]["context"] = config.files.context
    if config.metric.secondary:
        data["metric"]["secondary"] = [
            {"name": s.name, "pattern": s.pattern} for s in config.metric.secondary
        ]
    if config.judge.simplicity_note:
        data["judge"]["simplicity_note"] = config.judge.simplicity_note

    return yaml.dump(data, default_flow_style=False, sort_keys=False)
