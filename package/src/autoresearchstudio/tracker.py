"""Experiment tracking: SQLite local storage + optional API sync."""

import json
import sqlite3
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import requests

from .config import Config


@dataclass
class Experiment:
    id: Optional[int] = None
    run_tag: str = ""
    experiment_number: int = 0
    commit_hash: str = ""
    parent_commit_hash: str = ""
    metric_name: str = ""
    metric_value: Optional[float] = None
    secondary_metrics: dict = field(default_factory=dict)
    status: str = "running"  # running, keep, discard, crash
    description: str = ""
    diff: str = ""
    stdout_tail: str = ""
    duration_seconds: Optional[float] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    synced_at: Optional[str] = None
    created_at: str = ""


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_tag TEXT NOT NULL,
    experiment_number INTEGER NOT NULL,
    commit_hash TEXT DEFAULT '',
    parent_commit_hash TEXT DEFAULT '',
    metric_name TEXT DEFAULT '',
    metric_value REAL,
    secondary_metrics TEXT DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'running',
    description TEXT DEFAULT '',
    diff TEXT DEFAULT '',
    stdout_tail TEXT DEFAULT '',
    duration_seconds REAL,
    started_at TEXT,
    finished_at TEXT,
    synced_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class LocalTracker:
    """SQLite-backed experiment storage."""

    def __init__(self, db_path: str = "autoresearch.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def _row_to_experiment(self, row: sqlite3.Row) -> Experiment:
        d = dict(row)
        d["secondary_metrics"] = json.loads(d.get("secondary_metrics") or "{}")
        return Experiment(**d)

    def create_experiment(self, **kwargs) -> int:
        kwargs.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        if "secondary_metrics" in kwargs and isinstance(kwargs["secondary_metrics"], dict):
            kwargs["secondary_metrics"] = json.dumps(kwargs["secondary_metrics"])

        columns = ", ".join(kwargs.keys())
        placeholders = ", ".join("?" for _ in kwargs)
        values = list(kwargs.values())

        cur = self.conn.execute(
            f"INSERT INTO experiments ({columns}) VALUES ({placeholders})", values
        )
        self.conn.commit()
        return cur.lastrowid

    def update_experiment(self, exp_id: int, **kwargs):
        if "secondary_metrics" in kwargs and isinstance(kwargs["secondary_metrics"], dict):
            kwargs["secondary_metrics"] = json.dumps(kwargs["secondary_metrics"])

        sets = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [exp_id]
        self.conn.execute(f"UPDATE experiments SET {sets} WHERE id = ?", values)
        self.conn.commit()

    def get_experiment(self, exp_id: int) -> Optional[Experiment]:
        row = self.conn.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
        return self._row_to_experiment(row) if row else None

    def get_latest(self) -> Optional[Experiment]:
        row = self.conn.execute(
            "SELECT * FROM experiments ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return self._row_to_experiment(row) if row else None

    def get_best(self, direction: str = "minimize") -> Optional[Experiment]:
        order = "ASC" if direction == "minimize" else "DESC"
        row = self.conn.execute(
            f"SELECT * FROM experiments WHERE status = 'keep' AND metric_value IS NOT NULL "
            f"ORDER BY metric_value {order} LIMIT 1"
        ).fetchone()
        return self._row_to_experiment(row) if row else None

    def get_all(self, run_tag: Optional[str] = None, status: Optional[str] = None,
                last_n: Optional[int] = None) -> list[Experiment]:
        query = "SELECT * FROM experiments WHERE 1=1"
        params = []
        if run_tag:
            query += " AND run_tag = ?"
            params.append(run_tag)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY id ASC"
        if last_n:
            query = query.replace("ORDER BY id ASC", "ORDER BY id DESC")
            query += f" LIMIT {last_n}"
        rows = self.conn.execute(query, params).fetchall()
        experiments = [self._row_to_experiment(r) for r in rows]
        if last_n:
            experiments.reverse()
        return experiments

    def get_stats(self) -> dict:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) as cnt FROM experiments GROUP BY status"
        ).fetchall()
        stats = {row["status"]: row["cnt"] for row in rows}
        stats["total"] = sum(stats.values())
        return stats

    def next_experiment_number(self, run_tag: str) -> int:
        row = self.conn.execute(
            "SELECT MAX(experiment_number) as max_num FROM experiments WHERE run_tag = ?",
            (run_tag,)
        ).fetchone()
        return (row["max_num"] or 0) + 1

    def get_meta(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value)
        )
        self.conn.commit()

    def get_unsynced(self) -> list[Experiment]:
        rows = self.conn.execute(
            "SELECT * FROM experiments WHERE synced_at IS NULL AND status != 'running' ORDER BY id"
        ).fetchall()
        return [self._row_to_experiment(r) for r in rows]

    def close(self):
        self.conn.close()


class ApiSync:
    """Non-blocking HTTP sync to the webapp."""

    def __init__(self, endpoint: str, api_key: str, timeout_seconds: float = None):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._pending_threads: list[threading.Thread] = []

    def sync_experiment(self, experiment: Experiment, tracker: LocalTracker,
                        project_name: str = "") -> bool:
        """POST experiment data to the API. Non-blocking (runs in a thread)."""
        payload = {
            "project_name": project_name,
            "run_tag": experiment.run_tag,
            "experiment_number": experiment.experiment_number,
            "commit_hash": experiment.commit_hash,
            "parent_commit_hash": experiment.parent_commit_hash,
            "metric_name": experiment.metric_name,
            "metric_value": experiment.metric_value,
            "secondary_metrics": experiment.secondary_metrics,
            "status": experiment.status,
            "description": experiment.description,
            "diff": experiment.diff,
            "timeout_seconds": self.timeout_seconds,
            "duration_seconds": experiment.duration_seconds,
            "started_at": experiment.started_at,
            "finished_at": experiment.finished_at,
        }

        def _post():
            try:
                resp = requests.post(
                    f"{self.endpoint}/experiments/",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=10,
                )
                if not resp.ok:
                    print(f"Warning: API sync returned {resp.status_code}: {resp.text[:200]}")
                if resp.ok and experiment.id is not None:
                    tracker.update_experiment(
                        experiment.id,
                        synced_at=datetime.now(timezone.utc).isoformat()
                    )
            except Exception as e:
                print(f"Warning: API sync failed: {e}")

        thread = threading.Thread(target=_post, daemon=True)
        thread.start()
        self._pending_threads.append(thread)
        return True

    def wait(self, timeout: float = 15):
        """Wait for all pending sync threads to complete."""
        for t in self._pending_threads:
            t.join(timeout=timeout)
        self._pending_threads.clear()

    def sync_pending(self, tracker: LocalTracker):
        """Retry syncing any experiments that haven't been synced yet."""
        for exp in tracker.get_unsynced():
            self.sync_experiment(exp, tracker)


class Tracker:
    """Facade combining local storage + optional API sync."""

    def __init__(self, config: Config, db_path: str = "autoresearch.db"):
        self.config = config
        self.local = LocalTracker(db_path)
        self.api = (
            ApiSync(config.api.endpoint, config.api.key,
                    timeout_seconds=config.experiment.timeout)
            if config.api.key
            else None
        )

    def create_experiment(self, **kwargs) -> int:
        kwargs.setdefault("metric_name", self.config.metric.name)
        return self.local.create_experiment(**kwargs)

    def update_experiment(self, exp_id: int, **kwargs):
        self.local.update_experiment(exp_id, **kwargs)

    def sync(self, exp_id: int):
        """Sync a specific experiment to the API if configured."""
        if self.api:
            exp = self.local.get_experiment(exp_id)
            if exp:
                self.api.sync_experiment(exp, self.local, self.config.project.name)

    def sync_pending(self):
        if self.api:
            self.api.sync_pending(self.local)

    def get_latest(self) -> Optional[Experiment]:
        return self.local.get_latest()

    def get_best(self) -> Optional[Experiment]:
        return self.local.get_best(self.config.metric.direction)

    def get_all(self, **kwargs) -> list[Experiment]:
        return self.local.get_all(**kwargs)

    def get_stats(self) -> dict:
        return self.local.get_stats()

    def close(self):
        if self.api:
            self.api.wait()
        self.local.close()
