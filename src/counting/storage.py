"""Run output layout, and resume.

A run directory is keyed by name, not by timestamp:

    results/<run_name>/
        registry.csv         one row per measurement -- the unit of analysis
        responses.csv        one row per probe -- input count vs output count
        manifest.json        config, model roster, per-measurement summaries
        models/<key>.json    full trial records for one measurement
        trials/<key>.jsonl   append-only checkpoint, written during the run

    The two CSVs are the format analysis reads; see datastore.py. They are
    derived from the per-model JSON, so scripts/build_dataset.py can rebuild
    them, and several runs can be merged into one pair of tables.

Keying by name is what makes runs resumable: re-running the same config skips
models that already have a complete file, so an interrupted sweep picks up where
it stopped instead of paying for the whole thing again. Pass `--force` to
re-evaluate models that are already present.

Resume works at two granularities. Between models, a completed `models/<id>.json`
means that model is done. Within a model, `trials/<id>.jsonl` is appended to as
each response arrives, so a scan interrupted at length 4000 restarts at 4001
rather than at 1. Every response is written whether it parsed or not -- an
unparsed reply is data about the failure, and it was paid for either way.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import TrialResult


@dataclass
class RunPaths:
    run_name: str
    timestamp: str
    run_dir: Path
    models_dir: Path
    manifest_path: Path

    @classmethod
    def create(cls, output_dir: Path, run_name: str) -> "RunPaths":
        run_dir = Path(output_dir) / run_name
        models_dir = run_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "trials").mkdir(parents=True, exist_ok=True)
        return cls(
            run_name=run_name,
            timestamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            run_dir=run_dir,
            models_dir=models_dir,
            manifest_path=run_dir / "manifest.json",
        )

    def model_path(self, model: str) -> Path:
        return self.models_dir / model_filename(model)

    def trial_log_path(self, model: str) -> Path:
        return self.trials_dir / (model_filename(model).removesuffix(".json") + ".jsonl")

    @property
    def trials_dir(self) -> Path:
        return self.run_dir / "trials"


def model_filename(model: str) -> str:
    """Filesystem-safe name for a model *run key*.

    The key is the model id, optionally suffixed with a variant (see
    `run_key`). Two configurations of the same model -- reasoning on and
    reasoning off -- are different measurements and must not collide.
    """
    return f"{re.sub(r'[^A-Za-z0-9._-]+', '__', model)}.json"


def run_key(entry: dict[str, Any]) -> str:
    """Identity of one measurement: the model, plus a variant when set."""
    variant = entry.get("variant")
    return f"{entry['id']}#{variant}" if variant else entry["id"]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def completed_models(run_paths: RunPaths) -> dict[str, Path]:
    """Finished measurements on disk, keyed by run key.

    Keyed on the run key rather than the model id: one model can contribute two
    measurements (reasoning on and off), and keying on the id alone would report
    the second as already done. The key comes from the summary where present and
    from the filename otherwise, since the filename is derived from it.
    """
    found: dict[str, Path] = {}
    if not run_paths.models_dir.exists():
        return found
    for path in sorted(run_paths.models_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summary = payload.get("summary") or {}
        if "stable_count" not in summary:
            continue
        key = payload.get("run_key") or summary.get("run_key")
        if not key:
            model = payload.get("model")
            variant = summary.get("variant") or (payload.get("model_metadata") or {}).get("variant")
            key = f"{model}#{variant}" if variant else model
        if isinstance(key, str):
            found[key] = path
    return found


def load_model_summaries(paths: dict[str, Path]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for model, path in paths.items():
        try:
            summaries[model] = json.loads(path.read_text(encoding="utf-8"))["summary"]
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    return summaries


def write_model_output(
    run_paths: RunPaths,
    config: dict[str, Any],
    key: str,
    summary: dict[str, Any],
    trials: list[TrialResult],
    model_metadata: dict[str, Any] | None = None,
) -> Path:
    """Write one measurement.

    `key` is the run key (`model` or `model#variant`) and names the file. The
    plain model id is recorded separately, because anything joining against the
    model catalogue needs the id and not the key.
    """
    path = run_paths.model_path(key)
    write_json(
        path,
        {
            "generated_at_utc": run_paths.timestamp,
            "run_name": run_paths.run_name,
            "config": config,
            "run_key": key,
            "model": key.split("#", 1)[0],
            "model_metadata": model_metadata or {},
            "summary": summary,
            "trials": [asdict(trial) for trial in trials],
        },
    )
    return path


def write_run_indexes(
    run_paths: RunPaths,
    config: dict[str, Any],
    requested_models: list[str],
    model_summaries: dict[str, Any],
    status: str = "running",
) -> None:
    """Rewrite manifest.json.

    Called after every measurement so an interrupted run still leaves a readable
    index. The flat tables are written once at the end, from the per-model JSON.
    """
    write_json(
        run_paths.manifest_path,
        {
            "generated_at_utc": run_paths.timestamp,
            "run_name": run_paths.run_name,
            "status": status,
            "config": config,
            "requested_models": requested_models,
            "models": model_summaries,
            "model_files": {
                model: run_paths.model_path(model).name for model in model_summaries
            },
        },
    )


class TrialLog:
    """Append-only checkpoint of every response for one model.

    One JSON object per line, flushed on write, so a run killed at any moment
    keeps everything already returned. Reopening the same path resumes: existing
    lines are read back and their lengths skipped.

    A truncated final line -- the normal consequence of being killed mid-write --
    is dropped on read rather than raising, since the trial it represents was
    never completely recorded.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = None

    def completed(self) -> dict[int, TrialResult]:
        """Lengths already probed, keyed by length."""
        if not self.path.exists():
            return {}
        done: dict[int, TrialResult] = {}
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated tail from an interrupted write
                payload.pop("metadata", None)
                try:
                    trial = TrialResult(**payload)
                except TypeError:
                    continue  # written by an older schema
                if trial.phase == "scan":
                    done[trial.center_length] = trial
        return done

    def append(self, trial: TrialResult) -> None:
        if self._handle is None:
            self._handle = self.path.open("a", encoding="utf-8")
        self._handle.write(json.dumps(asdict(trial), sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "TrialLog":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
