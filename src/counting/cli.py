"""Command-line entry point.

    counting run configs/experiments/frontier_ladder.yaml
    counting run configs/experiments/dense_sweep.yaml --models gpt-5.4 --dry-run
    counting list-tasks
    counting show-config configs/experiments/frontier_ladder.yaml

One config drives one run. Command-line flags override the config for the things
you change while iterating (which models, where output goes, whether to redo work
already on disk) and nothing else -- anything that affects the measurement lives
in the config file so that a run is reproducible from it.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .config import CONFIG_ROOT, ConfigError, get, load_experiment, load_roster
from .datastore import build_from_run, load_catalogue, write_tables
from .metrics import format_metric
from .providers import PROVIDERS, build_provider, resolve_provider_name
from .runner import ProviderUnavailable, Runner, RunnerSettings
from .search import STRATEGIES, build_strategy
from .storage import (
    RunPaths,
    run_key,
    TrialLog,
    completed_models,
    load_model_summaries,
    write_model_output,
    write_run_indexes,
)
from .tasks import build_task, ladder
from .types import TierVerdict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="counting", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an experiment from a config file")
    run.add_argument("config", type=Path, help="path to a configs/experiments/*.yaml file")
    run.add_argument("--models", nargs="+", help="evaluate only these model ids")
    run.add_argument("--output-dir", type=Path, help="override results root (default: results/)")
    run.add_argument("--run-name", help="override the run directory name")
    run.add_argument(
        "--force", action="store_true", help="re-evaluate models already on disk"
    )
    run.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="models to evaluate in parallel (each model is still probed in order)",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve config, print the plan, and make no API calls",
    )

    sub.add_parser("list-tasks", help="print the complexity ladder")
    sub.add_parser("list-providers", help="print the registered providers")
    sub.add_parser("list-strategies", help="print the registered search strategies")

    show = sub.add_parser("show-config", help="print a fully resolved config")
    show.add_argument("config", type=Path)

    models = sub.add_parser("list-models", help="print the model roster a config resolves to")
    models.add_argument("config", type=Path)

    return parser


# -- commands ----------------------------------------------------------------

def cmd_list_tasks() -> int:
    print("Complexity ladder\n")
    for task_cls in ladder():
        print(f"  {task_cls.level}  {task_cls.name:<20} {task_cls.description}")
    return 0


def cmd_list_providers() -> int:
    print("Providers\n")
    for name in sorted(PROVIDERS):
        print(f"  {name}")
    return 0


def cmd_list_strategies() -> int:
    print("Search strategies\n")
    for name, cls in sorted(STRATEGIES.items()):
        print(f"  {name:<10} {cls.description}")
    return 0


def cmd_show_config(path: Path) -> int:
    print(json.dumps(load_experiment(path), indent=2, sort_keys=True))
    return 0


def cmd_list_models(path: Path) -> int:
    for entry in load_roster(load_experiment(path).get("models")):
        extras = {k: v for k, v in entry.items() if k != "id"}
        suffix = f"  {extras}" if extras else ""
        print(f"{entry['id']}{suffix}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = load_experiment(args.config)

    roster = load_roster(config.get("models"))
    if args.models:
        wanted = set(args.models)
        roster = [entry for entry in roster if entry["id"] in wanted]
        missing = wanted - {entry["id"] for entry in roster}
        # A model named on the command line but absent from the roster is still
        # evaluated -- it is the fastest way to try one model against a config.
        roster.extend({"id": model} for model in sorted(missing))
    if not roster:
        print("No models selected. Check `models:` in the config.", file=sys.stderr)
        return 1

    provider_name = resolve_provider_name(get(config, "provider.name", "openrouter"))
    provider_options = {
        k: v for k, v in (get(config, "provider", {}) or {}).items() if k != "name"
    }
    task_options = dict(get(config, "task", {}) or {})
    task_name = task_options.pop("name", "homogeneous_count")
    strategy_options = dict(get(config, "search", {}) or {})
    strategy_name = strategy_options.pop("strategy", "ladder")
    scan_seed_trial_logs = [
        Path(path) for path in (get(config, "scan_seed_trial_logs", []) or [])
    ]

    settings = RunnerSettings.from_config(config)
    output_dir = args.output_dir or Path(get(config, "output_dir", "results"))
    run_name = args.run_name or get(config, "run_name") or Path(args.config).stem

    print(f"config    {config.get('_source', args.config)}")
    print(f"provider  {provider_name}")
    print(f"task      {task_name}  {task_options or ''}")
    print(f"strategy  {strategy_name}")
    print(f"run       {output_dir}/{run_name}")
    print(f"models    {len(roster)}")

    if args.dry_run:
        probe = build_strategy(strategy_name, **strategy_options)
        planned = _plan_preview(probe)
        if probe.name == "scan":
            print(f"\nfirst lengths:  {planned}")
            print("requests/length: 1")
            print(f"output budget:   {settings.max_output_tokens} tokens, "
                  f"{'escalating' if settings.escalate_output_budget else 'fixed'}")
            print(f"stops on:        {strategy_options.get('stop_after_consecutive_errors')} "
                  f"consecutive errors, or "
                  f"{strategy_options.get('stop_at_first_mistake_multiple')}x the first mistake")
        else:
            print(f"\nfirst tiers: {planned}")
            print(f"trials/tier: {settings.samples_per_length}")
        build_task(task_name, **task_options)  # surfaces bad task options now
        print("\ndry run: no requests issued")
        return 0

    provider = build_provider(provider_name, **provider_options)
    task = build_task(task_name, **task_options)
    run_paths = RunPaths.create(output_dir, run_name)

    # Between models, a completed models/<id>.json means that model is done.
    # --force re-evaluates them.
    existing = {} if args.force else completed_models(run_paths)
    summaries: dict[str, Any] = load_model_summaries(existing)
    requested = [run_key(entry) for entry in roster]
    write_run_indexes(run_paths, config, requested, summaries, status="running")

    lock = threading.Lock()

    def evaluate(entry: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str]:
        """Run one model configuration. Returns (key, summary_or_None, line)."""
        model = entry["id"]
        key = run_key(entry)
        strategy = build_strategy(strategy_name, **strategy_options)
        is_scan = strategy.name == "scan"

        # A scan checkpoints every response as it arrives, so an interrupted
        # model resumes at the next length instead of restarting at 1.
        log = TrialLog(run_paths.trial_log_path(key)) if is_scan else None
        done = log.completed() if log else {}
        replay_done = done
        if is_scan and scan_seed_trial_logs:
            # Extension runs write to independent checkpoints, but their live
            # stopping criterion must see the stable prefix from the original
            # scan.  Merge seed records only for replay; new responses continue
            # to append exclusively to this run's own trial log.
            replay_done = {}
            for seed_path in scan_seed_trial_logs:
                if seed_path.exists():
                    replay_done.update(TrialLog(seed_path).completed())
            replay_done.update(done)

        # Per-model runner: the shared provider is stateless, but the callbacks
        # and the trial log are not, so each model gets its own.
        runner = Runner(provider, task, settings, on_trial=log.append if log else None)
        try:
            if is_scan:
                trials, summary = runner.run_scan(
                    model, strategy,
                    reasoning_effort=entry.get("reasoning_effort"),
                    already_done=replay_done,
                    temperature_locked=bool(entry.get("temperature_locked")),
                )
            else:
                trials, summary = runner.run_model(
                    model, strategy, reasoning_effort=entry.get("reasoning_effort")
                )
        except ProviderUnavailable as exc:
            return key, None, f"skipped ({exc})"
        except Exception as exc:  # noqa: BLE001 - one bad model must not end the sweep
            return key, None, f"FAILED {type(exc).__name__}: {exc}"
        finally:
            if log:
                log.close()

        summary["model_metadata"] = {k: v for k, v in entry.items() if k != "id"}
        summary["run_key"] = key
        summary["variant"] = entry.get("variant")
        with lock:
            write_model_output(
                run_paths, config, key, summary, trials, summary["model_metadata"]
            )
        if is_scan:
            line = (
                f"CC={summary['stable_count']}  first_mistake={summary['first_mistake']}  "
                f"probed={summary['lengths_probed']}  "
                f"ok/wrong/err={summary['correct_count']}/{summary['mistake_count']}/"
                f"{summary['error_count']}  {summary['stopped_early'] or 'range end'}"
            )
        else:
            line = f"stable_count={summary['stable_count']}"
        return key, summary, line

    todo = [e for e in roster if run_key(e) not in existing]
    if len(todo) < len(roster):
        print(f"\n{len(roster) - len(todo)} models already complete, skipping "
              f"(use --force to redo)")

    workers = max(1, args.concurrency)
    print(f"\nevaluating {len(todo)} models, {workers} at a time\n")
    finished = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for key, summary, line in pool.map(evaluate, todo):
                finished += 1
                print(f"  [{finished:>3}/{len(todo)}] {key:<48} {line}", flush=True)
                if summary is not None:
                    with lock:
                        summaries[key] = summary
                        write_run_indexes(
                            run_paths, config, requested, summaries, status="running"
                        )
    except KeyboardInterrupt:
        print("\ninterrupted; every response so far is checkpointed", file=sys.stderr)
        write_run_indexes(run_paths, config, requested, summaries, status="interrupted")
        return 130

    write_run_indexes(run_paths, config, requested, summaries, status="complete")
    _write_dataset(run_paths.run_dir)
    print(f"\nwrote {run_paths.run_dir}")
    return 0


def _write_dataset(run_dir: Path) -> None:
    """Refresh the flat registry/responses tables for a finished run.

    Derived from the per-model JSON rather than accumulated in memory, so it is
    correct whether the run completed in one pass or resumed across several.
    """
    catalogue = load_catalogue(
        CONFIG_ROOT.parent / "results" / "model_catalogue" / "openrouter_models.csv"
    )
    registry, responses = build_from_run(run_dir, catalogue)
    if not registry:
        return
    write_tables(run_dir, registry, responses)
    print(f"  registry.csv   {len(registry)} measurements")
    print(f"  responses.csv  {len(responses)} responses")


def _plan_preview(strategy: Any, limit: int = 8) -> list[int]:
    """First few centers a strategy would visit, assuming every tier passes."""
    grid = getattr(strategy, "grid", None)
    if grid is not None:
        return list(grid[:limit])
    centers: list[int] = []
    center = strategy.next_center()
    while center is not None and len(centers) < limit:
        centers.append(center)
        strategy.record(
            TierVerdict(
                center_length=center,
                phase="expansion",
                evaluation_index=len(centers) - 1,
                sample_low=center,
                sample_high=center,
                passed=True,
                trial_count=0,
                parse_rate=1.0,
                exact_match_rate=1.0,
                mape=0.0,
                mae=0.0,
            )
        )
        center = strategy.next_center()
    return centers


def _print_scan(verdict: TierVerdict) -> None:
    """One line per probe, but only when something notable happened.

    A step-1 scan to length 4000 would otherwise emit 4000 lines. Correct
    answers are summarised every 250 lengths; mistakes and errors always print,
    because those are the events the run exists to find.
    """
    from .search import classify

    outcome = classify(verdict)
    if outcome == "correct":
        if verdict.center_length % 250 == 0:
            print(f"  ...{verdict.center_length} correct")
        return
    predicted = (verdict.summary or {}).get("predicted")
    detail = f"said {predicted}" if outcome == "mistake" else "no parseable integer"
    print(f"  L={verdict.center_length:<7} {outcome.upper():<8} {detail}")


def _print_tier(verdict: TierVerdict) -> None:
    print(
        f"  L={verdict.center_length:<6} {verdict.phase:<10} "
        f"{'pass' if verdict.passed else 'FAIL'}  "
        f"nMAE={format_metric(verdict.mape)}  parse={format_metric(verdict.parse_rate)}"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return cmd_run(args)
        if args.command == "list-tasks":
            return cmd_list_tasks()
        if args.command == "list-providers":
            return cmd_list_providers()
        if args.command == "list-strategies":
            return cmd_list_strategies()
        if args.command == "show-config":
            return cmd_show_config(args.config)
        if args.command == "list-models":
            return cmd_list_models(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        print(f"(configs are resolved relative to {CONFIG_ROOT})", file=sys.stderr)
        return 1
    except KeyError as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
