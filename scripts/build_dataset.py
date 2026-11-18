#!/usr/bin/env python3
"""Build the flat dataset from one or more run directories.

    python scripts/build_dataset.py results/counting-data
    python scripts/build_dataset.py results/counting-data results/counting-full
    python scripts/build_dataset.py results/* --out results/counting-data

Writes `registry.csv` (one row per measurement) and `responses.csv` (one row per
probe) into `--out`, defaulting to the first run directory given.

Runs stack. Passing several directories concatenates them, tagged by `run`, so a
1..50 characterisation and a later full scan of the same models sit side by side
and can be compared or pooled. The tables are derived entirely from the
per-model JSON each run already wrote, so rebuilding is always safe and never
costs an API call.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counting.datastore import build_from_run, load_catalogue, write_tables  # noqa: E402

CATALOGUE = REPO_ROOT / "results" / "model_catalogue" / "openrouter_models.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, help="where to write (default: first run dir)")
    parser.add_argument(
        "--catalogue", type=Path, default=CATALOGUE,
        help="model catalogue CSV, for the static metadata columns",
    )
    args = parser.parse_args()

    runs = [d for d in args.run_dirs if (d / "models").is_dir()]
    if not runs:
        raise SystemExit("none of the given directories contains a models/ folder")

    catalogue = load_catalogue(args.catalogue)
    if not catalogue:
        print(f"note: no catalogue at {args.catalogue}; static columns will be blank",
              file=sys.stderr)

    # Later run directories override earlier ones for the same run key: a
    # configuration remeasured under a revised budget replaces its earlier
    # scan in the tables, and the `run` column records where each row came
    # from (the run name in the directory's manifest when it has one).
    import json
    registry, responses = [], []
    for run_dir in runs:
        run_name = None
        manifest = run_dir / "manifest.json"
        if manifest.exists():
            run_name = json.loads(manifest.read_text()).get("run_name")
        reg, resp = build_from_run(run_dir, catalogue, run=run_name)
        keys = {r["run_key"] for r in reg}
        dropped = sum(1 for r in registry if r["run_key"] in keys)
        registry = [r for r in registry if r["run_key"] not in keys] + reg
        responses = [r for r in responses if r["run_key"] not in keys] + resp
        print(f"  {(run_name or run_dir.name):<32} {len(reg):>4} measurements, {len(resp):>6} responses"
              + (f"  (replacing {dropped} earlier)" if dropped else ""), file=sys.stderr)

    out = args.out or runs[0]
    reg_path, resp_path = write_tables(out, registry, responses)

    def show(path: Path) -> str:
        # The run dirs may be given as relative paths; do not assume otherwise.
        resolved = path.resolve()
        try:
            return str(resolved.relative_to(REPO_ROOT))
        except ValueError:
            return str(resolved)

    print(f"\nwrote {show(reg_path)}  ({len(registry)} rows)")
    print(f"wrote {show(resp_path)}  ({len(responses)} rows)")

    matched = sum(1 for r in registry if r["release_date"])
    print(f"\n  {matched}/{len(registry)} measurements carry catalogue metadata")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
