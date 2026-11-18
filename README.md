# A minimal probe of task modeling in language models

Code and data for the paper *A minimal probe of task modeling in language models* ([arXiv:2605.02028](https://arxiv.org/abs/2605.02028)).

The paper asks how far a language model can count. A model is shown a sequence of identical items and asked for the exact number, at every length in ascending order. Its counting capacity (CC) is the length at which the exact answer stops being its majority behavior, fitted to the whole record of right and wrong replies. This repository holds the evaluation harness, the analysis code, and every reply behind the reported numbers: 205 models in 296 configurations, the control experiments, the execution ladder, the object-tracking scans, and the mechanistic probes of six open-weight models.

## Layout

```
src/counting/   the package: tasks, the scan harness, capacity estimation, probes, figures
scripts/        runners and analyses, one file per experiment or table
configs/        run configurations (every evaluation runs from one YAML file)
tests/          offline tests, no API key needed
data/           the result data, one archive per experiment (see data/README.md)
```

## Setup

Python 3.11 or newer.

```bash
pip install -e .            # the harness
pip install -e '.[all]'     # plus probes, dual-task data, analysis
```

Evaluations call models through OpenRouter and read `OPENROUTER_API_KEY` from the environment. The analyses below need no key.

## Data

The data are packed as `data/*.tar.xz`, about 770 MB unpacked. `data/README.md` lists what each archive holds and where the paper uses it. Unpack them to `results/`, which is where the code reads them:

```bash
scripts/unpack_data.sh                  # everything
scripts/unpack_data.sh counting-data    # or only the named archives
```

`data/cc_catalogue.csv` gives the fitted capacity of every measurement without unpacking anything.

Two changes were made to the records for release. Absolute paths into the authors' checkout are relative to the repository, and OpenRouter generation ids are replaced by a hash of themselves, so records that shared an id still join.

## Reproducing the numbers

With the data unpacked:

| Result | Command |
|---|---|
| Fitted CC of every measurement | `python scripts/compute_scc.py results/counting-data` |
| Coverage of the interval on the boundary (Appendix B) | `python scripts/check_interval_calibration.py` |
| Decoding replicates (Appendix A.2) | `python scripts/build_replicates_table.py` |
| Encodings, output formats, concurrent loads, the METR screen (Appendices A.6, A.7, H, J) | `python scripts/analyze_rebuttal.py` |
| Tasks with a writable running value (Appendix E) | `python scripts/report_anchored_control.py` |
| Reasoning-trace modes (Appendix G) | `python scripts/report_trace_modes.py` |
| Execution ladder (Appendix H) | `python scripts/fit_full_ladder.py`, then `python scripts/analyze_ladder_censoring.py` and `python scripts/analyze_ladder_ranks.py` |
| METR, Balrog, WeirdML against general capability (Appendix J) | `python scripts/analyze_benchmark_robustness.py` |
| Object tracking (Appendix K) | `python scripts/analyze_tracking.py` |

The figure modules are under `src/counting/analysis/figures/` and run as `python -m counting.analysis.figures.<name>` (`attractors` for Fig. 2, `resources_row` for Fig. 3, `mechanistic_suite` for Fig. 4). They and the table builders write under `paper/`, which is not part of this repository, so create the output folders first:

```bash
mkdir -p paper/figures paper/generated paper/appendix
```

## Running an evaluation

```bash
python scripts/fetch_openrouter_models.py                 # roster of what OpenRouter serves today
counting run configs/experiments/openrouter_formal.yaml   # the catalogue scan
```

A run writes one folder under `results/` with every probe, a per-model record, and the tables. The other experiments have their own runner in `scripts/`: `run_temperature_replicates.py`, `run_motif_sweep.py`, `run_copy_tool.py`, `run_budget_rerun.py`, `run_state_solo.py`, `run_full_ladder.py`, `run_ladder_extensions.py`, `launch_interference.py`, `scan_tracking.py`, and the `rebuttal_*.py` runners for the encoding panel, the output formats, and the replicated concurrent loads.

The mechanistic probes run on local open-weight checkpoints through `counting-probe`, with one configuration per model in `configs/probes/`; `scripts/round2_suite.sh` and `scripts/subspace_suite.sh` run the suites of Appendix I for one model.

## What is not here

- The manuscript source, which is on arXiv.
- The activation caches of the probe suite. They are large and `counting-probe` regenerates them; the records the paper uses are in `data/suite.tar.xz`.
- The endpoint scans of Appendix A.3. The reseller is not named in the paper and its records are not released. `scripts/fingerprint_endpoint.py` is the protocol.

## Earlier versions

The first two arXiv versions of this work used a different protocol and estimator. Their code and data are in this repository's history at commit `5271bf3`.

## License

Code is released under the Apache License 2.0 (`LICENSE`). The data are released under MIT terms, with the exception stated in `DATA_LICENSE.md` for the benchmark scores from Epoch AI.
