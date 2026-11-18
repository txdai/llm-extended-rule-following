# Data

One archive per results folder. `scripts/unpack_data.sh` checks the archives against
`SHA256SUMS` and unpacks them to `results/` at the repository root, where the code reads them.
`cc_catalogue.csv` is the fitted counting capacity of every measurement, for reading without unpacking.

| Archive | Holds | Used in | Files | Unpacked | Packed |
|---|---|---|---:|---:|---:|
| `counting-data.tar.xz` | Catalogue scans of 205 models in 296 configurations: every probe, the per-model records, and the fitted capacities | Sections 1 and 3; Appendices A to D | 638 | 131.0 MB | 5.9 MB |
| `temperature-replicates.tar.xz` | Five rescans of the transition region for ten models | Appendix A.2; Fig. 3D | 254 | 18.0 MB | 904 kB |
| `motif-sweep.tar.xz` | Eleven encodings for the three models of the main figure | Fig. 2F | 99 | 23.3 MB | 816 kB |
| `rebuttal-motif16.tar.xz` | Eleven encodings for thirteen more API models | Appendix A.6 | 360 | 62.3 MB | 2.1 MB |
| `rebuttal-motif-local.tar.xz` | Eleven encodings for the six open-weight models, run locally | Appendix A.6 | 132 | 396 kB | 13 kB |
| `rebuttal-exp1.tar.xz` | Output formats (integer, words, spaced digits, last digit), API models | Appendix A.7 | 100 | 21.0 MB | 935 kB |
| `rebuttal-exp1-local.tar.xz` | Output formats, open-weight models run locally | Appendix A.7 | 32 | 137 kB | 7 kB |
| `state-solo.tar.xz` | Tasks whose running value can be written down (pointer chains, update streams, an automaton) | Appendix E | 22 | 4.4 MB | 216 kB |
| `copy-tool.tar.xz` | Verbatim copy of the sequence into a tool call | Appendix F | 2 | 15 kB | 2 kB |
| `budget-32k.tar.xz` | Rescans of budget-bounded reasoning configurations with a fourfold output cap | Appendix G.1 | 81 | 51.3 MB | 1.6 MB |
| `harvested-traces.tar.xz` | Stored reasoning traces retrieved for the reasoning configurations | Appendix G | 122 | 11.7 MB | 959 kB |
| `harvested-traces-enriched.tar.xz` | The same traces joined to their probes | Appendix G | 118 | 11.4 MB | 935 kB |
| `trace-classification.tar.xz` | Labels assigned to the traces and the rubric outputs | Appendix G | 4 | 631 kB | 117 kB |
| `ladder-cohort.tar.xz` | Execution ladder: cohort roster and brackets | Appendix H | 15 | 160 kB | 26 kB |
| `ladder-rung2.tar.xz` | Execution ladder, pilot scans of the row tasks | Appendix H | 5 | 2.1 MB | 131 kB |
| `ladder-rung3.tar.xz` | Execution ladder, pilot scans of the rule tasks | Appendix H | 15 | 10.0 MB | 640 kB |
| `ladder-rung4.tar.xz` | Execution ladder, pilot scans of the grid tasks | Appendix H | 10 | 9.6 MB | 617 kB |
| `ladder-full.tar.xz` | Execution ladder, six tasks across the post-release cohort | Appendix H; Fig. 5G | 877 | 163.1 MB | 11.1 MB |
| `ladder-tailored-8192.tar.xz` | Execution ladder, scans tailored to each model's bracket | Appendix H | 178 | 111.4 MB | 14.1 MB |
| `ladder-extended-lower.tar.xz` | Execution ladder, extensions below the first bracket | Appendix H | 270 | 5.3 MB | 338 kB |
| `ladder-extended-upper-4x.tar.xz` | Execution ladder, extensions above the first bracket | Appendix H | 159 | 61.5 MB | 5.2 MB |
| `ladder-extensions.tar.xz` | Execution ladder, the roster of extension scans | Appendix H | 3 | 7 kB | 1 kB |
| `interference.tar.xz` | Counting under a concurrent load, single scans | Appendix H, Table 16 | 80 | 6.5 MB | 403 kB |
| `rebuttal-interference.tar.xz` | Counting under a concurrent load, five scans per cell | Appendix H, Table 17; Fig. 3C | 300 | 26.0 MB | 1.5 MB |
| `rebuttal-interference-local.tar.xz` | Counting under a concurrent load, open-weight models run locally | Appendix H | 156 | 520 kB | 22 kB |
| `excluded.tar.xz` | A run that was measured and excluded, kept as the record of why | Appendix H | 40 | 5.6 MB | 348 kB |
| `suite.tar.xz` | Mechanistic probe suite for six open-weight models: readouts, interventions, transplants | Section 4; Appendix I | 234 | 22.5 MB | 1.4 MB |
| `tracking.tar.xz` | Object-tracking scans, 33 models in 46 configurations | Section 5; Appendix K | 107 | 1.9 MB | 112 kB |
| `benchmark_data.tar.xz` | Public benchmark scores (Epoch AI Benchmarking Hub, CC BY 4.0) and the tables that match them to the catalogue | Section 5; Appendix J | 92 | 3.1 MB | 298 kB |
| `model_catalogue.tar.xz` | Snapshot of the OpenRouter model listing and prices used for model metadata | Appendix C | 3 | 275 kB | 19 kB |
| `tables.tar.xz` | The model inventory, the audit of replies that were not a single number, and the object-tracking subset | Appendices C, D and K | 4 | 3.9 MB | 201 kB |

Total: 4512 files, 768.9 MB unpacked, 51.0 MB packed.

Two changes were made to the records for release. Absolute paths into the authors' checkout are
relative to the repository. OpenRouter generation ids are replaced by a hash of themselves, so
records that shared an id still join.
