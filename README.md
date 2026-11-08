# Phase 3 CPU thread adaptation in federated learning

The implementation is in `src/`, configurations in `configs/`, correctness checks in `tests/`, and existing measurements in `results/`. Dataset downloads and caches use `dataset/`.

## Completed evaluation

The supplied report uses `results/short`: **45/45 runs**, 10,000 training images, ten rounds, two local epochs and three seeds. Nine correctness tests passed, and the completed analysis verified equal real training work and matching batch-order hashes. Resume behavior was also checked without rerunning completed smoke jobs.

| Load | Proposed versus pilot fixed, mean paired time change | Final accuracy difference |
|---|---|---|
| No injected load | 14.64% slower | 0.00 percentage points |
| Steady traffic | 5.66% slower | 0.00 percentage points |
| Intermittent traffic | 7.91% faster | 0.00 percentage points |

The proposed policy accepted **zero of 180 candidate probes**. The payback gate rejected zero candidates that passed the gain threshold; all candidates failed the gain test. Therefore these results do not demonstrate an adaptive-thread or payback-gate speedup. The intermittent timing advantage is reported as an observation, with substantial host/runtime variability and load-phase exposure as limitations. Identical fixed settings also produced different timings.

The contribution claim is the implemented matched-probe controller, its explicit heuristic and ablation, and the reproducible empirical study—not a first-ever scheduler or a proven performance advantage.

## Open in VS Code

Open the **overhead-aware-fl** folder. Select your virtual environment with **Python: Select Interpreter**. Run the commands below from the repository root.

## Setup on macOS or Linux

Use Python **3.10–3.12**; the pinned PyTorch 2.2.2 environment supports this Intel Mac. Create a local virtual environment for your machine.

```bash
cd /path/to/overhead-aware-fl
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/experiment.py check
python -m unittest discover -s tests -v
```

If Python 3.12 has another executable name, substitute it in the first command. An existing environment prepared in this workspace can simply be activated; no reinstall is needed. `PYTHON_BIN=python3.11 bash setup.sh` is an alternative installer. On Windows use `py -3.12 -m venv .venv` and `.venv\Scripts\Activate.ps1` in PowerShell; run the same `python` commands afterward.

## Run

```bash
# Small REAL-data correctness smoke experiment: 5 methods, one seed, no load.
python src/experiment.py all --config configs/smoke.json --out results/my-smoke

# Short research matrix: 10,000 examples, 10 rounds, 3 seeds, 5 methods × 3 loads = 45 runs.
python src/experiment.py all --config configs/short.json --out results/my-short

# Original 20-round budget, same 45-run matrix.
python src/experiment.py all --config configs/full.json --out results/my-full

# Regenerate tables and plots without retraining.
python src/experiment.py analyze --out results/my-short
python src/make_figures.py
```

`validation.json` is an explicitly smaller evaluation with 3,000 training examples and 3 rounds. It is a pipeline diagnostic, not equivalent to the 10-round configuration. The final report identifies its actual input run directory and configuration. Never relabel a smoke/validation run as the full experiment.

The exact transitive packages used for the executed experiment are also recorded in `requirements-lock.txt`. The smaller `requirements.txt` pins the main scientific dependencies.

Commands work from another directory if absolute paths are used. Standard output shows each pilot measurement and each completed round. Keep the computer plugged in and avoid unrelated heavy applications during measurements; power settings and thermal state are not automatically controlled. Memory traffic uses three 64 MiB arrays (192 MiB total), and the worker pool caches data in each worker.

### Resume safely

Repeat exactly the same command. Completed runs with matching configuration, code, data and environment identities are skipped. Interrupted runs are archived with `.interrupted-<timestamp>` and rerun from round zero. A changed experiment requires a new output directory. This prevents silent mixing of seeds or old configurations. Do not edit the four experiment source modules while a matrix is running.

## What is implemented

- Four sequential logical FedAvg clients; weighted aggregation by client sample count. Parallelism is in CPU tensor kernels, with a separate background process. This is not a multi-machine network benchmark.
- Original small CNN: unpadded 3×3 convolutions 1→16→32, ReLU and max pooling, linear 800→10, **12,810 parameters**.
- Fixed random Fashion-MNIST subset (seed 12345), official 10,000-image test set, original normalization 0.2860/0.3530, Dirichlet alpha 0.5.
- FP32 SGD, batch 64, learning rate 0.01, momentum 0.9. Momentum is preserved between a client's local epochs, reset at each global round. All models and real minibatch orders are paired across methods.
- Persistent worker processes for 1, 2 and 4 intra-op threads. Each fixes its thread limit before tensor operations and uses the same discarded two-batch warm-up procedure. Only one worker trains at a time. No hardware counters or affinity claims.
- Independent pilot seeds 100–102. `pilot_fixed` uses the fastest mean no-load pilot epoch wall time, without consulting test accuracy or evaluation seeds. Steady and compute-load pilot observations are diagnostics only.

| Method | Decision rule |
|---|---|
| `max_fixed` | Four threads throughout; “maximum” means maximum tested, not all CPU cores. |
| `pilot_fixed` | One fixed thread count selected from independent no-load pilots. |
| `reactive` | Starts at four; after two epochs below 90% pilot throughput, tries a lower count; retains it only for ≥5% throughput gain; may increase after two stable epochs. |
| `paired_ungated` | Same matched probes as proposed, ≥5% gain threshold, no cost-payback requirement. |
| `proposed` | Matched ABBA/BAAB microprobes, ≥5% gain and predicted savings exceeding 1.25× measured probing cost over at most four remaining local epochs. |

Both paired controllers start at the pilot winner. Every fourth local epoch they compare the incumbent with an adjacent count on the same first two minibatches, restoring the same model and momentum before every probe. Four probe calls use ABBA, alternating with BAAB. Probe weights are discarded. Real training runs exactly once afterward. The gain/payback rule is in `src/controllers.py`; all decisions and inputs are logged. This is an experimental heuristic: probe cost is already spent at acceptance time, so rejecting a faster candidate does not recover that cost. The report discusses moving cost control before probing as future work. Thread-dependent floating-point reduction differences remain possible despite identical work.

## Result files

Each output directory contains:

- `manifest.json`: exact configuration, data SHA-256, source hashes, host and library versions.
- `pilot.json`, `schedule.json`: independent pilot and reproducibly shuffled job order.
- `runs/<condition>__<method>__seed<N>/`: complete summary, exact partition, per-epoch JSONL, accuracy history and final NPZ checkpoint.
- `analysis/summary.csv`, `per_seed.csv`, `paired_comparisons.csv`: mean ± sample SD, individual seeds and paired comparisons.
- `analysis/completeness.json`: missing-run audit; never conceal an incomplete matrix.
- `analysis/*.png` and `*.svg`: plots with editable vector versions.

The primary runtime starts before model creation and worker startup and ends after final checkpoint creation/hash and process shutdown. It includes warm-up, background setup, probes, model transfers, aggregation, evaluation and in-run logging. Dataset preparation, partition generation, independent pilot and final summary writing are outside each run's boundary. Training work excludes disposable probes; their time is included in total runtime and probe examples are separately counted. `controller_s` **includes** `probe_s`; do not add those two fields.

Time reduction is computed per seed as `100 * (1 - proposed_time / baseline_time)`, then averaged. Positive means faster. Accuracy differences are percentage points. The descriptive usefulness criterion is positive mean paired time reduction with mean accuracy no more than 1 percentage point lower. Three seeds do not establish statistical significance. Negative results remain valid results.

## Contribution and claim boundaries

The implemented course contribution is a matched-probe thread controller with a measured-overhead gate, a gate-removal ablation, and a reproducible end-to-end evaluation. Adaptive thread selection already exists in prior research. This project does not establish a first-ever technique, direct DRAM-contention detection, distributed-network speedup, energy savings or convergence guarantees. The report makes performance claims only from completed measurements.

## Files to read

Start with `src/experiment.py` for orchestration, `src/controllers.py` for the contribution, `src/workers.py` for process execution, and `src/core.py` for the unchanged learning workload. `tests/test_pipeline.py` checks partition coverage, sample-weighted aggregation, payback decisions, momentum preservation, probe isolation and incompatible resume rejection. `src/analyze.py` verifies paired batch-order hashes and equal real training work before making tables.

Existing result manifests retain their original source hashes. Because file paths changed during reorganization, use a new output directory for future experiments; existing results remain available for analysis. Architecture figures generated by `src/make_figures.py` are saved to `results/short/analysis/`.
