# Study200 on RunPod

Study-only workflow: A100 SXM 80 GB, PascalVOC-SP and Peptides-func, seeds 1-3,
200 epochs for every full run (24 runs / 4,800 epochs). Original LRGB configurations,
attention, scheduling and non-study checkpoint behavior are preserved.

## Independent notebook execution

`colab/RunPod_Study200.ipynb` has common setup and **14 experiment-execution cells**:

| Cells | Runs |
| --- | --- |
| 1-4 | Seed 1: VOC adaptive; Peptides linear, exponential, adaptive |
| 5-8 | Same four runs, seed 2 |
| 9-12 | Same four runs, seed 3 |
| 13 | VOC linear/exponential, seeds 1-3: six full 200-epoch runs |
| 14 | Fixed50, both datasets/seeds 1-3: six full 200-epoch runs |

The order is recommended, not enforced. Every cell selects only its own job IDs;
no cell launches another cell. `run_job(manifest, id)` and `run_group(manifest, ids)`
default to command-only planning. Training requires explicit `execute=True`.
Run All resets notebook training/smoke switches to False. Completed jobs are skipped
only after validating their configuration, full finite epoch histories, checkpoint
audits, memory/timing histories, dataset identity and required adaptive diagnostics.

Use a persistent `/workspace/kmip_study200` volume. Output paths are isolated by
dataset/method/seed and never silently changed. Configuration hashes, repository
revision and pip freeze must match. A process launch lock prevents concurrent launches;
remove a stale lock manually only after verifying that no launcher/training process exists.

Existing schema-v1 manifests can be explicitly upgraded by `migrate_manifest(path)`.
It preserves config hashes, paths, statuses, attempts and approvals, saves an immutable
v1 backup, then changes the recommended presentation order. No outputs are moved.

## Global adaptive review and recovery

Every completed adaptive run blocks **all** experiment cells until explicit
`APPROVE <exact run ID>` confirmation is saved with the artifact fingerprint.
Approval persists across kernel/Pod restarts; modified artifacts invalidate it.
Review is a separate notebook utility, disabled by default; no batch auto-approval.
An incomplete/failed adaptive run blocks other jobs, but allows that exact job to resume.
Invalid artifacts must be investigated. Warnings never change hyperparameters automatically.

After investigating failures, use `run_job(..., execute=True, retry=True)` explicitly.
`study/latest.ckpt` retains model/optimizer/scheduler plus Python/NumPy/Torch/CUDA RNG.
Dataset content and ordering are verified before restore. Latest recovery publication
follows epoch metric/memory/timing writes, so interrupted suffixes can be replayed coherently.
Adaptive selected k is restored through model extra state and checked against its archive.

Successful full study training performs a CPU checkpoint audit of earliest-validation-best
and final recovery checkpoints: schema, epochs, model key/shape and finite-state checks.
Streaming checksums bind the audit to the actual files. Reporting verifies checksums without
requiring PyTorch. Pre-audit trusted local outputs can be audited explicitly with
`python -m study.checkpoints RUN_DIR --best-epoch E --final-epoch 199`.
Do not deserialize checkpoints from untrusted sources.

## CUDA/KeOps smoke workflow

After common environment/data setup, explicitly enable RUN_SMOKE. The launcher executes
eight isolated real-model cases (two datasets x four methods), two optimizer steps per
case, validation/test forward passes, synchronized allocator measurements, checkpoint
resume/RNG restoration, standalone state restoration and CPU integrity audits.
Each adaptive case performs one calibration: m=8500, Kmax=50, tau=.90, q=.95,
T=5, no artificial floor, and the planned 200-epoch DKW confidence horizon.
Calibration uses a deterministic eight-graph subset to keep pilot cost bounded.

`python -m study.smoke --manifest MANIFEST` is command-only by default; add `--execute`
to run the pilot. Results persist separately under `smoke/TIMESTAMP/`. Full training
preflight requires a passed eight-case receipt for the approved code/config/environment
snapshot. Compilation can dominate smoke runtime. Pilot graphs/batch size are intentionally
small; after this functional pilot, validate representative full-batch memory/runtime and
full-dataset calibration cost on RunPod. Local CPU tests cannot validate these GPU paths.

## Optional supplementary short profiling

`python -m study.profile MANIFEST` plans only. Explicit `--execute` enables eight
short profiling cases with two warmup optimizer steps, dataset study batch sizes,
and two measured train/eval/checkpoint windows. Defaults use 400 training graphs;
`--graphs` and `--batch-size` are explicit pilot controls. Baselines measure initial
and final requested k; adaptive measures initial and one sampled calibrated budget.
Models are freshly initialized, not trained final checkpoints. Calibration uses the pilot
subset, not the full training set. Results under `profiling/TIMESTAMP/` are labelled
`profiling_only`, and reported in a separate supplementary table.

These are exploratory subset-window peaks/times, not complete 200-epoch peaks/runtime,
trained-checkpoint estimates or a replacement for any full run. Do not extrapolate their
wall time to claim measured full-training runtime. Cells 13/14 remain full 200-epoch runs.

## Historical predictive integration

Use `study.historical.import_logs(manifest, source, import_id, method, prefix=...)`
or `python -m study.historical MANIFEST SOURCE ID --method fixed15|linear|exponential`.
SOURCE is a config-containing run directory or tar archive with an explicit `--prefix`.
Only config.yaml and original per-seed train/val/test stats are read/copied; archives are
never extracted. Sources stay untouched. Imports are checksummed and immutable, with
explicit rejection of partial/NaN/duplicate epochs and wrong schedule configurations.

Historical VOC fixed15 is a distinct method, never fixed50. All-seed 0-3 summaries and
paired-study-seed 1-3 summaries are separate, with earliest best-validation test F1 and
sample SD. Historical learning curves and paired F1 differences are retained. Hardware,
checkpoint integrity and unlogged schedule internals are not inferred. Historical runtime
and missing memory never enter matched A100 efficiency comparisons. Historical records do
not satisfy pending full-run jobs. Missing/corrupt imports produce explicit report alerts.

## Incremental independent reports

After every launched job, success or failure, `study.report.generate(manifest)` writes
`reports/latest/` and a timestamped snapshot. The notebook displays summaries and curves
after EACH job, including the two six-run batch cells. It can regenerate without training
via `python -m study.report MANIFEST`.

Per-dataset CSV/LaTeX tables and PDF/300-dpi PNG figures cover validation-selected test
F1/AP, learning curves, total/epoch runtime, allocated/reserved MiB, requested-k statistics,
calibration overhead, sampled mass coverage/required-k distributions/DKW parameters,
and paired performance-efficiency comparisons. Cross-dataset summaries normalize efficiency,
never pool F1 and AP. n=1 has no SD; n=2 is preliminary with sample SD; n=3 is complete
subject to outstanding adaptive review. Missing/failed/partial/invalid results are excluded.

Flags include nonfinite metrics, incomplete histories, config/identity mismatch, corrupt
checkpoints/archives, impossible allocator values, >2x allocated-memory jumps, k=50 for
three successive calibrations and calibration exceeding 25% of measured training time.
Warnings request investigation, not automatic parameter changes. Sampled calibration
coverage does not prove dense-attention coverage or between-calibration behavior.

Measured: saved predictive metrics, synchronized PyTorch allocated/reserved peaks,
phase/session wall times, calibration probabilities and observation metadata.
Derived: seed means/sample SD, paired ratios, budget/coverage distributions and overhead
fractions. Epoch peaks cover calibration, train, evaluation and epoch bookkeeping; overall
run peaks are maxima over observed epoch windows, retaining replay high-water marks.
Allocator measurements exclude non-PyTorch/driver memory and pre-loop model/data setup.
Reserved memory includes allocator cache. NVML snapshots are not allocator peaks.
Session runtime includes calibration/evaluation/checkpointing and observed replay costs;
launcher runtime additionally includes setup. Killed-session duration is never guessed.

Back up the complete study root, environment lock, source revision and dataset caches.
Verify backups before manually stopping the Pod. No training or billing automation is enabled.
