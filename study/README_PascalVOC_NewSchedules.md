# PascalVOC-SP schedule study

Open RunPod_PascalVOC_NewSchedules.ipynb on a machine with a working CUDA PyTorch
kernel, a compatible NVIDIA driver, and enough GPU memory for the tuned batch
size of 50. Set WORKSPACE_ROOT to a writable absolute path and select Run All.
Allow enough disk space for the dataset cache, source checkout and eight runs.

All study files are kept under WORKSPACE_ROOT.

- repository/<commit>/checkout holds the verified GitHub revision. A clean
  matching checkout is reused. The first fetch requires Git and read access to
  the revision. Private repositories need an existing Git credential helper
  or other Git authentication on the machine.
- datasets/ holds the PascalVOC-SP cache. The original repository loader
  reuses existing data and fetches missing data from its configured mirror.
- results/ holds the experiment attempts, manifests, metrics, reports, and
  optional ZIP archives.
- setup_logs/ holds Git, environment and dataset diagnostics, resolved
  software versions and worker caches.
- environment/ holds a managed worker environment when dependencies need to
  be installed.

For maintainers, publish the study code in an accessible commit before
distributing the notebook. Set PINNED_COMMIT in Cell 2 to that commit's full SHA.
The notebook can be distributed or committed separately from the code revision
it pins. The pre-release notebook stops at setup while the pin is unset.

Automatic setup is on by default. It verifies a previously created worker
environment or the notebook's Python first, including GPU operations in PyTorch
Geometric, torch-scatter and PyKeOps. When package repair is needed, it installs
into an isolated worker environment that inherits the existing CUDA PyTorch
installation. The same interpreter runs setup, training and export in separate
processes. The Jupyter kernel stays in place.

Use a GPU image with CUDA-enabled PyTorch, a compatible NVIDIA driver, Python
venv support, and the KeOps build toolchain. Automatic setup needs package-index
access and a matching torch-scatter wheel when installation is needed. If a
probe or installation fails, the error points to setup_logs/ for the exact
check and installation output.

## Running and retrying

Run All launches the eight method/seed cells sequentially. RUN_EXPERIMENTS
controls launching and RUN_SEEDS optionally restricts which cells execute. Each
seed has separate attempt directories under PascalVOC_NewSchedules. Valid
completions are skipped by default. Rerun a failed cell to start a fresh attempt
at epoch 0. Setting SKIP_COMPLETED = False also creates a fresh attempt for a
previously completed seed. All earlier logs are retained. The export uses the
newest validated completion per seed and lists every attempt and its status.

After a kernel restart, rerun the four setup cells before selecting a single
experiment cell. Setup compares code, dependencies, GPU, resolved configuration
and dataset identity with the manifest. Choose a new WORKSPACE_ROOT for a different
study identity. A worker lock prevents another launch against the same results
while an existing worker is active. A prior attempt with no verified process
exit is marked incomplete on the next setup, once its worker lock is released.
Isolated worker failures leave details in stdout.log and status.json and permit
the remaining cells to run. Shared prerequisite failures stop launches.

## Reading results

Retrieve results/ under WORKSPACE_ROOT. Each attempt includes its
source and resolved configurations, provenance, stdout.log, native train/val/test
stats.json, attention_budgets.jsonl, memory records, and execution status.
Training duration covers the synchronized training loop, evaluation and logging.
Setup and dataset loading occur outside that measured interval. GPU utilization,
when device UUID matching is available, is sampled for the whole GPU. CUDA
allocator peaks are separately measured in MiB within epoch windows.

Epochs 0 through 199 each evaluate train, validation and test. The earliest
maximum recorded validation F1 selects the corresponding test F1. The final
test F1 is recorded separately. Successful exit, complete finite histories,
matching configurations and full scored-node coverage determine completion.
The worker disables periodic, best-model and recovery saves and starts each
attempt at epoch 0. Recovered OOM retries are recorded. Dropped scored nodes
invalidate the attempt.

Both curves use Python int(round(value)), with exact halves rounded to even.
Mirrored exponential reflects the existing 50-to-10 exponential trajectory in
time and budget, using 60 minus the unrounded exponential value at the reversed
epoch. Decrease/increase reaches 10 at epoch 140 and 15 at epoch 199.

Each layer's epoch record distinguishes scheduled k, assigned k, and the
distribution of effective budgets min(assigned k, batch key width). Separate
histograms count the valid nodes in each graph presented to that layer.
Forward-call observations include retries if they occur. The original batched
attention can select padded keys. Valid graph-node counts therefore describe
the input graphs and cannot be used to infer the number of valid selected keys.

The final cell saves per-seed, aggregate, epoch and planned-schedule CSVs,
figures, and execution_status.json under summary. Only validated completions
contribute to aggregates, with sample standard deviations for two or more
seeds. Raw partial histories remain in their attempts. Rerun the final cell to
refresh reports. Set CREATE_ZIP = True to save an archive of the results under
results/.
