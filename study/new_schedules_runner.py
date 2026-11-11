"""Quiet orchestration for the checkpoint-free PascalVOC schedule study."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import sysconfig
import tempfile
import traceback


ROOT = Path(__file__).resolve().parents[1]
STUDY_NAME = "PascalVOC_NewSchedules"
METHODS = ("mirrored_exponential", "decrease_increase")
SEEDS = (0, 1, 2, 3)
EPOCHS = 200
SPLITS = ("train", "val", "test")
CONFIG_DIR = ROOT / "configs" / STUDY_NAME
TUNED = ROOT / "configs/LRGB/PascalVOC/tuned/GPS+kmip.yaml"


class SharedSetupError(RuntimeError):
    """A shared prerequisite failed, so subsequent cells must also stop."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_status(path):
    status = read_json(path)
    if not isinstance(status, dict) or status.get("status") not in (
        "pending", "running", "completed", "failed", "incomplete",
    ):
        raise ValueError(f"Invalid execution status in {path}")
    return status


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_lock(path):
    """The OS releases this lock on process exit, including abrupt termination."""
    with Path(path).open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise SharedSetupError(f"Another process holds {path}") from error
        else:
            import fcntl

            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise SharedSetupError(f"Another process holds {path}") from error
        yield


def repository_identity(repo):
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=repo, text=True, encoding="utf-8").strip()

    commit = git("rev-parse", "HEAD")
    dirty = git("--no-optional-locks", "status", "--porcelain")
    paths = [repo / "main.py", repo / "study/requirements.txt", repo / "study/__init__.py"]
    paths.extend((repo / "graphgps").rglob("*.py"))
    paths.extend((repo / "study").glob("new_schedules_*.py"))
    paths.extend((repo / "configs" / STUDY_NAME).glob("*.yaml"))
    paths.append(repo / TUNED.relative_to(ROOT))
    content = hashlib.sha256()
    for path in sorted(paths):
        content.update(path.relative_to(repo).as_posix().encode() + b"\0")
        content.update(bytes.fromhex(digest(path)))
    return {"commit": commit, "dirty": bool(dirty), "dirty_status": dirty, "source_sha256": content.hexdigest()}


def configuration_identity(config):
    # GraphGym's output paths, seed and resolved CUDA device vary per attempt.
    transient = {"out_dir", "run_dir", "run_id", "seed", "device"}
    stable = {key: value for key, value in config.items() if key not in transient}
    return hashlib.sha256(json.dumps(stable, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_epochs(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [row["epoch"] for row in rows] != list(range(EPOCHS)):
        raise ValueError(f"{path.name} must contain ordered epochs 0 through 199 exactly once")
    return rows


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_attempt(attempt, manifest, method, seed):
    """Validate native histories and independent execution records for a successful exit."""
    status = read_status(attempt / "status.json")
    identity = {"method": method, "seed": seed, "study_id": manifest["study_id"]}
    if status.get("identity") != identity or status.get("exit_code") != 0:
        raise ValueError("Attempt identity or successful process exit is missing")
    if status.get("status") not in ("running", "completed"):
        raise ValueError("Attempt is not eligible for completion")
    provenance = read_json(attempt / "provenance.json")
    if provenance.get("identity") != identity or provenance.get("dataset") != manifest["dataset"]:
        raise ValueError("Run provenance or dataset differs from the manifest")
    if provenance["environment"] != manifest["environment"]:
        raise ValueError("Run environment differs from the manifest")
    if digest(attempt / "source_config.yaml") != manifest["config_sha256"][method]:
        raise ValueError("Attempt source configuration differs from the manifest")
    effective = read_json(attempt / "effective_config.json")
    if (configuration_identity(effective) != manifest["effective_config_sha256"][method]
            or digest(attempt / "effective_config.yaml") != provenance["effective_yaml_sha256"]
            or provenance["rng"]["seed"] != seed
            or not (attempt / "stdout.log").stat().st_size):
        raise ValueError("Effective configuration, RNG provenance or worker log is missing or inconsistent")
    architecture = manifest["architecture"]
    if (effective["seed"] != seed or effective["gt"]["sparse"]["k_schedule"] != method
            or effective["optim"]["max_epoch"] != EPOCHS or effective["metric_best"] != "f1"
            or effective["train"]["eval_period"] != 1
            or effective["gt"]["dim_hidden"] != architecture["hidden_dimension"]
            or effective["gnn"]["layers_post_mp"] != 1 or effective["params"] != architecture["parameter_count"]
            or any(effective["train"][flag] for flag in ("enable_ckpt", "auto_resume", "ckpt_best", "ckpt_clean"))):
        raise ValueError("Effective configuration differs from the planned experiment")
    if any(path.name == "ckpt" or path.suffix == ".ckpt" for path in attempt.rglob("*")):
        raise ValueError("Unexpected checkpoint output")
    timing = read_json(attempt / "training.json")
    if (timing.get("identity") != identity or not finite(timing.get("training_seconds"))
            or timing["training_seconds"] <= 0):
        raise ValueError("Missing synchronized training runtime")
    histories = {split: read_epochs(attempt / split / "stats.json") for split in SPLITS}
    budgets = read_epochs(attempt / "attention_budgets.jsonl")
    memory = read_epochs(attempt / "memory/epochs.jsonl")
    planned = manifest["trajectories"][method]
    expected_nodes = {split: manifest["dataset"]["splits"][split]["nodes"] for split in SPLITS}
    epoch_rows = []
    for epoch, (budget, mem) in enumerate(zip(budgets, memory)):
        k = planned[epoch]
        if (budget["scheduled_k"] != k or mem["k"] != k
                or budget["scored_nodes"] != expected_nodes or len(budget["layers"]) != 8):
            raise ValueError(f"Unexpected attention budget or incomplete split coverage at epoch {epoch}")
        for layer in budget["layers"].values():
            if set(layer) != set(SPLITS):
                raise ValueError(f"Missing per-layer split observation at epoch {epoch}")
            for stats in layer.values():
                if stats["assigned_k"] != k or stats["calls"] <= 0:
                    raise ValueError(f"Unexpected applied layer budget at epoch {epoch}")
                clamped = {}
                for width, count in stats["batch_key_width_counts"].items():
                    width = int(width)
                    if width <= 0 or type(count) is not int or count <= 0:
                        raise ValueError(f"Invalid batch-width observation at epoch {epoch}")
                    key = str(min(k, width))
                    clamped[key] = clamped.get(key, 0) + count
                nodes = stats["valid_graph_node_counts"]
                if (clamped != stats["clamped_k_counts"] or sum(clamped.values()) != stats["calls"]
                        or not nodes or any(int(n) <= 0 or type(c) is not int or c <= 0 for n, c in nodes.items())
                        or max(map(int, nodes)) > max(map(int, stats["batch_key_width_counts"]))):
                    raise ValueError(f"Invalid clamped budget or node counts at epoch {epoch}")
        if (mem["metadata"]["seed"] != seed or mem["metadata"]["schedule"] != method
                or not mem["cuda_enabled"] or mem["units"] != "MiB" or mem["evaluated_splits"] != ["val", "test"]
                or not finite(mem["peak_allocated_mib"]) or not finite(mem["peak_reserved_mib"])
                or not 0 < mem["peak_allocated_mib"] <= mem["peak_reserved_mib"]):
            raise ValueError(f"Invalid GPU memory record at epoch {epoch}")
        duration = budget["epoch_seconds"]
        if not finite(duration) or duration <= 0:
            raise ValueError(f"Invalid epoch duration at epoch {epoch}")
        row = dict(epoch=epoch, scheduled_k=k, k_delta=k-planned[epoch-1] if epoch else 0,
                   epoch_seconds=duration, peak_allocated_mib=mem["peak_allocated_mib"],
                   peak_reserved_mib=mem["peak_reserved_mib"])
        for split, records in histories.items():
            metrics = records[epoch]
            if (not all(finite(metrics.get(key)) for key in ("loss", "f1", "lr"))
                    or not 0 <= metrics["f1"] <= 1 or metrics["lr"] < 0
                    or metrics.get("params") != architecture["parameter_count"]):
                raise ValueError(f"Invalid {split} metrics at epoch {epoch}")
            row[f"{split}_f1"] = metrics["f1"]
            row[f"{split}_loss"] = metrics["loss"]
        row["learning_rate"] = histories["train"][epoch]["lr"]
        epoch_rows.append(row)
    peak = read_json(attempt / "memory/run.json")
    if not peak["finished"] or not peak["complete_epoch_history"] or peak["recorded_epochs"] != list(range(EPOCHS)):
        raise ValueError("Incomplete run memory summary")
    for field in ("peak_allocated_mib", "peak_reserved_mib"):
        if peak["run_" + field] != max(row[field] for row in epoch_rows):
            raise ValueError(f"Inconsistent run {field}")
    if sum(row["epoch_seconds"] for row in epoch_rows) > timing["training_seconds"] + 0.01:
        raise ValueError("Epoch timing exceeds total training runtime")
    best = max(epoch_rows, key=lambda row: row["val_f1"])
    summary = dict(best_epoch=best["epoch"], best_val_f1=best["val_f1"], selected_test_f1=best["test_f1"],
                   final_test_f1=epoch_rows[-1]["test_f1"], training_seconds=timing["training_seconds"],
                   peak_allocated_mib=peak["run_peak_allocated_mib"], peak_reserved_mib=peak["run_peak_reserved_mib"],
                   mean_k=sum(planned)/EPOCHS, min_k=min(planned), max_k=max(planned),
                   initial_k=planned[0], final_k=planned[-1], best_epoch_k=best["scheduled_k"],
                   budget_changes=sum(a != b for a, b in zip(planned, planned[1:])),
                   oom_retry_events=timing["oom_retry_events"])
    return summary, epoch_rows


class ExperimentSession:
    def __init__(self, workspace_root, repo=ROOT, *, run_experiments=True, skip_completed=True,
                 seeds=SEEDS, device="cuda", create_zip=False):
        self.repo = Path(repo).resolve()
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.root = self.workspace_root / "results"
        self.setup = self.workspace_root / "setup_logs"
        self.dataset_dir = self.workspace_root / "datasets"
        self.venv = self.workspace_root / "environment"
        self.cache = self.setup / "cache"
        directories = (self.root, self.setup, self.dataset_dir, self.cache,
                       *(self.cache / name for name in ("keops", "cuda", "matplotlib", "pip")))
        for directory in (*directories, self.venv):
            if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
                raise SharedSetupError(f"Use a regular directory for {directory}.")
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        self.venv_python = self.venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        self.run_experiments = run_experiments
        self.skip_completed = skip_completed
        self.seeds = tuple(seeds)
        if len(set(self.seeds)) != len(self.seeds) or any(type(seed) is not int or seed not in SEEDS for seed in self.seeds):
            raise ValueError("RUN_SEEDS must contain unique seeds from 0 through 3")
        self.device = device
        self.create_zip = create_zip
        self.python = sys.executable
        self.environment = self.prepared = self.manifest = None
        self.blocked = False

    def _attempts(self, method, seed):
        return sorted((self.root / method / f"seed_{seed}" / "attempts").glob("attempt_*"))

    def _check_results_ownership(self):
        if not (self.root / "manifest.json").is_file() and any(
            item.name not in (".launch.lock", ".worker.lock") for item in self.root.iterdir()
        ):
            raise SharedSetupError(f"Existing files at {self.root} have no study manifest. Choose a new WORKSPACE_ROOT.")

    @contextmanager
    def _launch_guard(self):
        self._check_results_ownership()
        with exclusive_lock(self.root / ".launch.lock"):
            self._check_results_ownership()
            # Workers hold their own lock even if the notebook process is killed.
            with exclusive_lock(self.root / ".worker.lock"):
                for method in METHODS:
                    for seed in SEEDS:
                        for attempt in self._attempts(method, seed):
                            path = attempt / "status.json"
                            try:
                                status = read_status(path)
                            except (OSError, ValueError):
                                continue  # Export flags invalid statuses without changing their files.
                            if status["status"] == "running":
                                status.update(status="incomplete", ended_utc=utc_now(),
                                              reason="Previous launcher stopped before completion was verified")
                                atomic_json(path, status)
                                current = attempt.parent.parent / "status.json"
                                try:
                                    current_status = read_status(current)
                                except (OSError, ValueError):
                                    continue
                                if current_status.get("attempt") == attempt.name:
                                    atomic_json(current, status)
            yield

    def _setup_directory(self, name):
        return Path(tempfile.mkdtemp(prefix=name + "-", dir=self.setup))

    def _child_environment(self):
        env = os.environ.copy()
        env.update(
            KEOPS_CACHE_FOLDER=str(self.cache / "keops"),
            CUDA_CACHE_PATH=str(self.cache / "cuda"),
            MPLCONFIGDIR=str(self.cache / "matplotlib"),
            MPLBACKEND="Agg",
            PYTHONDONTWRITEBYTECODE="1",
        )
        return env

    def _worker(self, action, output, *extra):
        with (output / "stdout.log").open("a", encoding="utf-8") as log:
            return subprocess.run(
                [self.python, "-u", "-m", "study.new_schedules_worker", action,
                 "--root", str(self.root), "--output", str(output), "--device", self.device,
                 "--dataset-dir", str(self.dataset_dir), *extra],
                cwd=self.repo, env=self._child_environment(),
                stdout=log, stderr=subprocess.STDOUT, check=False,
            ).returncode

    def _install(self, log_path, cuda, repair, requirements_installed):
        with log_path.open("a", encoding="utf-8") as log:
            def install(*args):
                subprocess.run(
                    [str(self.venv_python), "-m", "pip", "--isolated", "--disable-pip-version-check",
                     "--cache-dir", str(self.cache / "pip"), "install", *args],
                    stdout=log, stderr=subprocess.STDOUT, check=True,
                )

            try:
                owner = self.venv / "kmip_worker.json"
                base = {"python": sys.executable, "torch": cuda["torch"], "cuda": cuda["cuda_build"]}
                if self.venv.exists():
                    if not owner.is_file() or read_json(owner) != base:
                        raise SharedSetupError(f"Existing environment at {self.venv} is not managed for this CUDA PyTorch kernel.")
                else:
                    self.venv.mkdir()
                    atomic_json(owner, base)
                if not self.venv_python.is_file():
                    subprocess.run(
                        [sys.executable, "-m", "venv", "--system-site-packages", str(self.venv)],
                        stdout=log, stderr=subprocess.STDOUT, check=True,
                    )
                if sys.prefix != sys.base_prefix:
                    # A nested venv inherits the base interpreter's site, not the kernel venv's site.
                    purelib = subprocess.check_output(
                        [str(self.venv_python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                        text=True, stderr=log,
                    ).strip()
                    worker_site = Path(purelib).resolve()
                    if not worker_site.is_relative_to(self.venv):
                        raise SharedSetupError("The worker package directory is outside its managed environment.")
                    kernel_paths = {sysconfig.get_path(name) for name in ("purelib", "platlib")}
                    (worker_site / "kmip_kernel_site.pth").write_text(
                        "".join(f"{path}\n" for path in sorted(kernel_paths)), encoding="utf-8")
                constraints = log_path.parent / "constraints.txt"
                constraints.write_text(f"torch=={cuda['torch']}\n", encoding="utf-8")
                component = repair["component"]
                if component == "torch-scatter":
                    major, minor, *_ = cuda["torch"].split("+")[0].split(".")
                    variant = cuda["cuda_build"].replace(".", "")
                    wheel_url = f"https://data.pyg.org/whl/torch-{major}.{minor}.0+cu{variant}.html"
                    reinstall = ["--force-reinstall"] if repair["upgrade"] else []
                    install("--no-index", "--no-deps", "--only-binary=:all:",
                            "--find-links", wheel_url, *reinstall, "torch-scatter")
                elif component != "requirements" and (repair["upgrade"] or requirements_installed):
                    upgrade = ["--upgrade", "--upgrade-strategy", "only-if-needed"] if repair["upgrade"] else []
                    install("-c", str(constraints), *upgrade, component)
                if not requirements_installed:
                    install("-c", str(constraints), "-r", str(self.repo / "study/requirements.txt"))
            except (OSError, subprocess.CalledProcessError, ValueError, SharedSetupError) as error:
                log.write(f"\n{error}\n")
                raise SharedSetupError(
                    f"Could not provision the isolated worker. Check venv support, package access and "
                    f"the matching PyG wheel for this CUDA PyTorch build. See {log_path}"
                ) from error

    def _probe_environment(self, python):
        self.python = str(python)
        output = self._setup_directory("environment")
        log_path = output / "stdout.log"
        try:
            code = self._worker("environment", output)
        except OSError as error:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"Could not start {self.python}: {error}\n")
            return output, False
        if code:
            return output, False
        self.environment = read_json(output / "environment.json")
        self.environment["repository"] = repository_identity(self.repo)
        atomic_json(output / "environment.json", self.environment)
        return output, True

    def prepare_environment(self, auto_setup=False):
        self.environment = self.prepared = self.manifest = None
        with self._launch_guard():
            if self.venv_python.is_file():
                _, verified = self._probe_environment(self.venv_python)
                if verified:
                    return
            output, verified = self._probe_environment(sys.executable)
            if verified:
                return
            base_output = output
            attempted = set()
            while auto_setup and (output / "repair.json").is_file():
                repair = read_json(output / "repair.json")
                key = (repair["component"], repair["upgrade"])
                if key in attempted:
                    break
                self._install(output / "install.log", read_json(base_output / "cuda.json"), repair,
                              requirements_installed=bool(attempted))
                attempted.add(key)
                output, verified = self._probe_environment(self.venv_python)
                if verified:
                    return
            raise SharedSetupError(
                "GPU setup could not be verified. Use a CUDA-enabled PyTorch kernel with a compatible "
                f"NVIDIA driver and KeOps build toolchain. See {output / 'stdout.log'} for the failed check."
            )

    def prepare_dataset(self):
        self.prepared = self.manifest = None
        if self.environment is None:
            raise SharedSetupError("Complete environment setup first")
        with self._launch_guard():
            output = self._setup_directory("dataset")
            if self._worker("prepare", output) != 0:
                raise SharedSetupError(f"PascalVOC preparation failed. Check the standard datasets cache, "
                                       f"mirror access and available disk space. See {output / 'stdout.log'}")
            self.prepared = read_json(output / "prepared.json")

    def initialize(self):
        self.manifest = None
        if self.prepared is None:
            raise SharedSetupError("Complete dataset preparation first")
        with self._launch_guard():
            if repository_identity(self.repo)["source_sha256"] != self.environment["repository"]["source_sha256"]:
                raise SharedSetupError("Source files changed during setup. Run the setup cells again.")
            candidate = dict(schema_version=1, environment=self.environment, **self.prepared)
            candidate["jobs"] = [{"method": method, "seed": seed, "epochs": EPOCHS,
                                  "path": f"{method}/seed_{seed}"} for method in METHODS for seed in SEEDS]
            # Record dirty status for provenance but bind identity to code content, not unrelated files.
            bound = json.loads(json.dumps(candidate))
            repo = bound["environment"]["repository"]
            repo.pop("dirty")
            repo.pop("dirty_status")
            candidate["study_id"] = hashlib.sha256(json.dumps(bound, sort_keys=True).encode()).hexdigest()
            path = self.root / "manifest.json"
            if path.exists():
                old = read_json(path)
                old_bound = json.loads(json.dumps(old))
                old_id = old_bound.pop("study_id")
                old_bound["environment"]["repository"].pop("dirty")
                old_bound["environment"]["repository"].pop("dirty_status")
                if old_id != candidate["study_id"] or old_bound != bound:
                    raise SharedSetupError("Existing study has different code, environment, data or settings. Choose a new WORKSPACE_ROOT.")
                candidate = old
            else:
                atomic_json(path, candidate)
            for name, value in (("environment.json", candidate["environment"]), ("dataset.json", candidate["dataset"])):
                metadata = self.root / name
                if not metadata.exists():
                    atomic_json(metadata, value)
                elif read_json(metadata) != value:
                    raise SharedSetupError(f"Existing metadata at {metadata} differs from the study manifest.")
            source = (self.repo / TUNED.relative_to(ROOT)).read_bytes()
            tuned = self.root / "authors_tuned_config.yaml"
            if not tuned.exists():
                with tuned.open("xb") as handle:
                    handle.write(source)
            elif tuned.read_bytes() != source:
                raise SharedSetupError(f"Existing tuned configuration at {tuned} differs from the repository.")
            for job in candidate["jobs"]:
                run = self.root / job["path"]
                run.mkdir(parents=True, exist_ok=True)
                status = run / "status.json"
                if not status.exists():
                    atomic_json(status, dict(method=job["method"], seed=job["seed"], status="pending"))
            self.manifest = candidate
            self.blocked = False

    def run_experiment(self, method, seed):
        if self.manifest is None or self.blocked:
            raise SharedSetupError("Shared setup failed or has not completed. Rerun setup before launching.")
        if method not in METHODS or type(seed) is not int or seed not in SEEDS:
            raise ValueError("Unknown experiment")
        if not self.run_experiments or seed not in self.seeds:
            return
        with self._launch_guard():
            current = repository_identity(self.repo)
            expected = self.manifest["environment"]["repository"]
            if any(current[field] != expected[field] for field in ("commit", "source_sha256")):
                self.blocked = True
                raise SharedSetupError("Repository source changed since setup")
            attempts = self._attempts(method, seed)
            if self.skip_completed:
                for attempt in reversed(attempts):
                    try:
                        validate_attempt(attempt, self.manifest, method, seed)
                    except (OSError, ValueError, KeyError, TypeError):
                        continue
                    return
            run = self.root / method / f"seed_{seed}"
            index = max((int(path.name.removeprefix("attempt_")) for path in attempts), default=0) + 1
            attempt = run / "attempts" / f"attempt_{index:06d}"
            attempt.mkdir(parents=True, exist_ok=False)
            identity = dict(method=method, seed=seed, study_id=self.manifest["study_id"])
            status = dict(status="running", identity=identity, attempt=attempt.name,
                          started_utc=utc_now(), launcher_pid=os.getpid(), hostname=socket.gethostname())
            atomic_json(attempt / "status.json", status)
            atomic_json(run / "status.json", status)
            try:
                code = self._worker("train", attempt, "--method", method, "--seed", str(seed))
                status["exit_code"] = code
                atomic_json(attempt / "status.json", status)
                if code:
                    status.update(status="failed", reason=f"Worker exited {code}. See stdout.log.")
                    if code == 2:
                        self.blocked = True
                        raise SharedSetupError(f"Shared worker prerequisite failed. See {attempt / 'stdout.log'}")
                else:
                    try:
                        summary, _ = validate_attempt(attempt, self.manifest, method, seed)
                        atomic_json(attempt / "metrics_summary.json", summary)
                        status["status"] = "completed"
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        status.update(status="incomplete", reason=str(error))
                        (attempt / "validation_error.log").write_text(traceback.format_exc(), encoding="utf-8")
            except KeyboardInterrupt:
                status.update(status="incomplete", reason="Interrupted by operator")
                raise
            except SharedSetupError:
                raise
            except Exception as error:
                status.update(status="failed", reason=repr(error))
                (attempt / "launcher_error.log").write_text(traceback.format_exc(), encoding="utf-8")
            finally:
                status["ended_utc"] = utc_now()
                atomic_json(attempt / "status.json", status)
                atomic_json(run / "status.json", status)

    def export(self):
        if self.manifest is None:
            raise SharedSetupError("Initialize the study before exporting")
        with self._launch_guard():
            logs = self.root / "export_logs"
            logs.mkdir(exist_ok=True)
            output = Path(tempfile.mkdtemp(prefix="export-", dir=logs))
            command = [self.python, "-m", "study.new_schedules_report", str(self.root)]
            if self.create_zip:
                command.append("--zip")
            with (output / "stdout.log").open("w", encoding="utf-8") as log:
                code = subprocess.run(command, cwd=self.repo, env=self._child_environment(),
                                      stdout=log, stderr=subprocess.STDOUT).returncode
            if code:
                raise SharedSetupError(f"Result export failed. See {output / 'stdout.log'}")
