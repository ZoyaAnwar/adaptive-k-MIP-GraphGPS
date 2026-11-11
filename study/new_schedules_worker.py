"""Subprocess entry points using the repository's native PascalVOC training pipeline."""
import argparse
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

from study.new_schedules_runner import (
    CONFIG_DIR, EPOCHS, METHODS, SEEDS, SPLITS, TUNED, SharedSetupError,
    atomic_json, configuration_identity, digest, exclusive_lock, read_json, repository_identity, utc_now, ROOT,
)


def cuda_environment(device):
    import platform
    import torch

    requested = torch.device(device)
    if requested.type != "cuda" or not torch.cuda.is_available():
        raise SharedSetupError("A working CUDA PyTorch installation and visible NVIDIA GPU are required")
    index = torch.cuda.current_device() if requested.index is None else requested.index
    torch.cuda.set_device(index)
    properties = torch.cuda.get_device_properties(index)
    uuid = getattr(properties, "uuid", None)
    uuid = str(uuid) if uuid is not None else None
    driver = None
    smi_id = None
    try:
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        ).splitlines()
        devices = dict(tuple(item.strip() for item in row.split(",", 1)) for row in rows)
        if uuid:
            for name, version in devices.items():
                if name.removeprefix("GPU-").lower() == uuid.removeprefix("GPU-").lower():
                    smi_id, driver = name, version
                    break
        # Driver is host-wide, even if the selected device UUID is unavailable.
        if driver is None and len(set(devices.values())) == 1:
            driver = next(iter(devices.values()))
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "driver_version": driver,
        "device": f"cuda:{index}",
        "gpu_name": properties.name,
        "gpu_uuid": uuid,
        "nvidia_smi_id": smi_id,
        "gpu_capacity_mib": properties.total_memory / 1024**2,
        "compute_capability": [properties.major, properties.minor],
    }


def package_freeze():
    return subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"],
                                   text=True, encoding="utf-8").splitlines()


@contextmanager
def dependency_probe(output, component, *, can_upgrade=True):
    try:
        yield
    except (ImportError, OSError, RuntimeError, AttributeError, ValueError) as error:
        if can_upgrade or isinstance(error, ModuleNotFoundError):
            missing = isinstance(error, ModuleNotFoundError) and error.name == component.replace("-", "_")
            atomic_json(output / "repair.json", {"component": component, "upgrade": can_upgrade and not missing})
        raise


def check_environment(output, device):
    import torch

    info = cuda_environment(device)
    atomic_json(output / "cuda.json", info)
    selected = info["device"]
    free_bytes, _ = torch.cuda.mem_get_info(selected)
    print(f"CUDA free at preflight: {free_bytes / 1024**2:.0f} MiB", flush=True)
    values = torch.ones(2, device=selected)
    indices = torch.zeros(2, dtype=torch.long, device=selected)
    with dependency_probe(output, "torch-geometric"):
        import torch_geometric
        from torch_geometric.utils import scatter

        if scatter(values, indices, reduce="sum").item() != 2:
            raise RuntimeError("PyTorch Geometric CUDA scatter returned an incorrect result")
    with dependency_probe(output, "torch-scatter"):
        import torch_scatter

        if torch_scatter.scatter_add(values, indices).item() != 2:
            raise RuntimeError("torch-scatter CUDA extension returned an incorrect result")
    with dependency_probe(output, "pykeops"):
        import pykeops
        from pykeops.torch import LazyTensor
    with dependency_probe(output, "matplotlib"):
        import matplotlib
        import matplotlib.pyplot
    with dependency_probe(output, "requirements", can_upgrade=False):
        import graphgps  # Register models, configs and loaders and validate dependencies.
        from graphgps.layer.sparse_attention_layer import SparseAttention

    # Explicitly request a GPU reduction, then exercise the same sparse attention path as training.
    points = torch.randn(4, 3, device=selected)
    squared_distances = ((LazyTensor(points[:, None, :]) - LazyTensor(points[None, :, :])) ** 2).sum(-1)
    nearest = squared_distances.argKmin(1, dim=1, backend="GPU")
    if not nearest.is_cuda or not torch.equal(nearest.flatten(), torch.arange(4, device=selected)):
        raise SharedSetupError("PyKeOps GPU nearest-neighbor probe failed")
    attention = SparseAttention(40, 10, 10, 4, 50, "none").to(selected)
    if not attention.use_keops:
        raise SharedSetupError("PyKeOps CUDA attention is unavailable")
    features = torch.randn(1, 56, 40, device=selected, requires_grad=True)
    result = attention(features)
    result.square().mean().backward()
    torch.cuda.synchronize(selected)
    if not result.is_cuda or not torch.isfinite(result).all() or features.grad is None or not torch.isfinite(features.grad).all():
        raise SharedSetupError("CUDA/KeOps preflight produced invalid values")
    info["versions"] = {
        "torch_geometric": torch_geometric.__version__,
        "torch_scatter": torch_scatter.__version__,
        "pykeops": pykeops.__version__,
        "matplotlib": matplotlib.__version__,
    }
    info["packages"] = package_freeze()
    atomic_json(output / "environment.json", info)


def check_source_configs():
    import copy
    import yaml

    tuned = yaml.safe_load(TUNED.read_text(encoding="utf-8"))
    hashes = {}
    for method in METHODS:
        source = CONFIG_DIR / f"{method}.yaml"
        expected = copy.deepcopy(tuned)
        expected["gt"]["sparse"]["k_schedule"] = method
        expected["wandb"]["use"] = False
        for flag in ("enable_ckpt", "auto_resume", "ckpt_best", "ckpt_clean"):
            expected["train"][flag] = False
        if yaml.safe_load(source.read_text(encoding="utf-8")) != expected:
            raise SharedSetupError(f"{source.name} differs from the tuned YAML beyond the approved settings")
        hashes[method] = digest(source)
    return hashes


def configure(method, seed, output, device, dataset_dir=None):
    import torch
    import graphgps
    from torch_geometric import seed_everything
    from torch_geometric.graphgym.config import cfg, set_cfg
    from torch_geometric.graphgym.utils.epoch import is_eval_epoch

    set_cfg(cfg)
    cfg.merge_from_file(str(CONFIG_DIR / f"{method}.yaml"))
    cfg.seed = seed
    cfg.run_id = seed
    cfg.accelerator = "cuda"
    cfg.device = device
    cfg.out_dir = str(output)
    cfg.run_dir = str(output)
    cfg.dataset.dir = str(Path(dataset_dir or cfg.dataset.dir).resolve())
    if (cfg.optim.max_epoch != EPOCHS or cfg.metric_best != "f1" or cfg.metric_agg != "argmax"
            or cfg.train.mode != "custom" or cfg.train.eval_period != 1
            or any(not is_eval_epoch(epoch) for epoch in range(EPOCHS))
            or cfg.study.enabled or cfg.wandb.use or cfg.pretrained.dir
            or cfg.gt.sparse.random_attention
            or any(getattr(cfg.train, flag) for flag in ("enable_ckpt", "auto_resume", "ckpt_best", "ckpt_clean"))):
        raise SharedSetupError("Resolved configuration does not meet the checkpoint-free 200-epoch protocol")
    torch.set_num_threads(cfg.num_threads)
    seed_everything(seed)
    return cfg


def load_pascalvoc(cfg):
    from graphgps.adaptive_k.run_identity import dataset_fingerprint
    from graphgps.utils import my_create_loader

    loaders = my_create_loader()
    if cfg.dataset.format != "PyG-VOCSuperpixels" or len(loaders) != len(SPLITS):
        raise SharedSetupError("Expected the standard train, validation and test PascalVOC splits")
    cache = Path(cfg.dataset.dir) / "VOCSuperpixels" / f"slic_compactness_{cfg.dataset.slic_compactness}" / cfg.dataset.name
    splits = {}
    seen = set()
    for split, loader in zip(SPLITS, loaders):
        dataset = loader.dataset
        indices = [int(index) for index in dataset.indices()]
        if not indices or len(set(indices)) != len(indices) or seen.intersection(indices):
            raise SharedSetupError(f"Empty, repeated or overlapping graph indices in {split}")
        seen.update(indices)
        splits[split] = dict(graphs=len(dataset), nodes=sum(int(graph.num_nodes) for graph in dataset),
                             source_indices=indices, sha256=dataset_fingerprint(dataset))
        if splits[split]["nodes"] <= 0 or not (cache / "processed" / f"{split}.pt").is_file():
            raise SharedSetupError(f"Incomplete PascalVOC cache or empty split: {split}")
    if seen != set(range(sum(s["graphs"] for s in splits.values()))):
        raise SharedSetupError("Expected the standard exhaustive PascalVOC split")
    return loaders, dict(format=cfg.dataset.format, name=cfg.dataset.name, split_index=cfg.dataset.split_index,
                         slic_compactness=cfg.dataset.slic_compactness, cache_path=str(cache.resolve()),
                         dataset_dir=cfg.dataset.dir, splits=splits)


def build_model(cfg, to_device):
    from main import find_max_hidden_dim, set_hidden_dim
    from torch_geometric.graphgym.model_builder import create_model
    from torch_geometric.graphgym.utils.comp_budget import params_count

    if cfg.model.auto_satisfy_param_budget:
        set_hidden_dim(cfg, find_max_hidden_dim(cfg, cfg.model.param_budget))
    model = create_model(to_device=to_device)
    cfg.params = int(params_count(model))
    architecture = dict(hidden_dimension=cfg.gt.dim_hidden, sparse_value_dimension=cfg.gt.sparse.val_dim,
                        parameter_count=cfg.params, layers=cfg.gt.layers, heads=cfg.gt.n_heads,
                        layers_post_mp=cfg.gnn.layers_post_mp)
    if (cfg.model.type != "GPSModel" or cfg.gt.layer_type != "CustomGatedGCN+SparseAttention"
            or cfg.gt.layers != 8 or cfg.gt.n_heads != 4 or cfg.gnn.layers_post_mp != 1
            or not cfg.model.auto_satisfy_param_budget or not 0 < cfg.params <= cfg.model.param_budget
            or cfg.gt.dim_hidden != cfg.gnn.dim_inner or cfg.gt.dim_hidden != 4 * cfg.gt.sparse.val_dim):
        raise SharedSetupError("Unexpected tuned architecture after automatic parameter-budget sizing")
    return model, architecture


def prepare(output, device, dataset_dir=None):
    import yaml
    from graphgps.adaptive_k.train_schedule import k_for_epoch

    hashes = check_source_configs()
    cfg = configure(METHODS[0], 0, output, device, dataset_dir)
    _, dataset = load_pascalvoc(cfg)
    _, architecture = build_model(cfg, to_device=False)
    trajectories = {}
    effective_hashes = {}
    for method in METHODS:
        cfg.gt.sparse.k_schedule = method
        trajectory = [k_for_epoch(cfg.gt.sparse, epoch, EPOCHS) for epoch in range(EPOCHS)]
        if (trajectory[0] != 50 or trajectory[-1] != (10 if method == METHODS[0] else 15)
                or min(trajectory) != 10 or max(trajectory) != 50
                or (method == METHODS[1] and trajectory[140] != 10)):
            raise SharedSetupError(f"Invalid planned trajectory for {method}")
        trajectories[method] = trajectory
        effective_hashes[method] = configuration_identity(yaml.safe_load(cfg.dump()))
        (output / f"{method}_effective_config.yaml").write_text(cfg.dump(), encoding="utf-8")
    atomic_json(output / "prepared.json", dict(dataset=dataset, architecture=architecture, trajectories=trajectories,
                tuned_config_sha256=digest(TUNED), config_sha256=hashes, effective_config_sha256=effective_hashes,
                rounding="Python int(round(value)), nearest integer with ties to even"))


class MetricsObserver:
    """Read-only training-loop observer. All checkpoints are disabled by config."""
    def __init__(self, model, output, trajectory, expected_nodes):
        from graphgps.layer.sparse_attention_layer import SparseAttention

        self.output = output
        self.trajectory = trajectory
        self.expected_nodes = expected_nodes
        self.layers = {name: module for name, module in model.named_modules() if isinstance(module, SparseAttention)}
        if len(self.layers) != 8:
            raise SharedSetupError("Expected eight sparse attention layers")
        self.hooks = [model.register_forward_pre_hook(self._read_batch)]
        for name, layer in self.layers.items():
            self.hooks.append(layer.register_forward_pre_hook(
                lambda module, inputs, name=name: self._observe_layer(name, module, inputs[0])))
        self.epoch = None

    def _read_batch(self, module, inputs):
        batch = inputs[0]
        self.split = batch.split
        if self.split not in SPLITS or self.epoch is None:
            raise RuntimeError("Unexpected forward outside a profiled train/val/test epoch")
        # One transfer per forward, shared across all eight attention hooks.
        self.node_counts = [int(count) for count in batch.ptr.diff().detach().cpu().tolist()]

    def _observe_layer(self, name, module, inputs):
        width = int(inputs.shape[-2])
        assigned = int(module.k)
        if (module.random_attention or assigned != self.trajectory[self.epoch]
                or len(self.node_counts) != inputs.shape[0] or max(self.node_counts) != width):
            raise ValueError(f"Unexpected applied k or batch width in {name} at epoch {self.epoch}")
        stats = self.observed[name].setdefault(self.split, dict(
            assigned_k=assigned, calls=0, batch_key_width_counts=Counter(),
            clamped_k_counts=Counter(), valid_graph_node_counts=Counter()))
        stats["calls"] += 1
        stats["batch_key_width_counts"][width] += 1
        stats["clamped_k_counts"][min(assigned, width)] += 1
        stats["valid_graph_node_counts"].update(self.node_counts)

    def validate_dataset(self, dataset):
        # All three loaded splits were checked against the manifest before the loop.
        pass

    def prepare(self, start_epoch):
        if start_epoch != 0:
            raise ValueError("Each attempt must start at epoch zero")

    def begin_epoch(self, epoch):
        self.started = time.perf_counter()
        self.epoch = epoch
        self.observed = {name: {} for name in self.layers}
        self.scored_nodes = {}

    def phase(self, name, seconds):
        pass

    def stage_recovery(self, epoch, model, optimizer, scheduler):
        pass

    def end_epoch(self, epoch, model, optimizer, scheduler, best_epoch):
        elapsed = time.perf_counter() - self.started  # Called after profiler CUDA synchronization.
        row = dict(epoch=epoch, scheduled_k=self.trajectory[epoch], scored_nodes=self.scored_nodes,
                   epoch_seconds=elapsed, layers=self.observed)
        with (self.output / "attention_budgets.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        self.epoch = None
        if self.scored_nodes != self.expected_nodes:
            raise RuntimeError(f"Dropped or repeated scored nodes at epoch {epoch}: "
                               f"observed {self.scored_nodes}, expected {self.expected_nodes}")

    def close(self):
        for hook in self.hooks:
            hook.remove()


class ObservedLogger:
    """Retain the native logger and record the number of nodes it actually scored."""
    def __init__(self, logger, observer, split):
        self.logger = logger
        self.observer = observer
        self.split = split

    def __getattr__(self, name):
        return getattr(self.logger, name)

    def write_epoch(self, epoch):
        nodes = int(self.logger._size_current)
        result = self.logger.write_epoch(epoch)
        self.observer.scored_nodes[self.split] = nodes
        return result


@contextmanager
def utilization_telemetry(output, gpu_id):
    if gpu_id is None:
        yield
        return
    stop = threading.Event()

    def collect():
        with (output / "gpu_telemetry.jsonl").open("a", encoding="utf-8") as log:
            while not stop.is_set():
                row = dict(timestamp_utc=utc_now(), gpu_uuid=gpu_id, scope="whole_device")
                try:
                    raw = subprocess.check_output(
                        ["nvidia-smi", "--id", gpu_id, "--query-gpu=utilization.gpu,memory.used",
                         "--format=csv,noheader,nounits"],
                        text=True, timeout=10, stderr=subprocess.DEVNULL,
                    )
                    utilization, memory = map(float, raw.strip().split(","))
                    row.update(utilization_percent=utilization, device_memory_used_mib=memory)
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    row["sample_error"] = type(error).__name__
                log.write(json.dumps(row) + "\n")
                log.flush()
                stop.wait(60)

    sampler = threading.Thread(target=collect, daemon=True)
    sampler.start()
    try:
        yield
    finally:
        stop.set()
        sampler.join(timeout=12)


def train(root, output, method, seed, device, dataset_dir=None):
    import torch
    import yaml
    from graphgps.logger import create_logger
    from graphgps.train.custom_train import custom_train
    from main import new_optimizer_config, new_scheduler_config, require_keops_for_research
    from torch_geometric.graphgym.optim import create_optimizer, create_scheduler

    manifest = read_json(root / "manifest.json")
    identity = dict(method=method, seed=seed, study_id=manifest["study_id"])
    status = read_json(output / "status.json")
    if status["status"] != "running" or status["identity"] != identity:
        raise RuntimeError("Attempt is no longer active")
    environment = manifest["environment"]
    try:
        actual = cuda_environment(device)
        if any(environment[key] != value for key, value in actual.items()) or package_freeze() != environment["packages"]:
            raise SharedSetupError("Runtime software or GPU differs from the prepared environment")
        repository = repository_identity(ROOT)
        if any(repository[key] != environment["repository"][key] for key in ("commit", "source_sha256")):
            raise SharedSetupError("Repository differs from the prepared source")
        if check_source_configs() != manifest["config_sha256"]:
            raise SharedSetupError("Source configuration changed after setup")
        cfg = configure(method, seed, output, actual["device"], dataset_dir)
        loaders, dataset = load_pascalvoc(cfg)
        if dataset != manifest["dataset"]:
            raise SharedSetupError("Dataset content, location or split order changed after setup")
    except Exception as error:
        raise SharedSetupError("Shared setup validation failed. Inspect this attempt's stdout.log.") from error
    model, architecture = build_model(cfg, to_device=True)
    if architecture != manifest["architecture"]:
        raise SharedSetupError("Effective architecture differs from setup")
    require_keops_for_research(model)
    optimizer = create_optimizer(model.parameters(), new_optimizer_config(cfg))
    scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
    shutil.copyfile(CONFIG_DIR / f"{method}.yaml", output / "source_config.yaml")
    effective_yaml = output / "effective_config.yaml"
    effective_yaml.write_text(cfg.dump(), encoding="utf-8")
    effective = yaml.safe_load(cfg.dump())
    if configuration_identity(effective) != manifest["effective_config_sha256"][method]:
        raise SharedSetupError("Resolved training configuration differs from setup")
    atomic_json(output / "effective_config.json", effective)
    atomic_json(output / "provenance.json", dict(identity=identity, environment=environment, repository=repository,
                dataset=dataset, architecture=architecture, started_utc=utc_now(), effective_yaml_sha256=digest(effective_yaml),
                rng=dict(seed=seed, initializer="torch_geometric.seed_everything",
                         deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                         cudnn_deterministic=torch.backends.cudnn.deterministic,
                         cudnn_benchmark=torch.backends.cudnn.benchmark,
                         float32_matmul_precision=torch.get_float32_matmul_precision(),
                         torch_threads=torch.get_num_threads())))
    observer = MetricsObserver(model, output, manifest["trajectories"][method],
                               {split: dataset["splits"][split]["nodes"] for split in SPLITS})
    loggers = [ObservedLogger(logger, observer, split) for logger, split in zip(create_logger(), SPLITS)]
    try:
        with utilization_telemetry(output, actual["nvidia_smi_id"]):
            torch.cuda.synchronize(actual["device"])
            started = time.perf_counter()
            custom_train(loggers, loaders, model, optimizer, scheduler, study_observer=observer)
            torch.cuda.synchronize(actual["device"])
            elapsed = time.perf_counter() - started
    finally:
        observer.close()
    with (output / "stdout.log").open(encoding="utf-8", errors="replace") as log:
        retries = sum("OOM error occurred" in line and "Retrying" in line for line in log)
    atomic_json(output / "training.json", dict(identity=identity, training_seconds=elapsed,
                ended_utc=utc_now(), oom_retry_events=retries,
                timing_scope="synchronized custom training loop including train, validation and test"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("environment", "prepare", "train"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    try:
        with exclusive_lock(args.root / ".worker.lock"):
            if args.action == "environment":
                check_environment(args.output, args.device)
            elif args.action == "prepare":
                prepare(args.output, args.device, args.dataset_dir)
            else:
                if args.method is None or args.seed is None:
                    parser.error("train requires --method and --seed")
                train(args.root, args.output, args.method, args.seed, args.device, args.dataset_dir)
    except SharedSetupError:
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
