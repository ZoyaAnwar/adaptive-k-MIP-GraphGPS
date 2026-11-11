"""Export verified PascalVOC results from native logs, without loading models."""
import argparse
import csv
import os
from pathlib import Path
import statistics
import tempfile
import zipfile

from study.new_schedules_runner import METHODS, SEEDS, atomic_json, read_json, read_status, utc_now, validate_attempt


LABELS = {"mirrored_exponential": "Mirrored exponential", "decrease_increase": "Decrease / increase"}
METRICS = ("best_epoch", "best_val_f1", "selected_test_f1", "final_test_f1", "training_seconds",
           "peak_allocated_mib", "peak_reserved_mib", "mean_k", "min_k", "max_k", "initial_k", "final_k",
           "best_epoch_k", "budget_changes", "oom_retry_events")
EPOCH_FIELDS = ("method", "seed", "attempt", "epoch", "scheduled_k", "k_delta", "learning_rate",
                "epoch_seconds", "peak_allocated_mib", "peak_reserved_mib",
                "train_f1", "train_loss", "val_f1", "val_loss", "test_f1", "test_loss")


def collect(root, manifest):
    results, epochs, statuses = [], [], []
    for method in METHODS:
        for seed in SEEDS:
            attempts = sorted((root / method / f"seed_{seed}" / "attempts").glob("attempt_*"))
            records = []
            accepted = None
            for attempt in attempts:
                record = {"attempt": attempt.name, "path": attempt.relative_to(root).as_posix()}
                try:
                    status = read_status(attempt / "status.json")
                    record.update(status)
                except (OSError, ValueError, TypeError) as error:
                    record.update(status="incomplete", reason=f"Missing or invalid status: {error}")
                    records.append(record)
                    continue
                if record["status"] == "completed":
                    try:
                        summary, history = validate_attempt(attempt, manifest, method, seed)
                        accepted = attempt.name, summary, history
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        record.update(status="incomplete", reason=f"Validation failed: {error}")
                records.append(record)
            latest = records[-1] if records else {"status": "pending"}
            row = dict(method=method, seed=seed, status=latest["status"], latest_attempt_status=latest["status"],
                       accepted_attempt=None, reason=latest.get("reason"))
            if accepted:
                name, summary, history = accepted
                row.update(summary, status="completed", accepted_attempt=name)
                epochs.extend(dict(method=method, seed=seed, attempt=name, **epoch) for epoch in history)
            results.append(row)
            statuses.append(dict(method=method, seed=seed, status=row["status"],
                                 accepted_attempt=row["accepted_attempt"], attempts=records))
    return results, epochs, statuses


def aggregate(results):
    groups = []
    for method in METHODS:
        completed = [row for row in results if row["method"] == method and row["status"] == "completed"]
        row = dict(method=method, completed_seeds=len(completed), planned_seeds=len(SEEDS),
                   status="complete" if len(completed) == len(SEEDS) else "incomplete")
        for key in METRICS:
            values = [result[key] for result in completed]
            row[f"{key}_mean"] = statistics.mean(values) if values else None
            row[f"{key}_sample_sd"] = statistics.stdev(values) if len(values) > 1 else None
        groups.append(row)
    return groups


def csv_file(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figures(directory, manifest, results, epochs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir()
    colors = dict(zip(METHODS, ("#0072B2", "#D55E00")))

    def save(fig, filename):
        fig.tight_layout()
        fig.savefig(directory / f"{filename}.png", dpi=250)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for method in METHODS:
        ax.step(range(len(manifest["trajectories"][method])), manifest["trajectories"][method],
                where="post", color=colors[method], label=LABELS[method])
    ax.set(xlabel="Epoch", ylabel="Scheduled k")
    ax.legend()
    save(fig, "k_schedules")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for split, ax in zip(("train", "val"), axes):
        for method in METHODS:
            for seed in SEEDS:
                rows = [row for row in epochs if row["method"] == method and row["seed"] == seed]
                if rows:
                    ax.plot([row["epoch"] for row in rows], [row[f"{split}_f1"] for row in rows],
                            color=colors[method], alpha=0.68, label=f"{LABELS[method]}, seed {seed}")
        ax.set(xlabel="Epoch", ylabel=f"{'Training' if split == 'train' else 'Validation'} F1")
        if ax.lines:
            ax.legend(fontsize=7)
        else:
            ax.text(.5, .5, "No completed runs", ha="center", transform=ax.transAxes)
    save(fig, "learning_curves")

    for fields, ylabel, filename in [
        (("selected_test_f1",), "Test F1 at best validation epoch", "validation_selected_test_f1"),
        (("training_seconds",), "Training runtime (s)", "training_runtime"),
        (("peak_allocated_mib", "peak_reserved_mib"), "Peak CUDA memory (MiB)", "peak_gpu_memory"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4))
        labels = []
        legend_seen = set()
        for index, row in enumerate(results):
            labels.append(f"{LABELS[row['method']]}\nseed {row['seed']}")
            if row["status"] != "completed":
                continue
            for offset, field in enumerate(fields):
                label = {"peak_allocated_mib": "Allocated", "peak_reserved_mib": "Reserved"}.get(field)
                delta = (offset - 0.5) * 0.32 if len(fields) == 2 else 0
                ax.bar(index + delta, row[field], width=.31 if len(fields) == 2 else .62,
                       color=colors[row["method"]], alpha=1 if offset == 0 else .45,
                       hatch=None if offset == 0 else "//",
                       label=label if label and label not in legend_seen else None)
                legend_seen.add(label)
        ax.set_xticks(range(len(labels)), labels, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        if not ax.patches:
            ax.text(.5, .5, "No completed runs", ha="center", transform=ax.transAxes)
        elif len(fields) == 2:
            ax.legend()
        save(fig, filename)


def export(root, create_zip=False):
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    results, epochs, statuses = collect(root, manifest)
    groups = aggregate(results)
    summary = root / "summary"
    summary.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".summary-", dir=root) as temporary:
        staging = Path(temporary)
        csv_file(staging / "per_seed_results.csv",
                 ["method", "seed", "status", "latest_attempt_status", "accepted_attempt", "reason", *METRICS],
                 results)
        csv_file(staging / "aggregate_results.csv", list(groups[0]), groups)
        csv_file(staging / "epoch_metrics.csv", EPOCH_FIELDS, epochs)
        csv_file(staging / "k_schedules.csv", ("method", "epoch", "planned_k", "k_delta"),
                 (dict(method=method, epoch=epoch, planned_k=k,
                       k_delta=k-trajectory[epoch-1] if epoch else 0)
                  for method, trajectory in manifest["trajectories"].items()
                  for epoch, k in enumerate(trajectory)))
        save_figures(staging / "figures", manifest, results, epochs)
        for source in sorted(staging.rglob("*")):
            if source.is_file():
                target = summary / source.relative_to(staging)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
    atomic_json(summary / "execution_status.json",
                dict(study_id=manifest["study_id"], exported_utc=utc_now(),
                     completed_runs=sum(row["status"] == "completed" for row in results),
                     planned_runs=len(METHODS)*len(SEEDS), jobs=statuses))
    if create_zip:
        stamp = utc_now().replace(":", "").replace("+", "_")
        archive = root / f"results_{stamp}.zip"
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(root)
                if (path.is_file() and not path.is_symlink() and path.suffix != ".zip"
                        and path.name not in (".launch.lock", ".worker.lock")):
                    zipped.write(path, Path(root.name) / relative)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_root")
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()
    export(args.study_root, args.zip)
