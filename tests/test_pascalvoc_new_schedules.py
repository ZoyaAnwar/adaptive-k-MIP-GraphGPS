"""Synthetic execution and real schedule tests for the standalone PascalVOC notebook."""
import ast
import copy
import hashlib
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

from study import new_schedules_runner as runner
from study import new_schedules_worker as worker
from study import new_schedules_report as report
from test_k_schedule import Attention, config, scheduler


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def prepare_fixture():
    return dict(
        dataset=dict(format="PyG-VOCSuperpixels", splits={split: {"nodes": 3} for split in runner.SPLITS}),
        architecture=dict(hidden_dimension=80, parameter_count=490000),
        trajectories={method: [scheduler.k_for_epoch(config(method), epoch, 200) for epoch in range(200)]
                      for method in runner.METHODS},
        config_sha256=worker.check_source_configs(),
    )


def effective_config(method, seed):
    source = yaml.safe_load((runner.CONFIG_DIR / f"{method}.yaml").read_text(encoding="utf-8"))
    source["seed"] = seed
    source["gt"]["dim_hidden"] = 80
    source["params"] = 490000
    return source


def populate(attempt, manifest, method, seed, epochs=200):
    identity = dict(study_id=manifest["study_id"], method=method, seed=seed)
    shutil.copyfile(runner.CONFIG_DIR / f"{method}.yaml", attempt / "source_config.yaml")
    effective = effective_config(method, seed)
    runner.atomic_json(attempt / "effective_config.json", effective)
    effective_path = attempt / "effective_config.yaml"
    effective_path.write_text(yaml.safe_dump(effective), encoding="utf-8")
    runner.atomic_json(attempt / "provenance.json",
                       dict(identity=identity, dataset=manifest["dataset"], environment=manifest["environment"],
                            rng={"seed": seed}, effective_yaml_sha256=runner.digest(effective_path)))
    runner.atomic_json(attempt / "training.json", dict(identity=identity, training_seconds=205., oom_retry_events=0))
    (attempt / "stdout.log").write_text("Synthetic native log\n", encoding="utf-8")
    for split in runner.SPLITS:
        jsonl(attempt / split / "stats.json",
              [dict(epoch=e, loss=.5, lr=.001, params=490000,
                    f1=(.6 if e in (10, 30) else .5) if split == "val" else .4+e/1000)
               for e in range(epochs)])
    memory, budgets = [], []
    for epoch, k in enumerate(manifest["trajectories"][method][:epochs]):
        stats = dict(assigned_k=k, calls=1, batch_key_width_counts={"2": 1},
                     clamped_k_counts={"2": 1}, valid_graph_node_counts={"1": 1, "2": 1})
        budgets.append(dict(epoch=epoch, scheduled_k=k, epoch_seconds=1.,
                            scored_nodes={split: 3 for split in runner.SPLITS},
                            layers={str(i): {split: stats for split in runner.SPLITS} for i in range(8)}))
        memory.append(dict(epoch=epoch, k=k, metadata=dict(seed=seed, schedule=method),
                           cuda_enabled=True, units="MiB", evaluated_splits=["val", "test"],
                           peak_allocated_mib=100., peak_reserved_mib=120.))
    jsonl(attempt / "attention_budgets.jsonl", budgets)
    jsonl(attempt / "memory/epochs.jsonl", memory)
    runner.atomic_json(attempt / "memory/run.json",
                       dict(finished=True, complete_epoch_history=True, recorded_epochs=list(range(epochs)),
                            run_peak_allocated_mib=100., run_peak_reserved_mib=120.))


class NotebookWorkflowTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.session = runner.ExperimentSession(temporary.name)
        self.repo = dict(commit="fixture", source_sha256="fixture", dirty=False, dirty_status="")
        mocked = patch.object(runner, "repository_identity", return_value=self.repo)
        mocked.start()
        self.addCleanup(mocked.stop)
        self.session.environment = {"repository": self.repo, "packages": []}
        self.session.prepared = prepare_fixture()
        self.session.prepared["effective_config_sha256"] = {
            method: runner.configuration_identity(effective_config(method, 0)) for method in runner.METHODS
        }
        self.session.initialize()

    def launch(self, method, seed, code=0, epochs=200):
        def child(action, output, *args):
            self.assertEqual(action, "train")
            populate(output, self.session.manifest, method, seed, epochs)
            return code

        with patch.object(self.session, "_worker", side_effect=child):
            self.session.run_experiment(method, seed)
        return self.session._attempts(method, seed)[-1]

    def test_manifest_and_skip_completed(self):
        jobs = self.session.manifest["jobs"]
        self.assertEqual([(j["method"], j["seed"]) for j in jobs],
                         [(m, s) for m in runner.METHODS for s in runner.SEEDS])
        self.assertTrue(all(j["epochs"] == 200 for j in jobs))
        self.assertTrue(all(runner.read_json(self.session.root / j["path"] / "status.json")["status"] == "pending"
                            for j in jobs))
        method = runner.METHODS[0]
        attempt = self.launch(method, 0)
        with patch.object(self.session, "_worker") as child:
            self.session.run_experiment(method, 0)
        child.assert_not_called()
        self.assertEqual(len(self.session._attempts(method, 0)), 1)
        summary = runner.read_json(attempt / "metrics_summary.json")
        self.assertEqual(summary["best_epoch"], 10)
        self.assertAlmostEqual(summary["selected_test_f1"], .41)
        self.assertAlmostEqual(summary["final_test_f1"], .599)
        self.assertFalse(list(attempt.rglob("*.ckpt")))

    def test_failure_continues_and_retry_preserves_epoch_zero_histories(self):
        method = runner.METHODS[0]
        failure = self.launch(method, 0, code=1, epochs=3)
        failed_log = (failure / "train/stats.json").read_bytes()
        self.launch(method, 1)
        retry = self.launch(method, 0)
        self.assertNotEqual(failure, retry)
        self.assertEqual((failure / "train/stats.json").read_bytes(), failed_log)
        self.assertEqual(runner.read_json(failure / "status.json")["status"], "failed")
        self.assertEqual(runner.read_epochs(retry / "train/stats.json")[0]["epoch"], 0)

    def test_incomplete_run_and_nonfinite_metrics_never_enter_aggregate(self):
        method = runner.METHODS[0]
        partial = self.launch(method, 0, epochs=199)
        self.assertEqual(runner.read_json(partial / "status.json")["status"], "incomplete")
        complete = self.launch(method, 1)
        path = complete / "test/stats.json"
        rows = runner.read_epochs(path)
        rows[5]["f1"] = float("nan")
        jsonl(path, rows)
        results, _, _ = report.collect(self.session.root, self.session.manifest)
        self.assertTrue(all(row["status"] != "completed" for row in results))
        self.assertEqual(report.aggregate(results)[0]["completed_seeds"], 0)

    def test_inconsistent_effective_config_and_coverage_invalidate_run(self):
        method = runner.METHODS[0]
        attempt = self.launch(method, 0)
        path = attempt / "effective_config.json"
        value = runner.read_json(path)
        value["optim"]["base_lr"] = .01
        runner.atomic_json(path, value)
        with self.assertRaisesRegex(ValueError, "Effective configuration"):
            runner.validate_attempt(attempt, self.session.manifest, method, 0)
        runner.atomic_json(path, effective_config(method, 0))
        budget = attempt / "attention_budgets.jsonl"
        rows = runner.read_epochs(budget)
        rows[0]["scored_nodes"]["test"] = 1
        jsonl(budget, rows)
        with self.assertRaisesRegex(ValueError, "coverage"):
            runner.validate_attempt(attempt, self.session.manifest, method, 0)

    def test_checkpoint_files_are_rejected(self):
        method = runner.METHODS[0]
        attempt = self.launch(method, 0)
        (attempt / "unexpected.ckpt").write_text("fixture")
        with self.assertRaisesRegex(ValueError, "Unexpected checkpoint"):
            runner.validate_attempt(attempt, self.session.manifest, method, 0)

    def test_orphan_worker_and_concurrent_launch_are_blocked(self):
        for lock in (".launch.lock", ".worker.lock"):
            with runner.exclusive_lock(self.session.root / lock):
                with patch.object(self.session, "_worker") as child:
                    with self.assertRaises(runner.SharedSetupError):
                        self.session.run_experiment(runner.METHODS[0], 0)
                child.assert_not_called()

    def test_interrupted_attempt_is_marked_incomplete(self):
        method = runner.METHODS[0]
        attempt = self.launch(method, 0, code=1, epochs=1)
        status = runner.read_json(attempt / "status.json")
        status["status"] = "running"
        runner.atomic_json(attempt / "status.json", status)
        self.session.initialize()
        self.assertEqual(runner.read_json(attempt / "status.json")["status"], "incomplete")

    def test_invalid_attempt_status_does_not_block_other_runs_or_export(self):
        method = runner.METHODS[0]
        first = self.launch(method, 0, code=1, epochs=1)
        runner.atomic_json(first / "status.json", [])
        self.launch(method, 1)
        results, _, _ = report.collect(self.session.root, self.session.manifest)
        self.assertEqual(results[0]["status"], "incomplete")
        self.assertIn("invalid status", results[0]["reason"])
        self.assertEqual(results[1]["status"], "completed")
        self.assertEqual(runner.read_json(first / "status.json"), [])
        retry = self.launch(method, 0)
        self.assertNotEqual(retry, first)

    def test_shared_failure_blocks_other_launches(self):
        with patch.object(self.session, "_worker", return_value=2) as child:
            with self.assertRaises(runner.SharedSetupError):
                self.session.run_experiment(runner.METHODS[0], 0)
            with self.assertRaises(runner.SharedSetupError):
                self.session.run_experiment(runner.METHODS[0], 1)
        self.assertEqual(child.call_count, 1)

    def test_export_partial_study_and_archive(self):
        self.launch(runner.METHODS[0], 0)
        self.launch(runner.METHODS[1], 3, code=1, epochs=4)
        report.export(self.session.root, create_zip=True)
        output = self.session.root / "summary"
        self.assertEqual(runner.read_json(output / "execution_status.json")["completed_runs"], 1)
        for name in ("per_seed_results.csv", "aggregate_results.csv", "epoch_metrics.csv", "k_schedules.csv"):
            self.assertTrue((output / name).is_file())
        self.assertEqual(len((output / "epoch_metrics.csv").read_text().splitlines()), 201)
        self.assertEqual(len(list((output / "figures").glob("*.png"))), 5)
        self.assertEqual(len(list(self.session.root.glob("*.zip"))), 1)


class NativeIntegrationTest(unittest.TestCase):
    def test_dataset_preparation_calls_the_existing_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "VOCSuperpixels/slic_compactness_30/edge_wt_region_boundary/processed"
            cache.mkdir(parents=True)
            loaders = []
            for index, split in enumerate(runner.SPLITS):
                (cache / f"{split}.pt").touch()
                class Split(list):
                    def indices(self):
                        return self.source_indices
                dataset = Split([SimpleNamespace(num_nodes=index+1)])
                dataset.source_indices = [index]
                loaders.append(SimpleNamespace(dataset=dataset))
            utils = ModuleType("graphgps.utils")
            utils.my_create_loader = Mock(return_value=loaders)
            identity = ModuleType("graphgps.adaptive_k.run_identity")
            identity.dataset_fingerprint = Mock(side_effect=lambda split: f"split-{split.indices()[0]}")
            cfg = SimpleNamespace(dataset=SimpleNamespace(dir=temporary, format="PyG-VOCSuperpixels",
                                   name="edge_wt_region_boundary", slic_compactness=30, split_index=0))
            with patch.dict(sys.modules, {"graphgps.utils": utils, "graphgps.adaptive_k.run_identity": identity}):
                actual, metadata = worker.load_pascalvoc(cfg)
            self.assertIs(actual, loaders)
            utils.my_create_loader.assert_called_once()
            self.assertEqual(metadata["cache_path"], str(cache.parent.resolve()))
            self.assertEqual([metadata["splits"][split]["nodes"] for split in runner.SPLITS], [1, 2, 3])

    def test_observer_distinguishes_budget_key_width_and_valid_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            observer = worker.MetricsObserver.__new__(worker.MetricsObserver)
            observer.output = Path(temporary)
            observer.trajectory = [50] * 200
            observer.layers = {"layer": None}
            observer.expected_nodes = {"train": 6}
            observer.begin_epoch(0)
            observer.split, observer.node_counts = "train", [2, 4]
            observer._observe_layer("layer", SimpleNamespace(k=50, random_attention=False),
                                    SimpleNamespace(shape=(2, 4, 72)))
            observer.scored_nodes = {"train": 6}
            observer.end_epoch(0, None, None, None, 0)
            row = json.loads((Path(temporary) / "attention_budgets.jsonl").read_text())
            stats = row["layers"]["layer"]["train"]
            self.assertEqual(row["scheduled_k"], 50)
            self.assertEqual(stats["assigned_k"], 50)
            self.assertEqual(stats["batch_key_width_counts"], {"4": 1})
            self.assertEqual(stats["clamped_k_counts"], {"4": 1})
            self.assertEqual(stats["valid_graph_node_counts"], {"2": 1, "4": 1})

    def test_native_loop_evaluates_all_epochs_and_never_calls_checkpoint_functions(self):
        # Compile the actual loop with CPU fixture callbacks. No model training occurs.
        path = runner.ROOT / "graphgps/train/custom_train.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "custom_train")
        function.decorator_list = []
        for method in runner.METHODS:
            raw = yaml.safe_load((runner.CONFIG_DIR / f"{method}.yaml").read_text(encoding="utf-8"))
            flags = raw["train"]
            cfg = SimpleNamespace(
                train=SimpleNamespace(**flags), gt=SimpleNamespace(sparse=config(method)),
                optim=SimpleNamespace(max_epoch=200, scheduler="cosine_with_warmup", batch_accumulation=1),
                wandb=SimpleNamespace(use=False), metric_best="f1", metric_agg="argmax",
                dataset=SimpleNamespace(name="fixture", format="fixture", split_index=0), seed=0, run_dir="unused",
            )
            model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="cpu")]),
                                    modules=lambda: iter([Attention()]))
            checkpoints = Mock(side_effect=AssertionError("Checkpoint called"))
            evaluate = Mock()
            profiler = Mock()
            loggers = [Mock(write_epoch=lambda epoch: dict(epoch=epoch, loss=.5, f1=.4)) for _ in range(3)]
            namespace = dict(cfg=cfg, logging=Mock(), time=time, np=np, train_epoch=Mock(),
                             eval_epoch=evaluate, _restore_perf_history=lambda *args: [[], [], []],
                             is_eval_epoch=lambda epoch: (epoch+1) % cfg.train.eval_period == 0,
                             set_epoch_k=scheduler.set_epoch_k, GPUMemoryProfiler=lambda *args: profiler,
                             load_ckpt=checkpoints, save_ckpt=checkpoints, clean_ckpt=checkpoints)
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
            namespace["custom_train"](loggers, [SimpleNamespace(dataset=[])]*3, model, None, Mock(), study_observer=Mock())
            self.assertEqual(evaluate.call_count, 400)
            checkpoints.assert_not_called()
            profiler.end_epoch.assert_called()

    def test_notebook_has_eight_independent_cells_and_no_embedded_source(self):
        notebook = json.loads((runner.ROOT / "colab/RunPod_PascalVOC_NewSchedules.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(len(notebook["cells"]), 13)
        for cell in notebook["cells"]:
            ast.parse("".join(cell["source"]))
            self.assertIsNone(cell["execution_count"])
        cells = notebook["cells"][4:12]
        self.assertEqual([c["metadata"]["experiment"] for c in cells],
                         [dict(method=m, seed=s) for m in runner.METHODS for s in runner.SEEDS])
        for cell in cells:
            values = cell["metadata"]["experiment"]
            self.assertEqual(ast.unparse(ast.parse("".join(cell["source"])).body[0]),
                             f"run_experiment(method='{values['method']}', seed={values['seed']})")
        setup = "".join(notebook["cells"][1]["source"])
        self.assertNotRegex(setup, r"(?i)bundle|base64|zlib")
        config = "".join(notebook["cells"][0]["source"])
        self.assertIn('WORKSPACE_ROOT = "/path/to/workspace"', config)
        self.assertIn("AUTO_SETUP = True", config)
        self.assertIn("Path(WORKSPACE_ROOT)", setup)
        self.assertIn("PINNED_COMMIT", setup)
        self.assertIn("session.prepare_environment", setup)
        self.assertIn('session.environment["repository"]["commit"] != PINNED_COMMIT', setup)


class WorkerSetupTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.session = runner.ExperimentSession(temporary.name)
        identity = patch.object(runner, "repository_identity", return_value={"commit": "fixture"})
        identity.start()
        self.addCleanup(identity.stop)

    def test_layout_and_worker_arguments_are_under_workspace_root(self):
        self.assertEqual(self.session.root, self.session.workspace_root / "results")
        self.assertEqual(self.session.setup, self.session.workspace_root / "setup_logs")
        self.assertEqual(self.session.dataset_dir, self.session.workspace_root / "datasets")
        self.assertEqual(self.session.venv, self.session.workspace_root / "environment")
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0)) as launch:
            self.session._worker("prepare", self.session.setup)
        args, options = launch.call_args
        self.assertEqual(args[0][-2:], ["--dataset-dir", str(self.session.dataset_dir)])
        self.assertEqual(Path(options["env"]["KEOPS_CACHE_FOLDER"]), self.session.setup / "cache/keops")

    def test_unrelated_results_are_preserved_before_environment_setup(self):
        marker = self.session.root / "environment.json"
        marker.write_text("existing content", encoding="utf-8")
        with patch.object(self.session, "_worker") as launch:
            with self.assertRaisesRegex(runner.SharedSetupError, "no study manifest"):
                self.session.prepare_environment(auto_setup=True)
            launch.assert_not_called()
        self.assertEqual(list(self.session.root.iterdir()), [marker])
        self.assertEqual(marker.read_text(encoding="utf-8"), "existing content")

    def test_verified_base_or_cached_environment_does_not_install(self):
        def verify(action, output, *args):
            self.assertEqual(action, "environment")
            runner.atomic_json(output / "environment.json", {"packages": ["fixture==1"]})
            return 0

        with patch.object(self.session, "_worker", side_effect=verify), patch.object(self.session, "_install") as install:
            self.session.prepare_environment(auto_setup=True)
            self.assertEqual(self.session.python, sys.executable)
            self.session.venv_python.parent.mkdir(parents=True)
            self.session.venv_python.touch()
            self.session.prepare_environment(auto_setup=True)
            self.assertEqual(self.session.python, str(self.session.venv_python))
            install.assert_not_called()
        environments = list(self.session.setup.glob("environment-*/environment.json"))
        self.assertEqual(len(environments), 2)
        self.assertEqual(runner.read_json(environments[0])["packages"], ["fixture==1"])

    def test_missing_dependency_selects_isolated_interpreter(self):
        cuda = {"torch": "2.5.1+cu124", "cuda_build": "12.4"}
        repair = {"component": "torch-scatter", "upgrade": False}

        def probe(action, output, *args):
            if self.session.python == sys.executable:
                runner.atomic_json(output / "cuda.json", cuda)
                runner.atomic_json(output / "repair.json", repair)
                return 1
            self.assertEqual(self.session.python, str(self.session.venv_python))
            runner.atomic_json(output / "environment.json", {"packages": ["torch-scatter==fixture"]})
            return 0

        with patch.object(self.session, "_worker", side_effect=probe), patch.object(self.session, "_install") as install:
            self.session.prepare_environment(auto_setup=True)
        args, options = install.call_args
        self.assertTrue(args[0].is_relative_to(self.session.setup))
        self.assertEqual(args[1:], (cuda, repair))
        self.assertFalse(options["requirements_installed"])
        self.assertEqual(self.session.python, str(self.session.venv_python))

    def test_incompatible_scatter_installs_only_matching_binary_and_requirements(self):
        cuda = {"torch": "2.5.1+cu124", "cuda_build": "12.4"}
        repair = {"component": "torch-scatter", "upgrade": True}
        with patch.object(sys, "prefix", sys.base_prefix), patch("subprocess.run") as launch:
            self.session._install(self.session.setup / "install.log", cuda, repair, False)
        venv, binary, requirements = [call.args[0] for call in launch.call_args_list]
        self.assertEqual(venv[-1], str(self.session.venv))
        self.assertEqual(binary[0], str(self.session.venv_python))
        self.assertIn("https://data.pyg.org/whl/torch-2.5.0+cu124.html", binary)
        self.assertIn("--force-reinstall", binary)
        self.assertIn("--no-index", binary)
        self.assertNotIn("--upgrade", requirements)
        self.assertEqual((self.session.setup / "constraints.txt").read_text().strip(), "torch==2.5.1+cu124")

    def test_failed_cuda_or_keops_probe_does_not_install_or_continue(self):
        with patch.object(self.session, "_worker", return_value=2), patch.object(self.session, "_install") as install:
            with self.assertRaisesRegex(runner.SharedSetupError, "GPU setup could not be verified"):
                self.session.prepare_environment(auto_setup=True)
            install.assert_not_called()
        self.assertIsNone(self.session.environment)


class NotebookCheckoutTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.workspace = root / "workspace"
        path = runner.ROOT / "colab/RunPod_PascalVOC_NewSchedules.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        setup = "".join(notebook["cells"][1]["source"]).split("\nworkspace_path =", 1)[0]
        self.namespace = {}
        exec(compile(setup, str(path), "exec"), self.namespace)
        self.resolve = self.namespace["resolve_repository"]
        self.cache = self.workspace / "repository"

    def test_unpublished_revision_stops_before_git_and_logs_error(self):
        self.namespace["PINNED_COMMIT"] = None
        with patch("subprocess.run") as git:
            with self.assertRaisesRegex(RuntimeError, "No study commit is pinned"):
                self.resolve(self.workspace)
            git.assert_not_called()
        log, = (self.workspace / "setup_logs").glob("source-*.log")
        self.assertIn("published to GitHub", log.read_text(encoding="utf-8"))
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_existing_unrelated_cache_folder_is_preserved(self):
        pin = "f" * 40
        self.namespace["PINNED_COMMIT"] = pin
        occupied = self.cache / pin
        occupied.mkdir(parents=True)
        marker = occupied / "existing.txt"
        marker.write_text("existing content", encoding="utf-8")
        with patch("shutil.which", return_value="git"), patch("subprocess.run") as git:
            with self.assertRaisesRegex(RuntimeError, "Expected a dedicated Git checkout"):
                self.resolve(self.workspace)
            git.assert_not_called()
        self.assertEqual(list(occupied.iterdir()), [marker])
        self.assertEqual(marker.read_text(encoding="utf-8"), "existing content")

    @unittest.skipUnless(shutil.which("git") and (runner.ROOT / ".git").exists(), "requires a Git checkout")
    def test_fetch_exact_revision_reuse_and_fail_closed(self):
        pin = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=runner.ROOT, text=True).strip()
        # Use the existing local commit as a transport fixture without publishing anything.
        self.namespace.update(PINNED_COMMIT=pin, SOURCE_URL=str(runner.ROOT), REQUIRED_FILES=("main.py",))
        checkout = self.resolve(self.workspace)
        self.assertEqual(checkout, self.cache / pin / "checkout")
        for name in ("datasets", "results", "setup_logs"):
            self.assertTrue((self.workspace / name).is_dir())
        self.assertEqual(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip(), pin)

        with patch.dict(self.namespace, git=Mock(wraps=self.namespace["git"])) as namespace:
            self.assertEqual(self.resolve(self.workspace), checkout)
            self.assertNotIn("fetch", [call.args[2] for call in namespace["git"].call_args_list])

        self.namespace["REQUIRED_FILES"] = ("nonexistent_study_source.py",)
        with self.assertRaisesRegex(RuntimeError, "Git ls-files failed"):
            self.resolve(self.workspace)
        self.namespace["REQUIRED_FILES"] = ("main.py",)

        (checkout / "local.txt").write_text("preserve me", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "has local changes"):
            self.resolve(self.workspace)
        self.assertTrue((checkout / "local.txt").is_file())

        missing = "f" * 40
        self.namespace["PINNED_COMMIT"] = missing
        with self.assertRaisesRegex(RuntimeError, f"Could not retrieve {missing}"):
            self.resolve(self.workspace)
        self.assertFalse((self.cache / missing).exists())


if __name__ == "__main__":
    unittest.main()
