"""CPU-only extended UDC and cross-commit source-manifest regression."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from common.hashing import sha256_file
from methods.glop.paper_results import fingerprint, write_json
from methods.udc.paper_protocol import budget_config, REGISTRY_PATH, REGISTRY_SHA256
from rebuttal.tsp2k.artifacts import Writer, read_artifact
from rebuttal.tsp2k.config import UDC_SWEEP, protocol, UPSTREAMS
from rebuttal.tsp2k import udc_extended as extended
from rebuttal.tsp2k import manifest as manifests
from tests.test_rebuttal_tsp2k import HAS_SIL, identity, row

ROOT = Path(__file__).resolve().parents[1]


class ExtendedBudgetAndStageTests(unittest.TestCase):
    def test_all_eleven_budgets_change_only_x(self):
        self.assertEqual(UDC_SWEEP, (1, 2, 5, 10, 20, 50, 100, 150, 200, 250, 300))
        frozen = budget_config("tsp", "fewer")
        for x in UDC_SWEEP:
            observed = protocol("UDC", x)["full_solver_config"]
            self.assertEqual(observed["x"], x)
            observed["x"] = frozen["x"]
            self.assertEqual(observed, frozen)
        self.assertEqual(sha256_file(REGISTRY_PATH), REGISTRY_SHA256)

    def test_only_udc_commands_and_no_partial_initial_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            lock = Path(temp) / "lock.json"
            write_json(lock, {"projects": {"UDC": {"path": "/udc-source"}}})
            for stage in extended.STAGES:
                commands = extended.commands(lock, Path(temp) / "runs", stage)
                self.assertEqual(len(commands), 11)
                for command in commands:
                    self.assertTrue(command[2].endswith("methods/udc/rebuttal_eval.py"))
                    self.assertEqual(command[command.index("--experiment-stage")+1], stage)
                    self.assertEqual(command[command.index("--count")+1], "64" if stage == "fullset" else "2")
                    self.assertNotIn("methods/sil", " ".join(command))
                    self.assertNotIn("methods/glop", " ".join(command))
            for stage in ("preflight", "calibration"):
                with self.assertRaises(ValueError):
                    extended.commands(lock, temp, stage, [100])
            for budgets in ([300, 300], [999], [True], []):
                with self.assertRaises(ValueError):
                    extended.commands(lock, temp, "fullset", budgets)

    def test_direct_fullset_cannot_bypass_stage_gate(self):
        from rebuttal.tsp2k.runner import main
        with tempfile.TemporaryDirectory() as temp:
            lock, output = Path(temp) / "lock.json", Path(temp) / "fullset/UDC/300"
            write_json(lock, {})
            common = ["--budget", "300", "--phase", "fullset",
                      "--asset-lock", str(lock), "--output-dir", str(output)]
            with patch("rebuttal.tsp2k.runner.verify_lock", return_value={}):
                with self.assertRaisesRegex(ValueError, "explicit gated stage"):
                    main(common, method="UDC")
                with patch.object(extended, "stage_gate", side_effect=ValueError("missing current preflight")):
                    with self.assertRaisesRegex(ValueError, "missing current preflight"):
                        main(common + ["--experiment-stage", "fullset", "--evidence-root", temp], method="UDC")

    def test_stage_gate_requires_current_preflight_and_calibration_and_keeps_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            root, lock, calibration = Path(temp), Path(temp) / "lock.json", Path(temp) / "cal.json"
            write_json(lock, {})
            locked = {"projects": {"UDC": {"commit": "a"*40}}}
            pre = {"stage": "preflight", "points": []}
            previous = {"stage": "calibration", "asset_lock_sha256": sha256_file(lock),
                        "project_commit": "a"*40, "preflight": fingerprint(pre)}
            points = []
            for b in UDC_SWEEP:
                directory = root / "calibration/UDC" / str(b)
                directory.mkdir(parents=True)
                write_json(directory / "metadata.json", {"identity": {"stage_evidence": previous}})
                points.append({"artifact_path": str(directory)})
            cal = {"stage": "calibration", "points": points}
            write_json(calibration, cal)
            def reports(output_root, stage, lock_path):
                return pre if stage == "preflight" else cal
            with patch.object(extended, "checked_lock", return_value=locked), \
                 patch.object(extended, "stage_report", side_effect=reports):
                self.assertIn("preflight", extended.stage_gate("calibration", root, lock))
                with self.assertRaises(ValueError):
                    extended.stage_gate("fullset", root, lock)
                passed = extended.stage_gate("fullset", root, lock, calibration)
                self.assertEqual(passed["calibration"]["sha256"], sha256_file(calibration))
                write_json(calibration, {"stage": "calibration", "points": []})
                with self.assertRaises(ValueError):
                    extended.stage_gate("fullset", root, lock, calibration)
                write_json(calibration, cal)
                previous["preflight"] = "f"*64
                write_json(Path(points[0]["artifact_path"]) / "metadata.json",
                           {"identity": {"stage_evidence": previous}})
                with self.assertRaisesRegex(ValueError, "exact preflight"):
                    extended.stage_gate("fullset", root, lock, calibration)
            # Retained evidence has not been cleared on rejection.
            self.assertTrue((Path(points[0]["artifact_path"]) / "metadata.json").is_file())

    def test_stage_report_refuses_mixed_stage_lock_and_commit(self):
        with tempfile.TemporaryDirectory() as temp:
            root, lock_path = Path(temp), Path(temp) / "lock.json"
            write_json(lock_path, {})
            locked = {"projects": {"UDC": {"commit": "a"*40, "path": temp}},
                      "dataset": {"sha256": "d"*64}}
            def measured(directory, **kwargs):
                Path(directory).mkdir(parents=True, exist_ok=True)
                write_json(Path(directory) / "metadata.json", {"cpu_test": True})
                ident = identity("UDC", kwargs["budget"])
                ident.update(experiment_stage="preflight", udc_evidence_root=str(root),
                    asset_lock={"path": str(lock_path), "sha256": sha256_file(lock_path)})
                return ident, [], {"mean_time_seconds": 35.0 if kwargs["budget"] == 300 else 2.0,
                                   "budget_value": kwargs["budget"]}
            with patch.object(extended, "checked_lock", return_value=locked), \
                 patch.object(extended, "verify_evidence", side_effect=measured):
                report = extended.stage_report(root, "preflight", lock_path)
                self.assertEqual(len(report["points"]), 11)
                self.assertEqual(report["over_30_second_budgets"], [300])
                with self.assertRaises(ValueError):
                    extended.stage_report(root, "calibration", lock_path)
                original = measured
                def wrong_lock(directory, **kwargs):
                    ident, records, summary = original(directory, **kwargs)
                    ident["asset_lock"]["sha256"] = "f"*64
                    return ident, records, summary
                with patch.object(extended, "verify_evidence", side_effect=wrong_lock):
                    with self.assertRaises(ValueError):
                        extended.stage_report(root, "preflight", lock_path)

    def test_old_asset_lock_is_not_an_extended_experiment(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "asset_lock.json"
            write_json(path, {"schema": "tsp2k-asset-lock.v1"})
            with self.assertRaisesRegex(ValueError, "newly audited"):
                extended.checked_lock(path)

    def test_changed_current_project_or_aggregation_commit_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "asset_lock.json"
            write_json(path, {"reference_lock": {}, "gpu_status": "VERIFIED_RTX4090_INVENTORY",
                "projects": {"UDC": {"path": temp, "commit": "a"*40},
                             "SIL": {"path": temp, "commit": "b"*40}}})
            with patch.object(extended, "verify_lock", side_effect=ValueError("project changed")):
                with self.assertRaisesRegex(ValueError, "project changed"):
                    extended.checked_lock(path)
            with patch.object(extended, "verify_lock"), \
                 patch.object(extended, "project", return_value={"commit": "c"*40}):
                with self.assertRaisesRegex(ValueError, "aggregation worktree"):
                    extended.checked_lock(path)

    def test_scheduler_stops_on_failure_and_refuses_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root, lock = Path(temp) / "runs", Path(temp) / "asset_lock.json"
            write_json(lock, {"projects": {"UDC": {"path": "/source"}}})
            argv = ["--asset-lock", str(lock), "--output-root", str(root),
                    "--stage", "preflight", "--execute"]
            def failed(command, **kwargs):
                output = Path(command[command.index("--output-dir")+1])
                output.mkdir(parents=True)
                write_json(output / "metadata.json", {"state": "FAILED"})
                raise subprocess.CalledProcessError(1, command)
            with patch.object(extended, "stage_gate"), \
                 patch.object(extended.subprocess, "run", side_effect=failed) as process:
                with self.assertRaises(subprocess.CalledProcessError):
                    extended.main(argv)
                self.assertEqual(process.call_count, 1)
            self.assertEqual(json.loads((root/"preflight/UDC/1/metadata.json").read_text())["state"], "FAILED")
            with self.assertRaisesRegex(ValueError, "existing UDC output"):
                extended.main(argv)

    def test_old_reference_lock_tamper_is_rejected(self):
        from rebuttal.tsp2k.audit import verify_reference_lock
        with tempfile.TemporaryDirectory() as temp:
            old = Path(temp) / "old.json"
            write_json(old, {})
            locked = {"reference_lock": {"path": str(old), "sha256": sha256_file(old)}}
            old.write_text("tampered")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                verify_reference_lock(locked)

    def test_reference_lock_binds_dataset_checkpoint_and_all_references(self):
        from rebuttal.tsp2k.audit import verify_reference_lock
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "original.json"
            methods = {m: {"upstream": {"commit": UPSTREAMS[m][1], "dirty": False,
                                       "url": "https://github.com/" + UPSTREAMS[m][0] + ".git"},
                           "assets": [{"sha256": "a"*64}]} for m in UPSTREAMS}
            dataset = {"sha256": "d"*64, "count": 64, "reference_objectives_sha256": "e"*64}
            old = {"schema": "tsp2k-asset-lock.v1", "methods": methods, "dataset": dataset,
                   "projects": {"GLOP": {"commit": manifests.LEGACY_COMMITS["GLOP"]},
                                "UDC": {"commit": manifests.LEGACY_COMMITS["GLOP"]},
                                "SIL": {"commit": manifests.LEGACY_COMMITS["SIL"]}}}
            write_json(path, old)
            current = deepcopy(old)
            current["projects"]["UDC"]["commit"] = "b"*40
            current["projects"]["SIL"]["commit"] = "c"*40
            current["reference_lock"] = {"path": str(path), "sha256": sha256_file(path)}
            verify_reference_lock(current)
            current["methods"]["UDC"]["upstream"]["url"] = (
                "git@github.com:" + UPSTREAMS["UDC"][0] + ".git")
            verify_reference_lock(current)
            for field in ("sha256", "count", "reference_objectives_sha256"):
                changed = deepcopy(current)
                changed["dataset"][field] = 63 if field == "count" else "f"*64
                with self.assertRaisesRegex(ValueError, "dataset/reference"):
                    verify_reference_lock(changed)
            current["methods"]["UDC"]["assets"][0]["sha256"] = "f"*64
            with self.assertRaisesRegex(ValueError, "checkpoint identity"):
                verify_reference_lock(current)


@unittest.skipUnless(HAS_SIL, "cross-method aggregation uses the new SIL-derived worktree")
class CrossCommitManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name)
        cls.repository = cls.base / "source"
        subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", str(ROOT),
                        str(cls.repository)], check=True)
        subprocess.run(["git", "checkout", "-q", "--detach", manifests.LEGACY_COMMITS["GLOP"]],
                       cwd=cls.repository, check=True)
        source = ROOT / "rebuttal/tsp2k/config.py"
        (cls.repository / "rebuttal/tsp2k/config.py").write_bytes(source.read_bytes())
        subprocess.run(["git", "add", "rebuttal/tsp2k/config.py"], cwd=cls.repository, check=True)
        subprocess.run(["git", "-c", "user.name=CPU test", "-c", "user.email=cpu@test.invalid",
                        "commit", "-qm", "CPU-only extended config fixture"], cwd=cls.repository, check=True)
        cls.udc_commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                cwd=cls.repository, text=True).strip()
        theta = np.linspace(0, 2*np.pi, 2000, endpoint=False)
        cls.points = np.stack((np.cos(theta), np.sin(theta)), axis=1).astype(np.float32)
        from problems.tsp.validate import validate
        cls.validate = staticmethod(validate)
        cls.tasks = [types.SimpleNamespace(
            points=cls.points, ref_sol=np.arange(2000),
            evaluate=lambda tour: validate(cls.points, tour)["independent_objective"],
            check_constraints=lambda tour: validate(cls.points, tour)["feasible"]) for _ in range(64)]
        cls.references = [task.evaluate(task.ref_sol) for task in cls.tasks]
        cls.source_cache = {}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def sources(self, method, commit):
        key = (method, commit)
        if key not in self.source_cache:
            folders = ("rebuttal/tsp2k/", "common/", f"methods/{method.lower()}/", "problems/tsp/")
            paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit],
                                            cwd=self.repository, text=True).splitlines()
            self.source_cache[key] = [{"path": p, "sha256": hashlib.sha256(
                manifests.git_bytes(str(self.repository), commit, p)).hexdigest()}
                for p in paths if p.endswith(".py") and p.startswith(folders)]
        return deepcopy(self.source_cache[key])

    def point(self, root, method, budget):
        from rebuttal.tsp2k.runner import validated, SIL_WARMUP_POLICY
        commit = self.udc_commit if method == "UDC" else manifests.LEGACY_COMMITS[method]
        ident = identity(method, budget, "fullset", 64)
        ident.update(project_commit=commit, project={"commit": commit, "dirty": False},
            source_provenance=self.sources(method, commit),
            asset_lock={"path": "/test-only/original-lock.json", "sha256": "a"*64})
        ident["dataset"]["reference_objectives_sha256"] = fingerprint(self.references)
        if method == "SIL":
            from methods.sil.paper_results import TIMING_SEMANTICS
            ident["timing"] = {"semantics": TIMING_SEMANTICS, "warmup": SIL_WARMUP_POLICY}
        elif method == "GLOP":
            from methods.glop.paper_results import TSP_TIMING_SEMANTICS
            ident["timing"] = {"semantics": TSP_TIMING_SEMANTICS, "warmup_instances": 2}
        else:
            from methods.udc.paper_production import PRODUCTION_TIMING, CUDA_ALLOCATOR_PROVENANCE
            ident.update(experiment_stage="fullset", udc_evidence_root=str(root),
                         stage_evidence={"calibration": {"path": str(root / "calibration.json")}})
            ident["timing"] = {"semantics": PRODUCTION_TIMING, "warmup_instances": 0,
                               "cuda_allocator": CUDA_ALLOCATOR_PROVENANCE}
        directory = root / method / str(budget)
        writer = Writer(directory, ident)
        for index, task in enumerate(self.tasks):
            runtime = None if method == "SIL" else 2.0
            record = validated(task, index, list(range(2000))+[0], self.references[index],
                               runtime, source="quality")
            if method == "GLOP":
                record["runtime_components"] = {"per_instance_solve_seconds": 2.0,
                                               "shared_ri_order_generation_seconds": 0.0}
            if method == "UDC":
                record["native_evidence"] = {
                    "x": budget, "completed_stages": [None]*budget, "alpha": 50,
                    "configured_pomo": 2, "effective_pomo": 2, "seed": 1234,
                    "status": "KIT_VALIDATED", "runtime_seconds": 2.0,
                    "timing_semantics": PRODUCTION_TIMING}
            writer.append(record)
        if method == "SIL":
            for index in range(3):
                writer.append(validated(self.tasks[index], index, list(range(2000))+[0],
                    self.references[index], 1.5, source="timing"))
        writer.finish()
        return directory

    def fixture(self, root):
        from rebuttal.tsp2k.config import GLOP_SWEEP, SIL_SWEEP
        for method, budgets in (("GLOP", GLOP_SWEEP), ("SIL", SIL_SWEEP), ("UDC", [2, 300])):
            for budget in budgets:
                self.point(root, method, budget)
        manifest = manifests.build_manifest(
            glop_root=root/"GLOP", sil_root=root/"SIL", udc_root=root/"UDC",
            udc_budgets=[2,300], udc_commit=self.udc_commit, source_repository=self.repository)
        path = root / "manifest.json"
        write_json(path, manifest)
        return path, manifest

    def measured_read(self, directory):
        # Disk fixtures remain SYNTHETIC_TEST. Only this CPU test's in-memory
        # return value enters the measured-data branch; no fake GPU artifact is emitted.
        ident, records, summary = read_artifact(directory)
        ident["evidence_kind"] = "SERVER_MEASUREMENT"
        return ident, records, summary

    def test_strict_cross_commit_manifest_legacy_readback_and_plot_exports(self):
        from contextlib import ExitStack
        from rebuttal.tsp2k.aggregate import main as aggregate_main
        from rebuttal.tsp2k.plot import main as plot_main
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, manifest = self.fixture(root)
            # The new read_artifact independently accepts the unchanged old SIL schema.
            sil_identity, records, summary = read_artifact(root/"SIL/10")
            self.assertEqual(sil_identity["project_commit"], manifests.LEGACY_COMMITS["SIL"])
            self.assertEqual(summary["quality_batch_size"], 64)
            self.assertEqual(summary["timing_sample_count"], 3)
            with self.assertRaises(ValueError):
                manifests.verify_manifest(path)  # All stored fixtures remain synthetic.
            with ExitStack() as stack:
                stack.enter_context(patch.object(manifests, "read_artifact", side_effect=self.measured_read))
                stack.enter_context(patch("rebuttal.tsp2k.aggregate.read_artifact", side_effect=self.measured_read))
                stack.enter_context(patch.object(manifests, "verify_assets", return_value={}))
                stack.enter_context(patch.object(manifests, "load_tasks", return_value=self.tasks))
                stage = read_artifact(root/"UDC/2")[0]["stage_evidence"]
                stack.enter_context(patch.object(extended, "stage_gate", return_value=stage))
                report, proof = manifests.verify_manifest(path)
                self.assertEqual(len(report["points"]), 12)
                self.assertEqual(proof["status"], "PASS")
                self.assertEqual(len(set(r["project_commit"] for r in proof["points"])), 3)
                aggregate_main(["--manifest", str(path), "--output", str(root/"curves.json")])
                plot_main(["--report", str(root/"curves.json"), "--output-prefix", str(root/"curves")])
                for suffix in ("csv", "json", "png", "pdf", "svg"):
                    self.assertTrue((root/f"curves.{suffix}").is_file())
                self.assertTrue((root/"curves_verification.json").is_file())
                bad = deepcopy(manifest)
                bad["artifacts"][0]["metadata_sha256"] = "f"*64
                write_json(path, bad)
                with self.assertRaisesRegex(ValueError, "metadata SHA256"):
                    manifests.verify_manifest(path)
                bad = deepcopy(manifest)
                bad["artifacts"][1]["expected_project_commit"] = self.udc_commit
                write_json(path, bad)
                with self.assertRaisesRegex(ValueError, "same-method commit"):
                    manifests.verify_manifest(path)

    def test_source_provenance_cannot_be_disabled_or_reinterpreted(self):
        ident = identity("SIL", 10)
        ident.update(project_commit=manifests.LEGACY_COMMITS["SIL"],
                     source_provenance=self.sources("SIL", manifests.LEGACY_COMMITS["SIL"]))
        manifests.verify_sources(ident, self.repository)
        ident["source_provenance"].pop()
        with self.assertRaisesRegex(ValueError, "omits"):
            manifests.verify_sources(ident, self.repository)
        ident = identity("UDC", 300)
        ident.update(project_commit=manifests.LEGACY_COMMITS["GLOP"],
                     source_provenance=self.sources("UDC", manifests.LEGACY_COMMITS["GLOP"]))
        with self.assertRaisesRegex(ValueError, "did not support"):
            manifests.verify_sources(ident, self.repository)


if __name__ == "__main__":
    unittest.main()
