from copy import deepcopy
import json
from pathlib import Path
import tempfile
import types
import unittest

from common.hashing import sha256_file
from common.protocol_rebind import (GENERATION_MODE, LEGACY_SOLVER_COMMIT,
                                    verify_quality_artifact,
                                    verify_timing_artifact,
                                    write_rebound_artifacts)
from methods.lehd.parallel_eval import (DERIVED_MODE, batch_ranges,
                                        derive_bs1_summary, parallel_scope,
                                        summarize_measured)
from methods.lehd.rebind_rrc50 import (
    LEGACY_RRC50_BATCH_OVERRIDES, _legacy_batch_size,
    _legacy_quality_protocol as lehd_legacy_protocol,
    build_plan as build_lehd_rebind_plan)
from methods.sil.rebind_prc50 import _legacy_quality_protocol as sil_legacy_protocol
from methods.lehd.config import (AUTHOR_BATCH_REGISTRY as LEHD_AUTHOR_BATCH_REGISTRY,
                                 resolve_author_batch_config as lehd_author_config)
from methods.sil.config import resolve_author_batch_config as sil_author_config


UPSTREAMS = {
    "LEHD": "274df3c4975384592b60fe7f79fbb2441ce11c15",
    "SIL": "9ec783e90a1631f7b95f84eb20f8f9751cb45c10",
}


class ArtifactFixture:
    def __init__(self, root: Path, method: str, *, problem: str = "tsp",
                 size: int = 3, count: int = 3, batch_size: int = 2,
                 dataset_name: str = "fixture.pkl",
                 checkpoint_name: str = "fixture.pt",
                 protocol: dict | None = None):
        root.mkdir(parents=True, exist_ok=True)
        self.root, self.method = root, method
        self.problem, self.size = problem, int(size)
        self.count, self.batch_size = int(count), int(batch_size)
        self.label_key = "protocol_label" if method == "LEHD" else "budget_label"
        self.budget_key = "RRC_budget" if method == "LEHD" else "budget"
        self.dataset_name = dataset_name
        self.checkpoint_name = checkpoint_name
        self.dataset = root / self.dataset_name
        self.checkpoint = root / self.checkpoint_name
        self.dataset.write_bytes(b"dataset")
        self.checkpoint.write_bytes(b"checkpoint")
        self.dataset_sha = sha256_file(self.dataset)
        self.checkpoint_sha = sha256_file(self.checkpoint)
        self.project = {"commit": LEGACY_SOLVER_COMMIT, "dirty": False,
                        "url": "https://github.com/xihe-820/neural-routing-baselines"}
        self.upstream = {"commit": UPSTREAMS[method], "dirty": False,
                         "url": "https://github.com/example/upstream"}
        self.source_files = [{"path": "runner.py", "sha256": "a" * 64}]
        self.protocol = deepcopy(protocol) if protocol is not None else {
            "method": method, "problem": self.problem.upper(),
            "actual_problem_size": self.size, self.label_key: "fewer",
            self.budget_key: 50, "solver_semantics": "frozen",
        }
        self.quality_dir = root / "quality"
        self.timing_path = root / "timing.json"
        self._write_quality()
        self._write_timing()

    def _record(self, index: int, runtime: float | None = None) -> dict:
        objective, reference = float(index + 2), float(index + 1)
        row = {
            "dataset_instance_index": index, "problem": self.problem.upper(),
            "problem_size": self.size, "evidence_status": "KIT_VALIDATED",
            "independent_feasible": True, "kit_feasible": True,
            "official_vs_independent": {"pass": True},
            "independent_vs_kit": {"pass": True},
            "independent_objective": objective, "reference_objective": reference,
            "gap_percent": (objective - reference) / reference * 100.0,
        }
        if runtime is not None:
            row["runtime_seconds"] = runtime
        return row

    def _write_quality(self):
        self.quality_dir.mkdir()
        records = [self._record(index) for index in range(self.count)]
        (self.quality_dir / "validated_records.jsonl").write_text("".join(
            json.dumps(row, sort_keys=True) + "\n" for row in records))
        metadata = {
            "state": "KIT_VALIDATED", "artifact_class": "baseline_result_reproduction",
            "protocol": self.protocol, "official_source_modified": False,
            "dataset": {"path": str(self.dataset), "sha256": self.dataset_sha,
                        "count": self.count},
            "checkpoint": {"path": str(self.checkpoint), "sha256": self.checkpoint_sha},
            "project": self.project, "upstream": self.upstream,
            "environment": {"gpu": "NVIDIA GeForce RTX 4090"},
            "source_files": self.source_files, "validated_count": self.count,
        }
        objectives = [row["independent_objective"] for row in records]
        references = [row["reference_objective"] for row in records]
        gaps = [row["gap_percent"] for row in records]
        summary = {
            "status": "KIT_VALIDATED", "artifact_class": "baseline_result_reproduction",
            "method": self.method, "problem": self.problem.upper(),
            "problem_size": self.size,
            "protocol": "fewer", self.budget_key: 50,
            "failed_count": 0, "validated_count": self.count,
            "paper_result_eligible_for_quality": True, "timing_column_eligible": False,
            "dataset_sha256": self.dataset_sha, "checkpoint_sha256": self.checkpoint_sha,
            "mean_objective": sum(objectives) / self.count,
            "mean_reference_objective": sum(references) / self.count,
            "mean_instance_gap_percent": sum(gaps) / self.count,
            "batch_size_requested": self.batch_size,
            "original_instance_batch_size": self.batch_size,
            "effective_batch_sizes": [
                min(self.batch_size, self.count - start)
                for start in range(0, self.count, self.batch_size)
            ],
            "number_of_batches": len(range(0, self.count, self.batch_size)),
            "total_wall_time_seconds": 2.0, "project": self.project,
            "upstream": self.upstream, "source_files": self.source_files,
        }
        (self.quality_dir / "metadata.json").write_text(json.dumps(metadata))
        (self.quality_dir / "summary.json").write_text(json.dumps(summary))

    def _write_timing(self):
        runtimes = [1.0, 2.0, 3.0]
        summary = {
            "status": "KIT_VALIDATED", "artifact_class": "baseline_bs1_timing_probe",
            "method": self.method, "problem": self.problem.upper(),
            "problem_size": self.size,
            "protocol": "fewer", self.budget_key: 50,
            "original_instance_batch_size": 1, "count": 3, "validated_count": 3,
            "failed_count": 0, "timing_column_eligible": True,
            "paper_result_eligible_for_quality": False,
            "official_source_modified": False, "individual_runtimes": runtimes,
            "mean_runtime_seconds": 2.0, "median_runtime_seconds": 2.0,
            "min_runtime_seconds": 1.0, "max_runtime_seconds": 3.0,
            "validation_records": [self._record(index, runtime)
                                   for index, runtime in enumerate(runtimes)],
            "dataset_sha256": self.dataset_sha, "checkpoint_sha256": self.checkpoint_sha,
            "project": self.project, "upstream": self.upstream,
            "environment": {"gpu": "NVIDIA RTX 4090"},
            "source_files": self.source_files,
        }
        self.timing_path.write_text(json.dumps(summary))

    def quality(self):
        return verify_quality_artifact(
            self.quality_dir, method=self.method, problem=self.problem,
            problem_size=self.size, protocol_label="fewer",
            budget_key=self.budget_key, budget=50, expected_protocol=self.protocol,
            expected_count=self.count, expected_batch_size=self.batch_size,
            expected_project_commit=LEGACY_SOLVER_COMMIT,
            upstream_commit=UPSTREAMS[self.method], dataset_filename=self.dataset_name,
            checkpoint_filename=self.checkpoint_name)

    def timing(self, quality):
        return verify_timing_artifact(
            self.timing_path, method=self.method, problem=self.problem,
            problem_size=self.size, protocol_label="fewer",
            budget_key=self.budget_key, budget=50, expected_count=3,
            expected_project_commit=LEGACY_SOLVER_COMMIT,
            upstream_commit=UPSTREAMS[self.method],
            dataset_sha256=quality["dataset"]["sha256"],
            checkpoint_sha256=quality["checkpoint"]["sha256"])


LEGACY_CVRP2000_OVERRIDE_REASON = (
    "RTX 4090 author batch 100 CUDA OOM; explicitly approved batch 50 retry"
)


def make_legacy_cvrp2000_fixture(root: Path):
    legacy_root = root / "legacy"
    staging = root / "staging"
    protocol = lehd_legacy_protocol(
        "cvrp", 2000, LEGACY_CVRP2000_OVERRIDE_REASON)
    fixture = ArtifactFixture(
        staging, "LEHD", problem="cvrp", size=2000, count=100, batch_size=50,
        dataset_name="cvrp2000_hgs-360s_57.181.pkl",
        checkpoint_name="checkpoint-40.pt", protocol=protocol)
    quality = legacy_root / "author_batch/cvrp2000/fewer"
    timing = legacy_root / "bs1_timing/cvrp2000/fewer.json"
    quality.parent.mkdir(parents=True)
    timing.parent.mkdir(parents=True)
    fixture.quality_dir.rename(quality)
    fixture.timing_path.rename(timing)
    return legacy_root, quality


class VerifiedRebindTests(unittest.TestCase):
    def test_current_more_is_algorithm_equivalent_to_legacy_fewer50(self):
        current = lehd_author_config("tsp", 100, "more")
        legacy = lehd_legacy_protocol("tsp", 100)
        for field in ("protocol_label", "budget_mapping_origin"):
            current.pop(field)
            legacy.pop(field)
        self.assertEqual(current, legacy)
        current = sil_author_config("tsp", 1000, "more")
        legacy = sil_legacy_protocol("tsp", 1000)
        for field in ("budget_label", "evaluation_mapping", "budget_mapping_origin"):
            current.pop(field, None)
            legacy.pop(field, None)
        self.assertEqual(current, legacy)

    def test_lehd_and_sil_quality_and_timing_positive_rebind(self):
        for method in ("LEHD", "SIL"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temp:
                fixture = ArtifactFixture(Path(temp), method)
                quality = fixture.quality()
                timing = fixture.timing(quality)
                current_protocol = deepcopy(fixture.protocol)
                current_protocol[fixture.label_key] = "more"
                destination = Path(temp) / "new" / "quality"
                timing_destination = Path(temp) / "new" / "timing.json"
                write_rebound_artifacts(
                    quality=quality, timing=timing,
                    quality_destination=destination,
                    timing_destination=timing_destination,
                    current_quality_protocol=current_protocol,
                    current_timing_protocol=current_protocol,
                    current_project={"commit": "new", "dirty": False, "url": "project"},
                    label_key=fixture.label_key, budget_key=fixture.budget_key,
                    rebind_source_files=[{"path": "rebind.py", "sha256": "b" * 64}])
                metadata = json.loads((destination / "metadata.json").read_text())
                summary = json.loads((destination / "summary.json").read_text())
                rebound_timing = json.loads(timing_destination.read_text())
                self.assertEqual(metadata["artifact_generation_mode"], GENERATION_MODE)
                self.assertEqual(metadata["solver_execution_project"]["commit"],
                                 LEGACY_SOLVER_COMMIT)
                self.assertEqual(metadata["solver_execution_project_commit"],
                                 LEGACY_SOLVER_COMMIT)
                self.assertEqual(metadata["protocol_rebinding_project"]["commit"], "new")
                self.assertEqual(metadata["protocol_rebinding_project_commit"], "new")
                self.assertEqual(summary["protocol"], "more")
                self.assertEqual(rebound_timing["protocol"], "more")
                self.assertEqual(summary[fixture.budget_key], 50)
                self.assertEqual(rebound_timing["count"], 3)
                self.assertEqual(sha256_file(destination / "validated_records.jsonl"),
                                 quality["hashes"]["records"])

    def test_quality_rebind_fail_closed_cases_for_both_methods(self):
        cases = {
            "wrong legacy budget": ("summary.json", lambda value, f: value.__setitem__(f.budget_key, 49)),
            "wrong protocol": ("summary.json", lambda value, f: value.__setitem__("protocol", "more")),
            "wrong project": ("metadata.json", lambda value, f: value["project"].__setitem__("commit", "wrong")),
            "wrong upstream": ("summary.json", lambda value, f: value["upstream"].__setitem__("commit", "wrong")),
            "dataset SHA mismatch": ("summary.json", lambda value, f: value.__setitem__("dataset_sha256", "0" * 64)),
            "checkpoint SHA mismatch": ("summary.json", lambda value, f: value.__setitem__("checkpoint_sha256", "0" * 64)),
            "wrong problem": ("summary.json", lambda value, f: value.__setitem__("problem", "CVRP")),
            "wrong size": ("summary.json", lambda value, f: value.__setitem__("problem_size", 4)),
            "wrong validated count": ("summary.json", lambda value, f: value.__setitem__("validated_count", 2)),
            "failed count": ("summary.json", lambda value, f: value.__setitem__("failed_count", 1)),
            "objective aggregate": ("summary.json", lambda value, f: value.__setitem__("mean_objective", 99.0)),
            "batch mapping": ("summary.json", lambda value, f: value.__setitem__("effective_batch_sizes", [3])),
            "algorithm semantics": ("metadata.json", lambda value, f: value["protocol"].__setitem__("solver_semantics", "tampered")),
            "official modified": ("metadata.json", lambda value, f: value.__setitem__("official_source_modified", True)),
        }
        for method in ("LEHD", "SIL"):
            for name, (filename, mutate) in cases.items():
                with self.subTest(method=method, case=name), tempfile.TemporaryDirectory() as temp:
                    fixture = ArtifactFixture(Path(temp), method)
                    path = fixture.quality_dir / filename
                    value = json.loads(path.read_text())
                    mutate(value, fixture)
                    path.write_text(json.dumps(value))
                    with self.assertRaises(ValueError):
                        fixture.quality()
            with self.subTest(method=method, case="tampered records"), tempfile.TemporaryDirectory() as temp:
                fixture = ArtifactFixture(Path(temp), method)
                path = fixture.quality_dir / "validated_records.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                rows[0]["independent_objective"] = 999.0
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaises(ValueError):
                    fixture.quality()

    def test_timing_rebind_fail_closed_cases_for_both_methods(self):
        cases = {
            "wrong protocol": lambda value, f: value.__setitem__("protocol", "more"),
            "wrong budget": lambda value, f: value.__setitem__(f.budget_key, 49),
            "wrong project": lambda value, f: value["project"].__setitem__("commit", "wrong"),
            "wrong upstream": lambda value, f: value["upstream"].__setitem__("commit", "wrong"),
            "wrong problem": lambda value, f: value.__setitem__("problem", "CVRP"),
            "wrong size": lambda value, f: value.__setitem__("problem_size", 4),
            "dataset SHA mismatch": lambda value, f: value.__setitem__("dataset_sha256", "0" * 64),
            "checkpoint SHA mismatch": lambda value, f: value.__setitem__("checkpoint_sha256", "0" * 64),
            "wrong count": lambda value, f: value.__setitem__("count", 2),
            "failed count": lambda value, f: value.__setitem__("failed_count", 1),
            "nonpositive runtime": lambda value, f: value["individual_runtimes"].__setitem__(0, 0),
            "aggregate mismatch": lambda value, f: value.__setitem__("mean_runtime_seconds", 99),
            "validation failure": lambda value, f: value["validation_records"][0].__setitem__("kit_feasible", False),
            "official modified": lambda value, f: value.__setitem__("official_source_modified", True),
        }
        for method in ("LEHD", "SIL"):
            for name, mutate in cases.items():
                with self.subTest(method=method, case=name), tempfile.TemporaryDirectory() as temp:
                    fixture = ArtifactFixture(Path(temp), method)
                    quality = fixture.quality()
                    value = json.loads(fixture.timing_path.read_text())
                    mutate(value, fixture)
                    fixture.timing_path.write_text(json.dumps(value))
                    with self.assertRaises(ValueError):
                        fixture.timing(quality)

    def test_existing_destination_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = ArtifactFixture(Path(temp), "LEHD")
            quality, timing = fixture.quality(), fixture.timing(fixture.quality())
            destination = Path(temp) / "exists"
            destination.mkdir()
            with self.assertRaisesRegex(ValueError, "destination already exists"):
                write_rebound_artifacts(
                    quality=quality, timing=timing, quality_destination=destination,
                    timing_destination=Path(temp) / "timing-new.json",
                    current_quality_protocol={"protocol_label": "more", "RRC_budget": 50},
                    current_timing_protocol={"protocol_label": "more", "RRC_budget": 50},
                    current_project={"commit": "new", "dirty": False},
                    label_key="protocol_label", budget_key="RRC_budget",
                    rebind_source_files=[{"path": "x", "sha256": "a" * 64}])

    def test_lehd_cvrp2000_legacy_bs50_override_rebinds_with_exact_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            legacy_root, _ = make_legacy_cvrp2000_fixture(root)
            plan = build_lehd_rebind_plan(
                legacy_root=legacy_root, output_root=root / "output",
                problem="cvrp", problem_size=2000,
                current_project={"commit": "current", "dirty": False},
                rebind_source_files=[{"path": "rebind.py", "sha256": "b" * 64}])
            self.assertEqual(len(plan), 1)
            cell = plan[0]
            self.assertEqual(cell["quality"]["summary"]["batch_size_requested"], 50)
            self.assertEqual(cell["quality"]["summary"]["effective_batch_sizes"], [50, 50])
            self.assertEqual(cell["quality"]["summary"]["number_of_batches"], 2)
            self.assertEqual(cell["quality"]["metadata"]["protocol"]["author_batch_size"], 100)
            self.assertEqual(cell["current_quality_protocol"]["author_batch_size"], 100)
            self.assertEqual(cell["current_quality_protocol"]["batch_size_requested"], 100)
            write_rebound_artifacts(
                quality=cell["quality"], timing=cell["timing"],
                quality_destination=cell["quality_destination"],
                timing_destination=cell["timing_destination"],
                current_quality_protocol=cell["current_quality_protocol"],
                current_timing_protocol=cell["current_timing_protocol"],
                current_project=cell["current_project"], label_key="protocol_label",
                budget_key="RRC_budget", rebind_source_files=cell["rebind_source_files"],
                preserve_solver_execution_batch_identity=True)
            metadata = json.loads(
                (cell["quality_destination"] / "metadata.json").read_text())
            summary = json.loads(
                (cell["quality_destination"] / "summary.json").read_text())
            expected = {
                "author_batch_size": 100,
                "batch_size_requested": 50,
                "original_instance_batch_size": 50,
                "effective_batch_sizes": [50, 50],
                "number_of_batches": 2,
                "batch_override_reason": LEGACY_CVRP2000_OVERRIDE_REASON,
            }
            self.assertEqual(metadata["solver_execution_batch_identity"], expected)
            self.assertEqual(summary["solver_execution_batch_identity"], expected)
            self.assertEqual(
                metadata["derived_from"]["solver_execution_batch_identity"], expected)
            self.assertEqual(metadata["protocol"]["batch_size_requested"], 100)
            self.assertEqual(summary["batch_size_requested"], 50)

    def test_lehd_cvrp2000_legacy_bs100_claim_conflicting_with_protocol_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            legacy_root, quality = make_legacy_cvrp2000_fixture(root)
            path = quality / "summary.json"
            summary = json.loads(path.read_text())
            summary.update({
                "batch_size_requested": 100,
                "original_instance_batch_size": 100,
                "effective_batch_sizes": [100],
                "number_of_batches": 1,
            })
            path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "batch mapping mismatch"):
                build_lehd_rebind_plan(
                    legacy_root=legacy_root, output_root=root / "output",
                    problem="cvrp", problem_size=2000,
                    current_project={"commit": "current", "dirty": False},
                    rebind_source_files=[])

    def test_lehd_cvrp2000_override_reason_is_required_and_semantic(self):
        for reason in (None, "", "manual batch override"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                legacy_root, quality = make_legacy_cvrp2000_fixture(root)
                path = quality / "metadata.json"
                metadata = json.loads(path.read_text())
                metadata["protocol"]["batch_override_reason"] = reason
                path.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, "override reason"):
                    build_lehd_rebind_plan(
                        legacy_root=legacy_root, output_root=root / "output",
                        problem="cvrp", problem_size=2000,
                        current_project={"commit": "current", "dirty": False},
                        rebind_source_files=[])

    def test_lehd_legacy_batch_override_is_isolated_to_cvrp2000(self):
        self.assertEqual(LEGACY_RRC50_BATCH_OVERRIDES,
                         {("cvrp", 2000): {"batch_size": 50}})
        for key, entry in LEHD_AUTHOR_BATCH_REGISTRY.items():
            protocol = (lehd_legacy_protocol(*key, LEGACY_CVRP2000_OVERRIDE_REASON)
                        if key == ("cvrp", 2000) else lehd_legacy_protocol(*key))
            expected = 50 if key == ("cvrp", 2000) else entry["batch_size"]
            self.assertEqual(_legacy_batch_size(*key), expected)
            self.assertEqual(protocol["batch_size_requested"], expected)
            self.assertEqual(protocol["author_batch_size"], entry["batch_size"])
        with self.assertRaisesRegex(ValueError, "not allowed"):
            lehd_legacy_protocol("tsp", 100, LEGACY_CVRP2000_OVERRIDE_REASON)


class LEHDParallelTests(unittest.TestCase):
    def test_exact_formal_matrix(self):
        for problem, sizes, parallel_batch in (
                ("tsp", (100, 500, 1000), 128),
                ("cvrp", (50, 100, 200), 100)):
            for size in sizes:
                for protocol in ("fewer", "more"):
                    for batch_size in (1, parallel_batch):
                        with self.subTest(problem=problem, size=size,
                                          protocol=protocol, batch=batch_size):
                            scope = parallel_scope(problem, size, protocol, batch_size)
                            self.assertEqual(scope["RRC_budget"],
                                             20 if protocol == "fewer" else 50)
        for args in (("tsp", 100, "fewer", 100),
                     ("cvrp", 50, "fewer", 128),
                     ("tsp", 2000, "more", 128),
                     ("cvrp", 500, "more", 100)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                parallel_scope(*args)

    def test_batch_split_and_runtime_aggregation(self):
        self.assertEqual(len(batch_ranges(1280, 128)), 10)
        self.assertEqual(len(batch_ranges(10000, 100)), 100)
        self.assertEqual(batch_ranges(5, 2), [(0, 2), (2, 4), (4, 5)])
        scope = {"expected_dataset_count": 5, "requested_batch_size": 2,
                 "problem": "TSP", "problem_size": 100,
                 "protocol": "fewer", "RRC_budget": 20}
        records = []
        for index in range(5):
            records.append({
                "dataset_instance_index": index, "evidence_status": "KIT_VALIDATED",
                "independent_feasible": True, "kit_feasible": True,
                "official_vs_independent": {"pass": True},
                "independent_vs_kit": {"pass": True},
                "independent_objective": index + 2.0,
                "reference_objective": index + 1.0,
                "gap_percent": 10.0,
            })
        timings = [
            {"batch_index": index, "dataset_index_start": start,
             "dataset_index_stop_exclusive": stop,
             "original_instance_count": stop - start,
             "solver_runtime_seconds": runtime}
            for index, ((start, stop), runtime) in enumerate(
                zip(batch_ranges(5, 2), (1.0, 2.0, 3.0)))
        ]
        summary = summarize_measured(records, timings, scope=scope)
        self.assertEqual(summary["effective_batch_sizes"], [2, 2, 1])
        self.assertEqual(summary["total_solver_runtime_seconds"], 6.0)
        self.assertEqual(summary["mean_batch_runtime_seconds"], 2.0)
        self.assertEqual(summary["median_batch_runtime_seconds"], 2.0)
        self.assertTrue(summary["parallel_table_eligible"])
        bad = deepcopy(records)
        bad[2]["kit_feasible"] = False
        with self.assertRaises(ValueError):
            summarize_measured(bad, timings, scope=scope)

    def test_derived_bs1_positive_and_fail_closed(self):
        scope = {"expected_dataset_count": 3, "requested_batch_size": 1,
                 "problem": "TSP", "problem_size": 100,
                 "protocol": "more", "RRC_budget": 50}
        quality_summary = {
            "status": "KIT_VALIDATED", "protocol": "more", "RRC_budget": 50,
            "validated_count": 3, "failed_count": 0, "mean_objective": 2.0,
            "mean_reference_objective": 1.9, "mean_instance_gap_percent": 5.0,
            "dataset_sha256": "d" * 64, "checkpoint_sha256": "c" * 64,
            "upstream": {"commit": UPSTREAMS["LEHD"], "dirty": False},
        }
        timing_summary = {
            "status": "KIT_VALIDATED", "protocol": "more", "RRC_budget": 50,
            "count": 3, "validated_count": 3, "failed_count": 0,
            "mean_runtime_seconds": 2.0, "individual_runtimes": [1.0, 2.0, 3.0],
            "dataset_sha256": "d" * 64, "checkpoint_sha256": "c" * 64,
        }
        quality = {"summary": quality_summary, "directory": Path("/quality"),
                   "hashes": {"summary": "1", "metadata": "2", "records": "3"},
                   "dataset": {"sha256": "d" * 64},
                   "checkpoint": {"sha256": "c" * 64}}
        timing = {"summary": timing_summary, "path": Path("/timing.json"),
                  "sha256": "4"}
        _, summary = derive_bs1_summary(
            scope=scope, quality=quality, timing=timing,
            current_project={"commit": "new", "dirty": False})
        self.assertEqual(summary["artifact_generation_mode"], DERIVED_MODE)
        self.assertEqual(summary["mean_objective"], 2.0)
        self.assertEqual(summary["mean_batch_runtime_seconds"], 2.0)
        self.assertEqual(summary["total_solver_runtime_seconds"], 6.0)
        cases = (
            ("quality protocol", quality_summary, "protocol", "fewer"),
            ("timing protocol", timing_summary, "protocol", "fewer"),
            ("quality budget", quality_summary, "RRC_budget", 20),
            ("dataset", timing_summary, "dataset_sha256", "x" * 64),
            ("checkpoint", timing_summary, "checkpoint_sha256", "x" * 64),
            ("count", timing_summary, "count", 2),
            ("invalid timing", timing_summary, "mean_runtime_seconds", 0),
            ("quality status", quality_summary, "status", "FAILED"),
        )
        for name, target, key, value in cases:
            original = target[key]
            target[key] = value
            with self.subTest(case=name), self.assertRaises(ValueError):
                derive_bs1_summary(
                    scope=scope, quality=quality, timing=timing,
                    current_project={"commit": "new", "dirty": False})
            target[key] = original


if __name__ == "__main__":
    unittest.main()
