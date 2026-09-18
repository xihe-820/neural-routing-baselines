import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from methods.neuopt.tsp.config import supported_config
from methods.neuopt.tsp.compat import configure_decoder
from methods.neuopt.tsp.production_eval import (
    official_option_args, timed_rollout_with_evidence_replay,
)
from methods.neuopt.tsp.paper_protocol import (
    FINAL_T_BY_BUDGET, FORMAL_BATCH_SIZES, UPSTREAM_COMMIT, formal_protocol,
    protocol_fingerprint,
)
from methods.neuopt.tsp.production_results import (
    BATCH_TIMINGS_FILE, METADATA_FILE, PAPER_RESULT_SCHEMA_VERSION, RECORDS_FILE,
    RNG_POLICY, TIMED_REPLAY, TIMING_SEMANTICS, WARMUP_POLICY,
    append_batch, build_paper_result,
    build_result_matrix, finalize_run, initialize_or_resume, load_run,
    mark_failed, read_paper_result, validate_identity, write_new_json,
)


def fake_identity(*, budget="fewer", batch_size=1, scope="preflight"):
    config = supported_config(100)
    protocol = formal_protocol(
        100, budget=budget, batch_size=batch_size,
        T_max=FINAL_T_BY_BUDGET[budget])
    count = batch_size if scope == "preflight" else config["dataset_count"]
    return {
        "method": "NeuOpt", "variant": "NeuOpt-GIRE", "problem": "TSP",
        "problem_size": 100, "paper_protocol": protocol,
        "protocol_fingerprint": protocol_fingerprint(protocol),
        "scope": scope, "expected_indices": list(range(count)),
        "native_batch_count": count // batch_size,
        "artifact_class": "formal_production_separate_from_calibration",
        "project": {"commit": "1" * 40, "dirty": False,
                    "url": "https://github.com/xihe-820/neural-routing-baselines"},
        "upstream": {"commit": UPSTREAM_COMMIT, "dirty": False,
                     "url": "https://github.com/yining043/NeuOpt"},
        "upstream_checkout_path": "/upstream",
        "checkpoint": {"path": "/upstream/pre-trained/tsp100.pt",
                       "filename": "tsp100.pt", "relative_path": "pre-trained/tsp100.pt",
                       "sha256": config["checkpoint_sha256"]},
        "dataset": {"path": "/data/" + config["dataset_filename"],
                    "filename": config["dataset_filename"],
                    "sha256": config["dataset_sha256"], "count": config["dataset_count"]},
        "adapter_mapping": {
            "problem_size": 100, "source_shape": [batch_size, 100, 2],
            "source_dtype": "float32", "model_input_dtype": "float32",
            "dtype_cast": False,
            "coordinate_transformation":
                "none; no normalization, regeneration, or node reordering",
        },
        "native_batching": {"one_official_rollout_call_per_batch": True,
                            "bs1_loop_emulation": False, "partial_batches": False},
        "warmup": dict(WARMUP_POLICY), "rng_policy": RNG_POLICY,
        "environment": {"gpu": "NVIDIA GeForce RTX 4090", "device": "cuda:0"},
        "tensorboard_compatibility": {
            "tensorboard_logger_available": False,
            "tensorboard_logger_import_shim": True,
            "official_source_modified": False,
        },
        "decoder_compatibility": {
            "bs1_shape_shim": batch_size == 1,
            "original_batch_size": batch_size, "D2A": 1, "val_m": 1,
            "internal_decoder_batch_size": batch_size,
            "official_source_modified": False,
            "action_reward_logits_rng_budget_changed": False,
        },
        "record_compatibility": {
            "record_step_shim": True,
            "pinned_step_sha256": "62775b74cf54c4d22f7a34ef93e0f6b2d8e26a88b9e8a7e7845e419b7dd0448d",
            "timed_rollout_shim_active": False,
            "evidence_replay_shim_active": True,
            "official_source_modified": False,
            "action_reward_logits_rng_budget_changed": False,
        },
        "source_provenance": [], "timing_semantics": TIMING_SEMANTICS,
    }


def fake_record(index, *, batch_size, objective=None, reference=9.0):
    objective = float(10.0 + index / 1000 if objective is None else objective)
    successor = list(range(1, 100)) + [0]
    canonical = list(range(100)) + [0]
    return {
        "dataset_instance_index": index, "instance_id": f"tsp100-{index}",
        "batch_index": index // batch_size, "position_in_batch": index % batch_size,
        "canonical_solution": canonical, "successor": successor,
        "reported_objective": objective, "timed_official_objective": objective,
        "replay_official_objective": objective,
        "official_recomputed_objective": objective,
        "independent_objective": objective, "reference_objective": reference,
        "kit_reference_objective": reference,
        "independent_reference_objective": reference,
        "benchmark_reference_agrees": True,
        "gap_percent": (objective - reference) / reference * 100.0,
        "independent_feasible": True, "reported_objective_agrees": True,
        "kit_feasible": True, "kit_objective": objective,
        "kit_objective_agrees": True,
        "constraint_details": {"each_node_once": True, "single_cycle": True},
        "source_coordinate_dtype": "float32", "model_input_dtype": "float32",
        "model_input_dtype_cast": False,
        "evidence_status": "KIT_VALIDATED",
    }


def fake_timing(batch_index, batch_size, *, runtime=0.25):
    start = batch_index * batch_size
    return {
        "batch_index": batch_index,
        "dataset_indices": list(range(start, start + batch_size)),
        "batch_size": batch_size, "runtime_seconds": runtime,
        "timed_replay": dict(TIMED_REPLAY),
        "rng_before_sha256": "a" * 64, "rng_after_sha256": "a" * 64,
        "native_vectorized_batch": {
            "one_official_rollout_call": True,
            "original_instances_in_call": batch_size,
            "bs1_loop_emulation": False,
        },
    }


def complete_run(directory, *, budget="fewer", batch_size=1, scope="preflight"):
    identity = fake_identity(budget=budget, batch_size=batch_size, scope=scope)
    initialize_or_resume(directory, identity)
    for batch_index in range(identity["native_batch_count"]):
        start = batch_index * batch_size
        records = [fake_record(index, batch_size=batch_size)
                   for index in range(start, start + batch_size)]
        append_batch(directory, records, fake_timing(batch_index, batch_size))
    return finalize_run(directory)


class NeuOptTSPFormalProtocolTests(unittest.TestCase):
    def test_only_exact_human_frozen_protocols(self):
        for budget, expected_t in FINAL_T_BY_BUDGET.items():
            for batch_size in FORMAL_BATCH_SIZES:
                protocol = formal_protocol(
                    100, budget=budget, batch_size=batch_size, T_max=expected_t)
                self.assertEqual(protocol["T_max"], expected_t)
                self.assertEqual(protocol["D2A"], 1)
                self.assertEqual((protocol["stall_limit"], protocol["k"]), (10, 4))
        for kwargs in (
                {"budget": "fewer", "batch_size": 1, "T_max": 5},
                {"budget": "more", "batch_size": 1, "T_max": 1},
                {"budget": "fewer", "batch_size": 8, "T_max": 1},
                {"budget": "fewer", "batch_size": 1, "T_max": 1, "d2a": 5},
                {"budget": "fewer", "batch_size": 1, "T_max": True},
                {"budget": "fewer", "batch_size": True, "T_max": 1},
                {"budget": "fewer", "batch_size": 1, "T_max": 1, "stall_limit": 9},
                {"budget": "fewer", "batch_size": 1, "T_max": 1, "k": 3}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                formal_protocol(100, **kwargs)

    def test_identity_fails_closed_on_assets_commits_dirty_gpu_and_batching(self):
        mutations = (
            (lambda value: value["checkpoint"].update(sha256="0" * 64), "identity"),
            (lambda value: value["dataset"].update(sha256="0" * 64), "identity"),
            (lambda value: value["upstream"].update(commit="0" * 40), "identity"),
            (lambda value: value["project"].update(dirty=True), "identity"),
            (lambda value: value["upstream"].update(dirty=True), "identity"),
            (lambda value: value["environment"].update(gpu="A100"), "RTX 4090"),
            (lambda value: value.update(expected_indices=[0, 2]), "scope"),
        )
        for mutate, message in mutations:
            identity = fake_identity()
            mutate(identity)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_identity(identity)

    def test_official_options_preserve_native_batch_size(self):
        class Device:
            type = "cuda"
        config = formal_protocol(100, budget="more", batch_size=128, T_max=5)
        args = official_option_args(
            config=config, checkpoint="tsp100.pt", device=Device(), batch_size=128)
        value = lambda flag: args[args.index(flag) + 1]
        self.assertEqual(value("--val_size"), "128")
        self.assertEqual(value("--val_batch_size"), "128")
        self.assertEqual(value("--val_m"), "1")
        self.assertEqual(value("--T_max"), "5")

    def test_bs16_and_bs128_require_unmodified_native_decoder(self):
        class Decoder:
            def forward(self):
                return None
        source = "def forward(self):\n    return None\n"
        digest = hashlib.sha256(source.encode()).hexdigest()
        with (patch("methods.neuopt.tsp.compat.inspect.getsource", return_value=source),
              patch("methods.neuopt.tsp.compat.PINNED_DECODER_FORWARD_SHA256", digest)):
            for batch_size in (16, 128):
                provenance = configure_decoder(
                    Decoder, original_batch_size=batch_size, val_m=1)
                self.assertFalse(provenance["bs1_shape_shim"])
                self.assertEqual(provenance["internal_decoder_batch_size"], batch_size)


class NeuOptTSPProductionTimedReplayTests(unittest.TestCase):
    def test_one_native_timed_call_then_one_untimed_replay(self):
        import torch

        class Agent:
            def __init__(self):
                self.records = []

            def rollout(self, problem, **kwargs):
                self.records.append(kwargs["record"])
                torch.rand(1)
                return (torch.ones(16), None, None,
                        ([], [], []) if kwargs["record"] else None)

        agent = Agent()
        native = {"coordinates": torch.zeros(16, 100, 2)}
        config = formal_protocol(100, budget="fewer", batch_size=16, T_max=1)
        with (patch.object(torch.cuda, "synchronize"),
              patch.object(torch.cuda, "get_rng_state_all", return_value=[]),
              patch.object(torch.cuda, "set_rng_state_all"),
              patch("methods.neuopt.tsp.production_eval.time.perf_counter",
                    side_effect=[4.0, 4.5]),
              patch("methods.neuopt.tsp.production_eval.replay_record_compatibility") as compat):
            compat.return_value.__enter__.return_value = {
                "record_step_shim": True,
            }
            timed, replay, elapsed, provenance, _, rng_fingerprints = timed_rollout_with_evidence_replay(
                agent, object(), native, config=config, batch_size=16,
                device="cuda:0", torch=torch)
        self.assertEqual(agent.records, [False, True])
        self.assertTrue(torch.equal(timed, replay[0]))
        self.assertEqual(elapsed, 0.5)
        self.assertEqual(provenance, TIMED_REPLAY)
        self.assertEqual(len(rng_fingerprints["before_sha256"]), 64)
        self.assertEqual(len(rng_fingerprints["after_sha256"]), 64)


class NeuOptTSPProductionArtifactTests(unittest.TestCase):
    def test_preflight_is_one_native_batch_and_never_paper_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "preflight"
            summary = complete_run(directory, batch_size=16)
            metadata, records, timings, loaded = load_run(directory)
            self.assertEqual((len(records), len(timings)), (16, 1))
            self.assertEqual(timings[0]["native_vectorized_batch"], {
                "one_official_rollout_call": True,
                "original_instances_in_call": 16,
                "bs1_loop_emulation": False,
            })
            self.assertFalse(metadata["paper_ready"])
            self.assertFalse(summary["paper_ready"])
            self.assertEqual(summary, loaded)
            with self.assertRaisesRegex(ValueError, "fullset"):
                build_paper_result(directory)

    def test_resume_only_exact_in_progress_and_rolls_back_uncommitted_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "resume"
            identity = fake_identity(batch_size=16)
            initialize_or_resume(directory, identity)
            records = [fake_record(i, batch_size=16) for i in range(16)]
            (directory / RECORDS_FILE).write_text(
                "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records))
            metadata, recovered_records, timings = initialize_or_resume(directory, identity)
            self.assertEqual((recovered_records, timings), ([], []))
            self.assertEqual(metadata["state"], "IN_PROGRESS")
            wrong = copy.deepcopy(identity)
            wrong["project"]["commit"] = "2" * 40
            with self.assertRaisesRegex(ValueError, "resume refused"):
                initialize_or_resume(directory, wrong)

    def test_existing_empty_directory_and_failed_run_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(ValueError, "must not already exist"):
                initialize_or_resume(empty, fake_identity())
            failed = Path(tmp) / "failed"
            identity = fake_identity()
            initialize_or_resume(failed, identity)
            mark_failed(failed, failure_type="CUDA_OUT_OF_MEMORY", message="oom")
            with self.assertRaisesRegex(ValueError, "exact IN_PROGRESS"):
                initialize_or_resume(failed, identity)

    def test_record_or_native_batch_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "tamper"
            complete_run(directory, batch_size=16)
            records_path = directory / RECORDS_FILE
            records = [json.loads(line) for line in records_path.read_text().splitlines()]
            records[0]["dataset_instance_index"] = 1
            records_path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_run(directory)

    def test_discontinuous_rng_state_is_rejected_before_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "rng"
            identity = fake_identity(batch_size=128, scope="fullset")
            initialize_or_resume(directory, identity)
            append_batch(
                directory,
                [fake_record(index, batch_size=128) for index in range(128)],
                fake_timing(0, 128))
            second = fake_timing(1, 128)
            second["rng_before_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "RNG stream"):
                append_batch(
                    directory,
                    [fake_record(index, batch_size=128) for index in range(128, 256)],
                    second)

    def test_mean_gap_is_mean_of_per_instance_gaps_and_time_is_native_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "stats"
            identity = fake_identity(batch_size=1)
            initialize_or_resume(directory, identity)
            first = fake_record(0, batch_size=1, objective=10.0, reference=5.0)
            append_batch(directory, [first], fake_timing(0, 1, runtime=0.4))
            summary = finalize_run(directory)
            self.assertEqual(summary["drop_mean_per_instance_gap_percent"], 100.0)
            self.assertEqual(summary["time_mean_native_batch_latency_seconds"], 0.4)
            self.assertEqual(summary["total_native_batch_solver_seconds"], 0.4)


class NeuOptTSPPaperResultTests(unittest.TestCase):
    def test_fullset_result_and_exact_six_cell_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "full"
            summary = complete_run(run, budget="fewer", batch_size=128, scope="fullset")
            self.assertTrue(summary["paper_ready"])
            result = build_paper_result(run)
            self.assertEqual(result["schema_version"], PAPER_RESULT_SCHEMA_VERSION)
            self.assertEqual((result["instance_count"], result["native_batch_count"]),
                             (1280, 10))
            result_path = root / "fewer_bs128.json"
            write_new_json(result_path, result)
            self.assertEqual(read_paper_result(result_path), result)

            rows = []
            for budget in ("fewer", "more"):
                for batch_size in FORMAL_BATCH_SIZES:
                    row = copy.deepcopy(result)
                    row["budget"] = budget
                    row["T_max"] = FINAL_T_BY_BUDGET[budget]
                    row["batch_size"] = batch_size
                    row["native_batch_count"] = 1280 // batch_size
                    rows.append(row)
            with patch(
                    "methods.neuopt.tsp.production_results.read_paper_result",
                    side_effect=rows):
                matrix = build_result_matrix([root / f"result-{i}.json" for i in range(6)])
            self.assertTrue(matrix["paper_ready"])
            self.assertEqual(len(matrix["rows"]), 6)
            self.assertEqual(matrix["main_complete_results_source"], "BS1")

    def test_result_reader_rejects_duplicate_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = supported_config(100)
            prototype = {
                "schema_version": PAPER_RESULT_SCHEMA_VERSION,
                "status": "PAPER_READY", "paper_ready": True,
                "problem": "TSP", "problem_size": 100, "budget": "fewer",
                "T_max": 1, "D2A": 1, "batch_size": 1,
                "Obj": 1.0, "Drop_percent": 0.0, "Time_seconds": 1.0,
                "Total_seconds": 1280.0, "instance_count": 1280,
                "native_batch_count": 1280,
                "project": {"commit": "1" * 40},
                "upstream": {"commit": UPSTREAM_COMMIT},
                "checkpoint_sha256": config["checkpoint_sha256"],
                "dataset_sha256": config["dataset_sha256"],
            }
            with patch(
                    "methods.neuopt.tsp.production_results.read_paper_result",
                    side_effect=[copy.deepcopy(prototype) for _ in range(6)]):
                with self.assertRaisesRegex(ValueError, "exactly"):
                    build_result_matrix([root / f"duplicate-{i}.json" for i in range(6)])


if __name__ == "__main__":
    unittest.main()
