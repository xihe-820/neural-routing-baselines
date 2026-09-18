"""Direct runtime for the audited standalone CVRPTW SymNCO source snapshot."""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys

from common.cvrptw_runtime import timed_call, to_tensordict
from methods.symnco.cvrptw.adapter import raw_ids_from_native
from methods.symnco.cvrptw.config import validate_snapshot


SNAPSHOT_PACKAGE = "_neural_routing_baselines_symnco_snapshot"


def _load_snapshot_package(root):
    root = Path(root).resolve()
    package_dir = root / "baselines/cvrptw_symnco"
    existing = sys.modules.get(SNAPSHOT_PACKAGE)
    if existing is not None:
        existing_path = Path(existing.__file__).resolve().parent
        if existing_path != package_dir:
            raise ImportError("a different SymNCO source snapshot is already loaded")
        return existing
    spec = importlib.util.spec_from_file_location(
        SNAPSHOT_PACKAGE, package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)])
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the audited SymNCO source snapshot")
    module = importlib.util.module_from_spec(spec)
    sys.modules[SNAPSHOT_PACKAGE] = module
    spec.loader.exec_module(module)
    return module


def select_e1_batch(samples, actions, lengths, select_candidates):
    """Run historical selection independently within each original's [A,K] block."""
    if actions.ndim != 4 or lengths.ndim != 3 or tuple(actions.shape[:3]) != tuple(lengths.shape):
        raise RuntimeError("unexpected SymNCO action/length batch layout")
    batch_size, augmentations, rollouts = lengths.shape
    if augmentations != 4 or rollouts != 1 or len(samples) != batch_size:
        raise RuntimeError("SymNCO E1 requires per-original [A=4,K=1] candidates")
    selections = []
    for batch_index, sample in enumerate(samples):
        selected = select_candidates(sample, actions[batch_index], lengths[batch_index])
        if not selected.get("feasible") or selected.get("tour") is None:
            raise RuntimeError(f"SymNCO E1 found no feasible candidate for batch row {batch_index}")
        augmentation, rollout, legacy_zero = selected["candidate"]
        if legacy_zero != 0:
            raise RuntimeError("unexpected historical SymNCO candidate sentinel")
        selections.append({
            "raw_action": [int(node) for node in selected["tour"]],
            "official_reward": -float(selected["cost"]),
            "instance_id": sample["instance_id"],
            "selected_candidate": {
                "augmentation_index": int(augmentation),
                "candidate_index": int(rollout),
                "rollout_index": int(rollout),
                "flat_index": int(augmentation * rollouts + rollout),
                "layout": "per-original [augmentation,rollout] with A=4,K=1",
            },
        })
    return selections


class Runtime:
    def __init__(self, upstream, checkpoint, problem_size, device, torch):
        self.upstream = Path(upstream).resolve()
        self.problem_size, self.device, self.torch = problem_size, device, torch
        self.snapshot_identity = validate_snapshot(self.upstream)
        _load_snapshot_package(self.upstream)
        self.data = importlib.import_module(f"{SNAPSHOT_PACKAGE}.data")
        self.evaluate = importlib.import_module(f"{SNAPSHOT_PACKAGE}.evaluate")
        self.model, payload = self.evaluate.load_model(checkpoint, device)
        self.model.eval()
        config = payload.get("config", {})
        if config.get("nodes_num") != problem_size:
            raise ValueError("SymNCO checkpoint N does not match --problem-size")
        self.checkpoint_state = {
            "format": payload.get("format"), "architecture": payload.get("architecture"),
            "config": config, "model_state_keys": len(payload.get("model", {})),
            "load_strict": True, "missing_keys": [], "unexpected_keys": [],
            "source_manifest_sha256": self.snapshot_identity["manifest_sha256"],
        }

    def _inputs(self, native):
        instance_ids = raw_ids_from_native(native)
        samples = []
        batch_size = len(instance_ids)
        for index, instance_id in enumerate(instance_ids):
            sample = {
                key: self.torch.as_tensor(value[index]).cpu()
                for key, value in native.items()
            }
            sample["instance_id"] = instance_id
            samples.append(sample)
        td = to_tensordict(
            native, torch=self.torch, device=self.device,
            expected_batch_size=batch_size)
        return samples, td, instance_ids

    def _call(self, samples, td, instance_ids, *, augmentations):
        with self.torch.no_grad():
            result = self.model(
                td, a=augmentations, k=1, decode="greedy", seed=1234,
                instance_ids=instance_ids)
        actions, lengths = result["actions"].cpu(), result["lengths"].cpu()
        if augmentations == 1:
            selected = []
            for index, sample in enumerate(samples):
                row = self.evaluate.select_candidates(sample, actions[index], lengths[index])
                if not row.get("feasible"):
                    raise RuntimeError("SymNCO E0 warm-up produced no feasible candidate")
                selected.append(row)
            return selected
        return select_e1_batch(
            samples, actions, lengths, self.evaluate.select_candidates)

    def warmup_batch(self, native):
        samples, td, instance_ids = self._inputs(native)
        return timed_call(
            lambda: self._call(samples, td, instance_ids, augmentations=1),
            torch=self.torch, device=self.device, timed=False)[0]

    def official_format_smoke(self):
        line = self.data.synthetic_line(self.problem_size, seed=1234)
        sample = self.data.parse_raw(line, self.problem_size)
        td = self.data.collate_raw([sample]).to(self.device)
        result, _ = timed_call(
            lambda: self._call([sample], td, [sample["instance_id"]], augmentations=4),
            torch=self.torch, device=self.device, timed=False)
        return result[0]

    def solve(self, native, *, timed=True):
        selections, elapsed = self.solve_batch(native, timed=timed)
        if len(selections) != 1:
            raise RuntimeError("SymNCO solve() is the batch-one compatibility API")
        return selections[0], elapsed

    def solve_batch(self, native, *, timed=True):
        samples, td, instance_ids = self._inputs(native)
        return timed_call(
            lambda: self._call(samples, td, instance_ids, augmentations=4),
            torch=self.torch, device=self.device, timed=timed)
