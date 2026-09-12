#!/usr/bin/env python3
"""Single-repository import / strict CPU model-load audit; never inference.

Constructor settings mirror Routing-MVMoE test.py or NeuOpt options.py/PPO.
Run one method per process; results apply only to the recorded interpreter.
"""
import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.hashing import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--module", action="append", default=[])
    parser.add_argument("--module-path", action="append", default=[],
                        help="Additional relative module directory within this same official repository")
    parser.add_argument("--method", choices=("MVMoE", "NeuOpt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    for relative in args.module_path:
        module_path = (repo / relative).resolve()
        if not module_path.is_relative_to(repo):
            parser.error("--module-path must stay within the single --repo")
        sys.path.insert(0, str(module_path))
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "hostname": platform.node(),
              "executable": sys.executable, "python": sys.version, "repo": str(repo),
              "module_paths": args.module_path,
              "modules": {}, "loads": [], "inference_performed": False}
    for name in args.module:
        try:
            mod = importlib.import_module(name)
            report["modules"][name] = {"import_ok": True, "path": mod.__file__}
        except Exception as exc:
            report["modules"][name] = {"import_ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if args.method:
        import torch
        report["torch"] = torch.__version__
        for n in (50, 100):
            item = {"size": n, "strict_load": False}
            report["loads"].append(item)
            try:
                if args.method == "MVMoE":
                    from models.MOEModel import MOEModel
                    path = repo / f"pretrained/mvmoe_4e_n{n}/epoch-5000.pt"
                    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                    config = dict(embedding_dim=128, sqrt_embedding_dim=128**0.5,
                                  encoder_layer_num=6, decoder_layer_num=1, qkv_dim=16,
                                  head_num=8, logit_clipping=10, ff_hidden_dim=512,
                                  num_experts=4, eval_type="argmax", norm="instance", norm_loc="norm_last",
                                  expert_loc=[f"Enc{i}" for i in range(6)] + ["Dec"],
                                  problem=checkpoint["problem"], topk=2, routing_level="node",
                                  routing_method="input_choice", device="cpu")
                    model = MOEModel(**config)
                    state = checkpoint["model_state_dict"]
                    item["config_source"] = "test.py:args2dict and argument defaults; explicit supported device=cpu"
                else:
                    from nets.actor_network import Actor
                    path = repo / f"pre-trained/cvrp{n}.pt"
                    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                    seq_length = 70 if n == 50 else 120
                    config = dict(embedding_dim=128, hidden_dim=128, n_heads_actor=4, n_layers=3,
                                  normalization="layer", v_range=6., seq_length=seq_length, k=4,
                                  with_RNN=True, with_feature1=True, with_feature3=True, with_simpleMDP=True)
                    # Actor.__init__ reads only problem.NAME. No environment or
                    # initial solution is constructed during this load audit.
                    model = Actor(problem=SimpleNamespace(NAME="cvrp"), **config)
                    state = checkpoint["actor"]
                    item["config_source"] = "options.py and agent/ppo.py; README dummy_rate 0.4/0.2"
                result = model.load_state_dict(state, strict=True)
                model.eval()
                item.update(strict_load=True, config=config, checkpoint=str(path), sha256=sha256_file(path),
                            missing_keys=list(result.missing_keys), unexpected_keys=list(result.unexpected_keys),
                            parameters=sum(p.numel() for p in model.parameters()))
            except Exception as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
