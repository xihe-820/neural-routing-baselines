# MVMoE/4E / CVRP50+100 — LOCAL_VERIFIED integration

This directory implements the same method/problem adapter for the two official sizes `50` and `100`. It does not implement CVRPTW. Both sizes pin Routing-MVMoE commit `af29e5af0595f94f3ecc3bc46d72df1089a62682`, an exact ML4CO dataset SHA, an exact 4E checkpoint SHA, and the native capacity (`40` or `50`).

- `config.py` is the only size-to-asset identity table; filenames are never used to infer model identity.
- `prepare_instances.py` validates dataset SHA, task class, point shape and capacity, then writes `mvmoe-cvrp-input-v2` NPZ data with raw demands and capacity.
- `adapter.py` accepts only size 50/100 and applies `raw_demand / raw_capacity` exactly once for official `CVRPEnv.load_problems` input.
- `run.py` directly constructs official `MOEModel` and `CVRPEnv`, strict-loads the checkpoint, and follows `load_problems → reset → pre_forward → pre_step → model/step`. It creates no optimizer and performs no training, backward or fine-tuning.
- `decode.py` selects the exact reward's augmentation/POMO trajectory. It collapses only terminal finished-POMO depot padding; internal depot separators, duplicates and missing customers remain available to the validator.
- `validate_with_kit.py` adds independent ML4CO-Kit feasibility/objective evidence. Full `LOCAL_VERIFIED` requires independent feasibility, Kit feasibility, reported-objective agreement and Kit-objective agreement.

The model is official MOE/4E (not MOE_LIGHT): embedding 128, 6 encoder layers, 1 decoder layer, qkv 16, 8 heads, ff 512, 4 experts, top-k 2, node/input-choice routing, argmax, instance norm/norm_last, and experts at Enc0..5+Dec. The environment uses `problem_size=pomo_size=N`; seed 2024 also copies the two CuDNN side effects from `Routing-MVMoE/utils.py:seed_everything`. Reduced smoke uses aug1 and is explicitly debug-only. Completion uses the official search/decode configuration aug8, on only 2/5 instances; it is not the authors' complete evaluation command or a paper benchmark.

The local RTX4060 first-5 completion runs are 5/5 independently feasible and 5/5 Kit feasible for both sizes; every reported and Kit objective agrees with the independently recomputed objective at `rtol=atol=1e-6`. User-executed RTX4090 validation at project commit `f6db50e694cbab8870f4f1a1544856c49e9e0106` reproduced both sizes. See the [CVRP50 manifest](../../../manifests/mvmoe_cvrp50.json), [CVRP100 manifest](../../../manifests/mvmoe_cvrp100.json), and [server evidence](../../../manifests/server_mvmoe_cvrp.json).

```bash
# Set N=50 or N=100, with its matching pinned dataset and checkpoint.
python -B methods/mvmoe/cvrp/prepare_instances.py \
  --dataset "$CVRP_DATASET" --problem-size "$N" --offset 0 --count 5 \
  --output "artifacts/mvmoe_cvrp${N}/input_first5.npz"

python -B methods/mvmoe/cvrp/run.py \
  --problem-size "$N" --input "artifacts/mvmoe_cvrp${N}/input_first5.npz" \
  --upstream "$MVMOE_UPSTREAM" --checkpoint "$MVMOE_CHECKPOINT" \
  --output "artifacts/mvmoe_cvrp${N}/official_search_aug8_first5.json" \
  --aug-factor 8 --seed 2024 --device auto

python -B methods/mvmoe/cvrp/validate_with_kit.py \
  --input "artifacts/mvmoe_cvrp${N}/official_search_aug8_first5.json" \
  --dataset "$CVRP_DATASET" \
  --output "artifacts/mvmoe_cvrp${N}/official_search_aug8_first5_validated.json"
```

The result's per-row `runtime_seconds` is the total rollout time divided by the original batch size. It is an amortized engineering-smoke measurement, not single-instance latency and not paper-comparable.
