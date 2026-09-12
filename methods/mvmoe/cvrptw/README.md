# MVMoE/4E / CVRPTW50 — LOCAL_VERIFIED integration

This directory implements only MVMoE/4E CVRPTW50. It has its own config, adapter, decoder, preparation, runner and Kit validation code; it does not import the CVRP adapter/decoder and does not implement CVRPTW100.

The integration pins Routing-MVMoE commit `af29e5af0595f94f3ecc3bc46d72df1089a62682`, dataset SHA256 `a16975d9dd242739973191256e1cdbd8166c4405759ef195b985a8c3ec49df40`, and n50 checkpoint SHA256 `3417f302fbddf232fd19a2a886cd1c7f44290b6d8c7280fcb0ae3777eeed3192`.

Native mapping:

- `depot_xy`: benchmark depot → `[B,1,2]`
- `node_xy`: customer points → `[B,50,2]`
- `node_demand`: raw demand / raw capacity exactly once → `[B,50]`
- `service_time`: depot-inclusive `service[:,1:]` → `[B,50]`
- `tw_start`, `tw_end`: depot-inclusive `tw[:,1:,0/1]` → `[B,50]`
- `env.depot_start/end`: exact benchmark `tw[:,0,:]`, assigned before `load_problems`

Coordinates, time windows and service times are not scaled. A batch with differing depot windows is rejected; depot service must be zero. The actual dataset's 1000 tasks use depot window `[0,4.599999904632568]`, task threshold `1e-4`, speed 1 and unrounded Euclidean distance.

The runner strict-loads official MOE/4E (`Train_ALL`, epoch 5000), uses POMO50, argmax and seed2024 with the official CuDNN seeding side effects. It creates no optimizer and performs no training, backward or fine-tuning. The decoder gathers `selected_node_list` at the exact winning augmentation/POMO reward index, preserves internal depot separators, and removes only terminal finished-POMO padding.

The local RTX4060 aug8 first-5 run passed 5/5 independent feasibility, Kit feasibility, reported-objective agreement and Kit-objective agreement. Independent validation simulates arrival, waiting, service start/duration and final depot return using original benchmark data. See [the evidence manifest](../../../manifests/mvmoe_cvrptw50.json).

```bash
python -B methods/mvmoe/cvrptw/prepare_instances.py \
  --dataset "$CVRPTW50_DATASET" --problem-size 50 --offset 0 --count 5 \
  --output artifacts/mvmoe_cvrptw50/input_first5.npz

python -B methods/mvmoe/cvrptw/run.py \
  --problem-size 50 --input artifacts/mvmoe_cvrptw50/input_first5.npz \
  --upstream "$MVMOE_UPSTREAM" --checkpoint "$MVMOE_N50_CHECKPOINT" \
  --output artifacts/mvmoe_cvrptw50/official_search_aug8_first5.json \
  --aug-factor 8 --seed 2024 --device auto

python -B methods/mvmoe/cvrptw/validate_with_kit.py \
  --input artifacts/mvmoe_cvrptw50/official_search_aug8_first5.json \
  --dataset "$CVRPTW50_DATASET" \
  --output artifacts/mvmoe_cvrptw50/official_search_aug8_first5_validated.json
```

Runtime fields are engineering-smoke evidence: per-row values are amortized batch runtime, not single-instance latency and not paper-comparable. All CVRPTW50 server fields remain `NOT_RUN`.
