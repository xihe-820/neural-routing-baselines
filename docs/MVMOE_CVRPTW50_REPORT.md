# MVMoE/4E + CVRPTW50 integration report

Status: `LOCAL_VERIFIED_INTEGRATION_SERVER_NOT_RUN`. Scope stops at CVRPTW50; CVRPTW100 and other methods were not implemented.

## Identity and native mapping

- Official upstream: `https://github.com/RoyalSkye/Routing-MVMoE` at `af29e5af0595f94f3ecc3bc46d72df1089a62682`, clean before and after.
- Checkpoint: `pretrained/mvmoe_4e_n50/epoch-5000.pt`, SHA256 `3417f302fbddf232fd19a2a886cd1c7f44290b6d8c7280fcb0ae3777eeed3192`, strict-loaded as MOE/4E `Train_ALL`, epoch 5000.
- Dataset: `cvrptw50_pyvrp-10s_16.038.pkl`, 1000 tasks, SHA256 `a16975d9dd242739973191256e1cdbd8166c4405759ef195b985a8c3ec49df40`.
- Native fields: depot `[B,1,2]`, customers `[B,50,2]`, demand `raw/capacity` exactly once, and customer-only service/TW formed by stripping depot row 0 from benchmark-inclusive arrays.
- Coordinates, TW and service are not scaled. Speed is 1.0 and objective is unrounded closed-route Euclidean distance.

The formal dataset uses raw capacity 40, depot service 0, task threshold `1e-4`, and depot TW `[0,4.599999904632568]`. The adapter rejects mixed depot windows. The runner assigns both `env.depot_start` and `env.depot_end` before `load_problems`, following the official Tester's depot-horizon override mechanism while retaining the benchmark lower bound. It never uses the environment's default upper bound 3.

## Smoke progression

Reduced aug1 first-2 passed 2/2 four-part completion gates; total rollout was 1.1099848s. Official search/decode configuration aug8/POMO50 first-2 passed 2/2; total rollout was 1.0406155s.

The final aug8/POMO50 first-5 ran in one batch. `selected_node_list` had shape `[40,50,72]`, reward `[40,50]`, and total rollout was 0.8995715s. Timings are engineering evidence, not paper-comparable.

| index | reported | independent | reported agrees | Kit | Kit agrees | reference | gap % | feasible | best aug / POMO |
|---:|---:|---:|:---:|---:|:---:|---:|---:|:---:|---:|
| 0 | 14.086605072 | 14.086604810 | true | 14.086606026 | true | 13.199351311 | 6.721948 | true / true | 6 / 19 |
| 1 | 15.332053185 | 15.332053087 | true | 15.332053185 | true | 14.116131783 | 8.613700 | true / true | 6 / 16 |
| 2 | 16.374761581 | 16.374761188 | true | 16.374761581 | true | 14.913759232 | 9.796336 | true / true | 5 / 16 |
| 3 | 20.872287750 | 20.872285863 | true | 20.872289658 | true | 17.828571320 | 17.072117 | true / true | 3 / 27 |
| 4 | 15.760301590 | 15.760301681 | true | 15.760303497 | true | 14.198093414 | 11.002944 | true / true | 6 / 27 |

All five actual routes visit every customer once, respect capacity, start customer service within its window after any waiting, add service duration before the next leg, and return by the depot upper bound under the task tolerance. The maximum route return times for indices 0–4 are respectively `4.143523`, `4.379849`, `4.339021`, `4.359311`, and `4.317549`, all below 4.6.

Index 0 actual canonical solution:

```text
[0,20,34,32,36,31,0,8,26,16,42,2,0,48,27,28,43,50,17,6,0,15,12,41,0,40,9,4,29,0,47,22,35,44,14,0,10,13,33,19,46,0,25,21,49,0,45,30,38,24,7,0,39,11,23,0,3,18,37,1,0,5,0]
```

Model environment: Python 3.10.20, PyTorch 2.5.1 with CUDA build 12.1, NumPy 1.24.3, NVIDIA GeForce RTX 4060 Laptop GPU. Kit validation used the existing Python 3.10.20 / PyTorch 2.12.0+cpu / NumPy 2.2.6 / ML4CO-Kit 0.5.4 environment.

The complete reference-enabled unit suite passed all 65 tests after the integration changes.

The compact evidence is [mvmoe_cvrptw50.json](../manifests/mvmoe_cvrptw50.json); the full validated artifact is ignored from Git and has SHA256 `e72ede28ca97afb7f75fe2e14f0ffe5823e671e04e540acee1c13aad44d5d89c`. Server validation remains `NOT_RUN`.
