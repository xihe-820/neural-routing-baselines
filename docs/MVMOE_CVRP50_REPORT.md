# MVMoE/4E + CVRP50 integration report

Status: `LOCAL_VERIFIED_INTEGRATION_SERVER_NOT_RUN`. This report records the first committed CVRP50 integration at `54271cf4f97d7bd99784f6e6ee5ce52ee9f5994b`. The shared CVRP adapter has since been generalized to CVRP100; see [the CVRP100 report](MVMOE_CVRP100_REPORT.md). CVRPTW and other methods remain untouched.

## A. Changed files

- Result/provenance: `common/result_schema.py`, `common/provenance.py`, `common/README.md`.
- Method integration: `methods/mvmoe/cvrp/{adapter,decode,prepare_instances,run,validate_with_kit}.py` and README.
- Tests: `tests/test_result_schema.py`, `tests/test_mvmoe_cvrp.py`.
- Evidence/status: `manifests/mvmoe_cvrp50.json`, `manifests/status.json`, `docs/STATUS.md`.
- Minimal hygiene/runbook updates: README, AUDIT_REPORT, DESIGN, REPOSITORY_TREE, SCOPE, SERVER_RUNBOOK; corrected the already-confirmed CVRPTW `depot_end` fact in NATIVE_IO_AUDIT, SIZE_COMPATIBILITY and its method README.

No official source/checkpoint was edited. No dependency install, training, backward, or optimizer occurred. The work was later committed as `54271cf4f97d7bd99784f6e6ee5ce52ee9f5994b`.

## B. Tests

`ML4CO_REFERENCE_TESTS=1 ... -m unittest discover -s tests -v`: **44 tests passed** in the existing Kit environment. This includes 5 result-schema tests, 7 MVMoE adapter/decoder/selection tests, all prior validator/audit tests, and the six-file reference suite.

## C–D. Model/checkpoint identity

Official upstream: `af29e5af0595f94f3ecc3bc46d72df1089a62682`, clean before and after. Checkpoint `pretrained/mvmoe_4e_n50/epoch-5000.pt`, SHA256 `3417f302fbddf232fd19a2a886cd1c7f44290b6d8c7280fcb0ae3777eeed3192`.

Model is `MOEModel` / `model_type=MOE`, not MOE_LIGHT: embedding128; encoder6; decoder1; qkv16; heads8; clipping10; FF512; 4 experts; topk2; routing node/input_choice; argmax; instance norm/norm_last; experts Enc0..5+Dec; checkpoint problem Train_ALL. CVRP problem/pomo size 50, seed2024, loc_scaler null, fine_tune epochs0.

## E–F. Reduced and official search/decode configuration two-instance smoke

Both used the first two hash-pinned ML4CO CVRP50 instances and the RTX4060.

| Config / index | Reported | Independent | Kit | Independent / Kit feasible | Best aug / POMO |
|---|---:|---:|---:|---|---|
| reduced aug1 / 0 | 11.254352570 | 11.254353142 | 11.254355431 | true / true | 0 / 48 |
| reduced aug1 / 1 | 9.884812355 | 9.884812193 | 9.884811401 | true / true | 0 / 31 |
| official search/decode aug8 / 0 | 11.156522751 | 11.156523627 | 11.156524658 | true / true | 5 / 5 |
| official search/decode aug8 / 1 | 9.705957413 | 9.705957737 | 9.705956459 | true / true | 5 / 19 |

Reduced total rollout time was 0.8651s; official two-instance total was 0.7153s. These warm local engineering timings are not paper-comparable benchmarks.

## G. Official search/decode configuration first-five completion gate

Dataset SHA256: `eea12fbefe9c1bcc008d56ecfc1c50dadd64ac774f3547774c9fade8a7baa6c2`. All rows used aug8/POMO50/argmax/seed2024 and passed independent and Kit feasibility.

| Index | Reported | Independent | Kit | Reference | Gap % | Feasible | Best aug / POMO |
|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 11.156522751 | 11.156523627 | 11.156524658 | 10.973381996 | 1.668963 | true | 5 / 5 |
| 1 | 9.705957413 | 9.705957737 | 9.705956459 | 9.475135803 | 2.436080 | true | 5 / 19 |
| 2 | 10.951583862 | 10.951584171 | 10.951584816 | 10.940146446 | 0.104548 | true | 6 / 30 |
| 3 | 10.604280472 | 10.604281638 | 10.604281425 | 10.585029602 | 0.181880 | true | 1 / 13 |
| 4 | 12.467794418 | 12.467793850 | 12.467792511 | 12.329596519 | 1.120858 | true | 5 / 48 |

Maximum reported/independent absolute error: `1.1663e-6`; maximum Kit/independent absolute error: `1.3389e-6`. Five-instance total rollout time was 0.8043s. Full result artifact SHA256 is `001abf3ae3b5ef25f6ddb684560742750ad79a01ad509e8869fbd4828d13ba79`; compact committed evidence is [mvmoe_cvrp50.json](../manifests/mvmoe_cvrp50.json).

After generalizing the code to 50/100 and copying the official CuDNN seeding side effects, the final-source v2 CVRP50 first-5 run reproduced all five canonical routes and all three objective fields exactly. All 5/5 rows also passed the new four-part completion gate. The new total rollout time was 0.6829683s; its validated artifact SHA256 is `f8b41326d26a86c5221af6345e39d9d1aa07c2ed3204496f880e198120667ff2`.

## H–I. Actual solution and candidate selection

Index 0 canonical depot-separated route:

```text
[0,6,50,32,20,11,33,0,28,1,42,25,16,9,2,0,38,47,44,41,7,45,22,49,0,31,13,26,29,27,36,24,39,0,5,34,14,23,30,37,0,15,12,19,48,18,43,0,3,4,8,17,21,40,0,10,35,46,0]
```

Official rollout output had shape `[40,50,60]` for five original instances (`8*5` augmented batch, 50 POMOs). Reward was reshaped exactly as `[8,5,50]`; max POMO was selected inside each augmentation, then max augmentation for each original instance. The same indices gathered `selected_node_list` without rerunning the model. 8-fold transformations preserve node ordering. Decoder retained internal depot separators and collapsed only the terminal depot run caused by finished-POMO synchronization.

## J–L. Environment, cleanliness, status

Model environment: Python3.10.20, Torch2.5.1+CUDA12.1 build, NumPy1.24.3, RTX4060 Laptop GPU, `cuda:0`. Secondary environment: Python3.10.20, Torch2.12.0+cpu, NumPy2.2.6, ML4CO-Kit0.5.4. Local success is separate from server evidence.

`external/Routing-MVMoE` was clean before and after (`git status --porcelain` empty). The CVRP50 status row now has local model smoke, adapter, decoder, independent validator, small smoke, and Kit validation all `LOCAL_VERIFIED`; all four server columns remain `NOT_RUN`.

## M. Server-ready command

Use [SERVER_RUNBOOK](SERVER_RUNBOOK.md). The first committed CVRP50 implementation is `54271cf4f97d7bd99784f6e6ee5ce52ee9f5994b`; server execution must still checkout the exact project commit approved for that run.

## N. Remaining issues

- Server dataset/checkpoint identity, cp311_base environment and RTX4090 execution remain `NOT_RUN`.
- CVRP50 code is committed; server execution and server asset/environment evidence remain pending.
- Runtime values are small-smoke engineering observations, not benchmark timing claims.
- CVRP100 is now covered separately; CVRPTW remains intentionally untouched. For future CVRPTW work, the official tester already exposes a per-instance `env.depot_end` mechanism; actual validation is still required.
