# MVMoE/4E + CVRP100 integration report

Status: `SERVER_VERIFIED_INTEGRATION`. The implementation is included in clean project commit `f6db50e694cbab8870f4f1a1544856c49e9e0106` and has user-executed RTX4090 evidence.

## Identity and execution

- Project base commit: `54271cf4f97d7bd99784f6e6ee5ce52ee9f5994b`; the worktree was dirty during this uncommitted extension run and that provenance is preserved.
- Official upstream: `https://github.com/RoyalSkye/Routing-MVMoE` at `af29e5af0595f94f3ecc3bc46d72df1089a62682`, clean before and after.
- Checkpoint: `pretrained/mvmoe_4e_n100/epoch-5000.pt`, SHA256 `554d6daea825e17d62c1b9db40d56869312923848504bdf4970663673c971bdc`, strict load, epoch 5000, problem `Train_ALL`.
- Dataset: `cvrp100_hgs-20s_15.563.pkl`, SHA256 `bb47d5a113848e5a404edefc562d5d2ef6b0ade1aafc287828bdf60364e23532`, raw capacity 50.
- Model environment: Python 3.10.20, PyTorch 2.5.1 / CUDA build 12.1, NumPy 1.24.3, NVIDIA GeForce RTX 4060 Laptop GPU.
- Kit environment: Python 3.10.20, PyTorch 2.12.0+cpu, NumPy 2.2.6, ML4CO-Kit 0.5.4.

No official source/checkpoint was edited. No dependency was installed. There was no training, optimizer, backward or fine-tuning.

The full suite passed **53/53 tests** with `ML4CO_REFERENCE_TESTS=1` in the existing Kit environment. It covers all six benchmark reference sets, all existing TSP/CVRP/CVRPTW validators and audits, both MVMoE sizes, objective-completion failure paths, Git URL normalization, and rollout timing metadata.

## Smoke progression

Reduced aug1 first-2 passed the four-part completion gate for 2/2 instances; final-source total rollout time was 0.8051194s. Official search/decode configuration aug8/POMO100 first-2 also passed 2/2; final-source total rollout time was 0.9046776s. These are small engineering smokes, not the authors' full evaluation command and not paper-comparable timings.

The final aug8/POMO100 first-5 fit in one batch. Final-source total rollout time was 1.0965037s. Each row's recorded runtime is the amortized total divided by five; it is not single-instance latency.

| index | reported | independent | reported agrees | Kit | Kit agrees | reference | gap % | feasible | best aug / POMO |
|---:|---:|---:|:---:|---:|:---:|---:|---:|:---:|---:|
| 0 | 14.775226593 | 14.775226690 | true | 14.775224686 | true | 14.605233192 | 1.163922 | true / true | 0 / 87 |
| 1 | 14.362275124 | 14.362274230 | true | 14.362271309 | true | 14.043173790 | 2.272281 | true / true | 7 / 25 |
| 2 | 14.842935562 | 14.842935233 | true | 14.842935562 | true | 14.578950882 | 1.810723 | true / true | 4 / 17 |
| 3 | 15.408880234 | 15.408880468 | true | 15.408877373 | true | 15.096461296 | 2.069486 | true / true | 0 / 62 |
| 4 | 18.993721008 | 18.993722386 | true | 18.993722916 | true | 18.816236496 | 0.943259 | true / true | 2 / 4 |

Agreement uses `numpy.isclose(..., rtol=1e-6, atol=1e-6)`. All five rows are independently feasible, Kit feasible, reported-objective agreeing and Kit-objective agreeing, so all are `LOCAL_VERIFIED`.

An actual canonical solution for index 0 is:

```text
[0,88,73,96,50,32,86,59,20,6,65,0,90,49,87,22,45,7,41,75,79,14,76,85,0,52,54,5,66,34,78,89,46,91,0,30,61,23,71,74,97,98,44,70,93,47,77,0,37,53,33,69,11,9,16,92,63,2,35,0,10,38,72,1,42,25,67,99,28,83,0,64,43,31,58,13,26,80,81,24,39,48,0,40,21,82,17,94,8,4,57,3,0,18,56,36,55,95,29,27,60,19,12,84,0,62,15,100,68,51,0]
```

The full local row data, runtime timestamps, source hashes and all canonical solutions are in [the CVRP100 manifest](../manifests/mvmoe_cvrp100.json). Server first-5 reproduced the reported/independent objectives and best augmentation/POMO selections; all four gates passed. The server artifact SHA256 is `8b210b55436767ce621ad56c1dead4f1c3381e1b1f67c48155ff7397869cc04d`. Low-order Kit/reference differences from the server environment are preserved in [server evidence](../manifests/server_mvmoe_cvrp.json).
