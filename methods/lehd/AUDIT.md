# LEHD source audit

The official source is `CIAM-Group/NCO_code` at
`274df3c4975384592b60fe7f79fbb2441ce11c15`, subtree
`single_objective/LEHD`. A read-only `git ls-remote` check on 2026-09-20
confirmed that this commit remains the default-branch HEAD. The local checkout
was clean. The most recent commit touching the subtree is the same repository
HEAD.

## Assets and training scale

The pinned subtree tracks exactly two `.pt` model checkpoints:

- TSP `result/20230509_153705_train/checkpoint-150.pt`; `TSP/train.py` uses
  `data/train_TSP100_n100w.txt`.
- CVRP `result/20230817_235537_train/checkpoint-40.pt`; `CVRP/train.py` uses
  `data/vrp100_hgs_train_100w.txt`.

Both are size-100 trained models. No size-specific 500/1K/2K checkpoint is
claimed. SHA256 and strict-load evidence remain pending the server audit; the
manifest intentionally contains null hashes.

Official `TSP/test.py` and `CVRP/test.py` enumerate synthetic test sizes
100/200/500/1000. This establishes TSP/CVRP 500 and 1000 and CVRP200 as
author-tested generalization scales. TSP100 and CVRP100 are native training
scale. The project excludes TSP200, while CVRP50 and CVRP2000 are explicitly
senior-approved scale adaptations.

## Architecture and load path

Both official test scripts use `mode=test`, embedding dimension 128, square-root
embedding dimension `sqrt(128)`, decoder layers 6, Q/K/V dimension 16, 8 heads,
and feed-forward dimension 512. Both Tester constructors call
`model.load_state_dict(checkpoint["model_state_dict"])` without overriding
PyTorch's `strict=True` default.

## Greedy and RRC

Each `_test_one_batch` first constructs one greedy solution. It then executes
`for ... in range(RRC_budget)`: sample/invert a portion of the incumbent,
destroy it, reconstruct it with the same model, and accept only an improving
replacement. Subpath length is sampled by the official environment from the
actual current problem size. There is no SIL PRC, random insertion, kNN, POMO,
beam width, or fixed repair-length protocol in LEHD.

The author reports several RRC budgets including 50 and 500. Mapping those to
project labels `fewer` and `more` is recorded as
`project_protocol_mapping_of_author_reported_budgets`; they are not called
official fewer/more configurations.

## RNG and warm-up

The TSP test path has no author-defined test-time seed reset. Its RRC sampling
uses the continuing Torch stream. The formal integration seeds Python, NumPy,
Torch CPU, and all CUDA devices with project seed 123 and labels this
`project_reproducibility_policy`.

`CVRP/VRPTester.py::_test_one_batch` explicitly sets `random_seed = 12` and
calls `torch.manual_seed(random_seed)` for every batch. This author behavior is
retained. The NumPy seed found in `VRPEnv.drawPic_VRP` affects visualization
only and is outside inference.

Both models and environments retain mutable tensors such as encoded nodes and
selected solutions. The untimed warm-up therefore uses a dedicated Tester.
That object is discarded, CUDA cache is cleared, and all four RNG classes are
restored before constructing the formal Tester. No warm-up model/env state can
enter formal inference.

## CVRP units and solution shape

The official parser casts capacity and demands to integers. `CVRP_Encoder`
normalizes raw demand by the actual capacity, and the decoder normalizes
remaining capacity by the same value. The project adapter therefore rejects
fractional raw units and injects unnormalized benchmark demand with each task's
true capacity. It never rounds or substitutes a synthetic capacity.

The native solution is `[customer_id, route_start_flag]`, with customer IDs
1..N and flag 1 meaning a new depot route. The project decoder validates the
permutation and flags before converting it to the canonical depot-delimited
ML4CO representation.

## Hardware status

The current correctness workflow requires RTX4090. The manuscript currently
states H800. `HARDWARE_PROTOCOL_PENDING` remains true, so correctness smoke and
preflight may proceed, but even a complete fullset cannot become `PAPER_READY`
until the timing hardware protocol is resolved.
