# NeuOpt CVRP Paper Protocol

## Latest frozen protocol

The final NeuOpt-GIRE paper matrix is CVRP50/100, D2A (`val_m`) 1, T 20/50,
and original batch size 1/100. All eight configurations use
`stall_limit=10`, `k=4`, random initialization, seed 6666, evaluation only,
one RTX 4090, and the pinned official checkpoints and datasets in
`methods/neuopt/cvrp/config.py`.

BS1 supplies the Complete Results rows. BS100 supplies the Parallel rows. The
pinned decoder has a BS1 shape bug because dimensionless `squeeze()` turns its
`[1,1]` stopped comparison into a scalar. The repository therefore installs
the guarded, in-memory `squeeze()` to `squeeze(-1)` shim only for BS1. It
requires the exact pinned function text and never changes the official source
on disk. BS100 must use the exact unmodified pinned decoder and refuses an
already installed shim.

The 2026-09-16 manuscript PDF has four blank NeuOpt rows in Table 8, named
`NeuOpt (fewer/more) (BS = 1/100)`, and two blank NeuOpt rows in Table 10,
named `NeuOpt (fewer/more)`. Those labels do not yet state D2A or T. Per the
latest senior-author protocol, fewer means D2A=1/T=20 and more means
D2A=1/T=50; Table 8 BS100 and Table 10 BS1 are the intended final rows. The
repository does not edit the manuscript.

## Inference, timing, and validation

BS100 is a real vectorized batch: one native tensor batch containing 100
original instances enters one official `agent.rollout`. It is never emulated
by 100 BS1 calls. Every returned successor is independently decoded and every
instance must pass official-objective correspondence, independent feasibility
and objective validation, and ML4CO-Kit validation. One failure rejects the
entire batch artifact.

For each batch, the evaluator saves the RNG state, synchronizes CUDA, and times
one `record=False` official rollout. It synchronizes CUDA again, restores the
saved state, and performs an untimed `record=True` replay to recover actual
successors. Timed and replay official objectives and their final RNG states
must match exactly. Model/checkpoint loading, input parsing, warm-up, resume
replay, evidence replay, decoding, validation, and artifact I/O are outside
paper time.

Per-instance gap and full-set statistics are:

```text
gap_i = (independent_objective_i - reference_i) / reference_i * 100
Obj   = mean(independent_objective_i)
Drop  = mean(gap_i)
Total = sum(batch_runtime_seconds)
```

Parallel Table 8 `Time` is mean batch latency and is not divided by BS100. Two
manuscript cross-checks are `1.326 min * 60 / 100 = 0.7956 s` for CVRP50
COReformer-fewer BS100 and `8.386 min * 60 / 100 = 5.0316 s` for CVRP100
COReformer-more BS100, matching its displayed 0.795s and 5.031s. At BS1, mean
batch latency equals mean instance latency and supplies Complete Results time.

## Artifacts, aggregation, and fallback

Each resumable chunk contains `metadata.json`, `validated_records.jsonl`,
`batch_timings.jsonl`, and `summary.json`. A recommended BS100 full set uses
ten 1,000-instance chunks, each with ten native batches. Aggregation requires
exact expected coverage with no missing or duplicate index and identical
problem, size, D2A, T, BS, dataset/checkpoint hashes, project/upstream state,
GPU, timing semantics, shim mode, RNG policy, and source provenance.

If a BS1 full set is too slow, the supported hybrid candidate uses BS100
full-set Obj/Drop and BS1 fixed-subset Time. It explicitly records separate
`quality_source` and `time_source`, `hybrid=true`, `estimated=true`, the exact
sample indices and timing distribution, and `bs1_fullset_completed=false`.
It also compares BS1 and BS100 quality on the exact timing-subset indices. No
automatic discrepancy threshold is applied; a human reviewer may set
`HYBRID_QUALITY_WARNING` after examining the reported deltas.

## Historical protocol

The earlier D2A=5/T=1k/5k runtime calibration is
`LEGACY_PROTOCOL_EXPLORATION`, superseded by the latest senior-author
protocol. Its reader and artifacts remain available as development evidence.
They cannot pass the D2A1 production identity or aggregator gates and must not
be used for final paper rows.
