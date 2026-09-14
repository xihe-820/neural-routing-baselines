# MVMoE paper evaluation protocol

This protocol produces the four MVMoE/4E rows for CVRP50, CVRP100,
CVRPTW50 and CVRPTW100. It is separate from the integration evidence in
`docs/STATUS.md` and from the historical first-five runners.

## Fixed method and inference configuration

- Variant: MVMoE/4E (`model_type=MOE`), with four experts, node-level
  `input_choice` routing.
- One original instance is evaluated at a time (`original_batch_size=1`).
- Each instance keeps the official search budget: POMO size N and eight-fold
  coordinate augmentation, followed by argmax candidate selection.
- Seed is 2024. Fine tuning, training, backward passes and optimizers are all
  disabled.
- A single NVIDIA GeForce RTX 4090 is the required evaluation device.
- Each chunk loads its prepared data, official checkpoint and model once.
  The default two warm-up instances are rerun formally and excluded from time.

POMO starts and augmentation are candidates inside one solver invocation for
one original instance. They do not increase the original-instance batch size.

## Canonical CVRPTW input scaling

MVMoE CVRPTW formal evaluation uses the canonical continuous official-style
input scaling established by the isolated A/B audit. For each original ML4CO
instance it computes

```
s = max(max(original depot/customer coordinates), original depot TW end / 3)
```

and divides coordinates, time windows and service times uniformly by `s` for
model inference. Raw demand is not scaled; the adapter applies
`raw_demand/raw_capacity` exactly once. Speed remains 1.0, `loc_scaler` is
disabled, and no distance or Solomon integer-coordinate rounding is applied.
The external CVRPTWLib rounding path is unsuitable here because it changes the
continuous ML4CO geometry instead of making a uniform unit conversion.

The chosen route is validated and scored again on the original ML4CO instance.
Reference gap, independent objective and ML4CO-Kit validation therefore remain
in the original domain. The evaluator also requires the selected route's
scaled objective multiplied by `s` to agree with its original-domain objective.
Scaling is input adaptation before the formal timer. Warm-up uses scaled input,
writes no record, and each warmed instance is rerun under the formal timer.

The existing `cvrptw50` and `cvrptw100` full-set artifacts are preserved as the
verified unscaled control and engineering evidence. They are not the canonical
MVMoE CVRPTW paper result. Canonical scaled chunks use the separate
`cvrptw50_scaled` and `cvrptw100_scaled` paths and a distinct resume/protocol
identity.

## Datasets and assets

| Problem | Size | Expected instances | Dataset |
|---|---:|---:|---|
| CVRP | 50 | 10,000 | `cvrp50_hgs-1s_10.366.pkl` |
| CVRP | 100 | 10,000 | `cvrp100_hgs-20s_15.563.pkl` |
| CVRPTW | 50 | 1,000 | `cvrptw50_pyvrp-10s_16.038.pkl` |
| CVRPTW | 100 | 1,000 | `cvrptw100_pyvrp-20s_25.431.pkl` |

Dataset and official n50/n100 checkpoint SHA256 values come from the existing
MVMoE configs and manifests. The upstream is
`RoyalSkye/Routing-MVMoE@af29e5af0595f94f3ecc3bc46d72df1089a62682` and must be clean.

## Metrics and timing

For instance i, the independently evaluated route cost is `objective_i` and
the benchmark reference is `reference_i`:

```
gap_i = (objective_i - reference_i) / reference_i * 100
Obj.  = mean_i(objective_i)
Drop  = mean_i(gap_i)
Time  = mean_i(runtime_seconds_i)
```

The manuscript calls this percentage metric **Drop**. The implementation
stores `gap_percent` per instance and reports their mean; it never computes a
gap from the two mean objectives.

Time is single-original-instance wall-clock latency. After input adaptation,
the timer starts following a CUDA synchronization. It includes the official
Aug8/POMO rollout, synchronization, device-to-host result transfer and CPU
best-candidate selection, and stops after selection. It excludes dataset and
checkpoint loading, model construction, neutral-format conversion, warm-up,
provenance, independent validation, ML4CO-Kit validation, artifact writes and
aggregation.

## Evidence and completion gate

Each chunk stores provenance once in `metadata.json`, resumable inference rows
in `inference_records.jsonl`, and Kit-enriched rows in
`validated_records.jsonl`. Resume is refused if the input, chunk, source,
checkpoint, dataset, environment, warm-up policy or inference configuration
changes.

`PAPER_READY` requires exact indices `0..count-1` once each, consistent dataset
and checkpoint SHA, config, upstream, project source and hardware across all
chunks, plus independent feasibility, reported-objective agreement,
ML4CO-Kit feasibility and Kit-objective agreement for every instance. Missing,
duplicate, failed, NaN or infinite records make aggregation fail.

## Manuscript notes

The current paper uses Obj./Drop/Time columns. Its CVRPTW main and complete
tables contain an empty MVMoE row for sizes 50/100, while the CVRP tables do
not currently contain an MVMoE row. The appendix says all models are tested on
an NVIDIA H800 (80G), single GPU, batch size 1. The current approved protocol
uses the existing RTX4090 server, so the H800 statement is a manuscript
inconsistency to resolve later. This repository does not modify the paper.

| Paper row | Pipeline status |
|---|---|
| MVMoE/4E CVRP50 | `READY_FOR_SERVER_PREFLIGHT` |
| MVMoE/4E CVRP100 | `READY_FOR_SERVER_PREFLIGHT` |
| MVMoE/4E CVRPTW50 | `READY_FOR_SCALED_FORMAL_PREFLIGHT` |
| MVMoE/4E CVRPTW100 | `READY_FOR_SCALED_FORMAL_PREFLIGHT` |

These states describe the executable pipeline only. No row is `PAPER_READY`
until the complete server run passes both validation gates and strict summary.
