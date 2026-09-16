# NeuOpt CVRP Paper Protocol

NeuOpt-GIRE CVRP50/100 integration is already server verified. Its historical
runner remains an engineering smoke using D2A=1 (`val_m=1`), T=1000 and an
amortized batch runtime. Those artifacts are not paper-comparable.

The new calibration protocol uses one original benchmark instance per rollout,
D2A=5 (`val_m=5`), `stall_limit=10`, `k=4`, random initialization and seed
6666. It retains the official CVRP50/100 checkpoints, dataset identities,
dummy rates, GIRE behavior, reward and action selection. T is the only search
parameter used to align runtime, and CVRP50 and CVRP100 may select different T
values.

The latest manuscript currently labels the candidate rows as D2A=5, T=1k and
D2A=5, T=5k. They remain `MANUSCRIPT_CANDIDATE` until RTX4090 batch-one
calibration is reviewed. The comparison targets are the COReformer rows:

| Problem | fewer | more | Source |
|---|---:|---:|---|
| CVRP50 | 0.197 s | 0.804 s | Latest manuscript CVRP small-scale table |
| CVRP100 | 0.307 s | 1.193 s | Latest manuscript CVRP small-scale table |

The calibration report records mean, median, minimum and maximum runtime,
objective and correctness gates for every tested T. It does not encode a fixed
percentage margin or select a final T. The final low/high settings require user
review under the requested "slightly above" criterion. Until then, all four
final T values and any manuscript protocol update remain
`PENDING_CALIBRATION`, and no NeuOpt full-set run is authorized.

Formal timing starts after model loading, dataset parsing and single-instance
adaptation. It synchronizes CUDA, times the complete official rollout, then
synchronizes CUDA again. Warm-up, successor decoding, independent validation,
ML4CO-Kit validation, provenance, JSON writing and aggregation are excluded.

Pinned NeuOpt's `kopt_Decoder.forward` applies dimensionless `squeeze()` to
the `[internal_batch, 1]` comparison that produces `stopped`. At internal batch
one, `[1, 1]` becomes scalar shape `[]`; at batch two, `[2, 1]` correctly becomes
`[2]`. The scalar makes the next iteration fail at
`k_action_left[stopped, i]`. The repository-owned runtime shim changes only
those two comparisons to `squeeze(-1)`, preserving `[1]` at batch one. It
verifies the exact pinned source shape before installation, operates in memory,
and never edits the official checkout. Any source drift fails closed. During
development, batch-two equivalence was verified with identical model state,
input and RNG state before enabling the shim in the formal runner.

Every candidate artifact stores its exact T, D2A, original batch size, dataset
subset, checkpoint/dataset hashes, project/upstream provenance, runtime
semantics, canonical successor-derived solution, official objective,
independent objective and ML4CO-Kit results. Aggregation rejects mixed T, D2A,
checkpoint or dataset identity. The calibration comparison is a separate
operation that intentionally compares individually validated T candidates over
the same fixed subset while leaving final T unset.
