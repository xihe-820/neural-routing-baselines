# Neural Routing Baselines

This repository provides independent, reproducible integrations of official neural-routing implementations and pretrained checkpoints with ML4CO-Bench. Each completed integration runs the official model, captures its actual solution, validates feasibility and objective value independently, and uses ML4CO-Kit as a secondary check.

## Formal scope

The original cross-method matrix targets sizes 50 and 100. Explicit
reviewer-scale extensions are documented per method; SIL targets the large-scale
rows shown below. Training, fine-tuning and modified checkpoints are outside
scope.

| Method | TSP | CVRP | CVRPTW |
|---|---|---|---|
| GLOP | 50 / 100 | 50 / 100 | — |
| UDC | 50 / 100 | 50 / 100 | — |
| CaDA | — | 50 / 100 | 50 / 100 |
| MVMoE/4E | — | 50 / 100 | 50 / 100 |
| RF-TE | — | 50 / 100 | 50 / 100 |
| MoSES(CaDA) | — | 50 / 100 | 50 / 100 |
| NeuOpt | — | 50 / 100 | — |
| SIL | 1K / 2K / 5K / 10K | 1K / 2K | — |
| LEHD | 100 / 500 / 1K | 50 / 100 / 200 / 500 / 1K / 2K | — |

The original 50/100 matrix contains 13 method/problem integrations and 26
independent size rows. See [STATUS](docs/STATUS.md) for current evidence and
[SIL server handoff](methods/sil/README.md) for the separate reviewer-scale
extension.

MVMoE CVRP/CVRPTW 50/100 and NeuOpt CVRP50/100 are server verified. GLOP TSP50/100 is locally verified and ready for user-executed server validation; GLOP CVRP50/100 remains blocked because the official small-size partitioner configuration is undefined.

SIL TSP1K/2K/5K/10K and CVRP1K/2K have a local integration. Formal SIL
evaluation is run on a single NVIDIA RTX 4090 with original-instance BS=1.

LEHD uses the two official size-100-trained checkpoints for the frozen
large-scale generalization matrix. Formal LEHD evaluation is run on a single
NVIDIA RTX 4090 with original-instance BS=1.

## Architecture

- `external/`: ignored, read-only official checkouts pinned by `manifests/upstreams.yaml`.
- `methods/<method>/<problem>/`: problem-specific input preparation, native adapter, exact solution decoder, runner and optional ML4CO-Kit validation.
- `problems/<problem>/`: independent objective and constraint validation using original benchmark data.
- `common/`: result schema, provenance, hashing and the shared objective-agreement policy (`rtol=1e-6`, `atol=1e-6`).
- `scripts/`: read-only environment, dataset, upstream and checkpoint audits.
- `manifests/`: source of truth for identities, evidence, environments, historical provenance and detailed run results.
- `artifacts/`: ignored generated inputs and run outputs.

Official source, datasets, checkpoints, logs and full result artifacts are never committed here. Paths are supplied through CLI arguments or environment variables; source code does not depend on a server-specific absolute path.

## Evidence levels

- `SOURCE_CONFIRMED`: official documentation or source identifies an intended asset/configuration.
- `LOCAL_VERIFIED`: the recorded check ran in the local WSL environment.
- `SERVER_VERIFIED`: the user ran the recorded check on the target server and returned its evidence.
- `NOT_RUN`, `NOT_IMPLEMENTED`, `BLOCKED`, `NOT_APPLICABLE`: no stronger claim is made.

An integration is complete at an execution location only when its actual decoded solutions pass independent feasibility, ML4CO-Kit feasibility, reported-objective agreement and Kit-objective agreement. Checkpoint deserialization alone is not integration evidence. Runtime values are engineering diagnostics and are not paper-comparable.

## Basic workflow

Run the dependency-light unit suite:

```bash
python -B -m unittest discover -s tests -v
```

Enable the available ML4CO reference checks:

```bash
ML4CO_REFERENCE_TESTS=1 python -B -m unittest discover -s tests -v
```

Audit configured assets without modifying official repositories or environments:

```bash
python3 -B scripts/audit_assets.py --upstream-root external
```

Implemented method commands and the required return artifacts are in [SERVER_RUNBOOK](docs/SERVER_RUNBOOK.md). Detailed asset identities, run results and historical dirty-worktree facts remain in `manifests/`.

Development uses local WSL plus user-executed server validation. No training or fine-tuning is performed, and official source remains read-only.
