# SIL integration plan

This integration extends the repository's existing baseline workflow. It does not
introduce a second framework.

## Existing architecture audit

The official checkouts live under the ignored `external/` tree and are pinned in
`manifests/upstreams.yaml`. They remain read-only and clean. Project-owned code
lives under `methods/<method>/`, with problem-specific adapters, decoders, and
entrypoints. NeuOpt, GLOP, and UDC all keep checkpoints, datasets, generated
inputs, logs, and artifacts outside tracked source and accept their paths through
the CLI.

SIL follows these established choices:

- **Directory and import isolation:** mirror GLOP's `external/GLOP` plus
  `methods/glop/{tsp,cvrp}` separation. `methods/sil/paper_eval.py` is one
  dedicated process; it adds only the single pinned SIL root to `sys.path` after
  all source/provenance gates. It clears generic `TSP`, `CVRP`, and `utils`
  namespaces before the official import. It never combines official repositories
  on a global `PYTHONPATH`.
- **Runner and timing:** mirror the mature NeuOpt TSP and UDC production runners:
  fail-closed project/upstream/checkpoint/dataset/GPU gates, CUDA synchronization,
  sequential ordered execution, atomic progress, exact RNG checkpoints, and
  resume only under an identical fingerprint. Formal SIL uses original-instance
  BS=1.
- **Artifacts:** mirror UDC production's `metadata.json`,
  `validated_records.jsonl`, `batch_timings.jsonl`, `summary.json`, and an
  additional `checkpoint_state.json` for exact RNG resume. A preflight becomes
  `KIT_VALIDATED`. While the manuscript H800 versus current RTX4090 hardware
  protocol remains unresolved, a complete ordered full dataset becomes
  `HARDWARE_PROTOCOL_PENDING` and cannot become `PAPER_READY`.
- **Dataset boundary:** mirror GLOP's exact ML4CO task checks. The existing pickle
  is read in place. TSP keeps all real nodes. CVRP passes depot, customer
  coordinates, raw demands, and each task's true capacity.
- **Solution schema:** TSP stores the raw official permutation and canonical
  node-0-rooted closed tour. CVRP stores the official `[customer, route_start]`
  pairs, customer order, route-start flags, decoded routes, and canonical
  `0,...,0` solution used by the shared validator and Kit.
- **Independent validation:** reuse `problems/tsp/validate.py` and
  `problems/cvrp/validate.py`; reuse `common/objective_agreement.py`. The official
  SIL objective is evidence, not ground truth. ML4CO-Kit is the secondary gate.
- **Aggregation:** match current production code: each record stores
  `(solution-reference)/reference*100`; summary Drop is the arithmetic mean of
  those per-instance values, never the gap of mean objectives.
- **Provenance:** reuse `common/provenance.py` and `common/hashing.py` for project,
  upstream, source-file, dataset, and checkpoint identity. Environment evidence
  records Python, NumPy, Torch, CUDA, device, and GPU.
- **Server handoff:** add project-owned commands here and to
  `docs/SERVER_RUNBOOK.md`, following the existing audit -> official smoke ->
  our-data smoke -> preflight -> fullset sequence.

## SIL-specific implementation

`config.py` owns the six-size senior-approved scope (TSP1K/2K/5K/10K and
CVRP1K/2K), official checkpoint registry, size adaptations, and exact formal
Greedy/PRC50/PRC500 protocols. `runtime.py` invokes pinned
`TSPTester._test_one_batch` or
`VRPTester._test_one_batch`; it does not reproduce the solver. A temporary
instance-method hook observes existing `_get_travel_distance_2` calls and retains
the final full solution argument. It forwards every call and return unchanged,
adds no solver/objective call, and clones the retained solution only after the
timed region.

The official tester's checkpoint `load_state_dict` remains strict. The project
does not patch SIL, does not add training code, and does not commit checkpoint
binaries. The runner uses one persistent official tester over ordered BS1 tasks;
resume restores Python, NumPy, Torch CPU, and all CUDA RNG states after tester
construction.

## Shared infrastructure left unchanged

The independent TSP/CVRP objectives and validators, objective tolerance,
provenance helpers, hashing helpers, ML4CO datasets, and all NeuOpt/GLOP/UDC
source and result artifacts remain unchanged. SIL only adds its upstream registry
entry, checkpoint registry, method-owned implementation/tests, and runbook
section.
