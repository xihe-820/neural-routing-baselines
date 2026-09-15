# GLOP official/paper protocol audit

Audit date: 2026-09-15. Project base: `5068e95d9dec7aa36c46cb8d2de7568396c2abcf`.
Official source: `https://github.com/henry-yeh/GLOP` at
`e540bc0153a0598e923e35116deeaecaf9c1cfff`, clean when inspected. This
document records protocol evidence and blockers. It does not authorize formal
inference.

## Paper row mapping audit

The GLOP paper names two TSP budgets: plain **GLOP** and **GLOP (more
revisions)**. Its Table 9 specifies both at TSP100/500/1K/10K. This supports the
main-table hypothesis `fewer = plain GLOP` and `more = GLOP (more revisions)`.
The current manuscript never states that mapping for external baselines.

The manuscript is internally inconsistent. `main/tsp_main.tex` has two GLOP
rows at all six sizes. `appendix/tsp_full.tex` has a single plain GLOP row for
100/500/1K, but a single `GLOP (fewer)` row for 2K/5K/10K. The CVRP main table
has fewer/more rows at 1K/2K, while the appendix has one plain GLOP row at
500/1K/2K. The official CVRP protocol has one neural GLOP-G budget and no
fewer/more nomenclature. Therefore:

`MAIN/APPENDIX FEWER-MORE MAPPING = UNRESOLVED_MANUSCRIPT_MAPPING`.

The likely TSP mapping above is a hypothesis, not the frozen manuscript
protocol. An author must select the appendix budget and decide whether CVRP
should have one official row or two newly defined adaptations.

## Source review and execution semantics

Reviewed official paths include `README.md`, `main.py`, `eval_cvrp.py`,
`eval_cvrplib_neural.py`, `problems/tsp/problem_tsp.py`, `problems/cvrp.py`,
`heatmap/cvrp/{infer,inst,sampler,train}.py`, `nets/partition_net.py`,
`utils/{functions,insertion}.py`, and the reviser model/loader path.

- TSP input is a pickle list converted to float32 tensors. The output of
  `reconnect` is the improved coordinate sequence and its closed Euclidean
  length.
- Each requested TSP width slot gets one `torch.randperm(N)` insertion order.
  `random-insertion` 0.3.0.post1 receives the points and that order. After all
  revisions, the first minimum-cost candidate is retained.
- For TSP sizes above 100, width is exactly the number of independent RI
  candidates. With pruning enabled, all but the best candidate are removed
  after the first reviser. `--no_prune` retains candidates through every
  reviser and selects the best at the end.
- Every local revision min-max normalizes each SHPP. Unless `--no_aug` is set,
  it evaluates identity, x reflection, y reflection and xy reflection; each is
  decoded in both directions, so the local choice is best of eight.
- The special `N<=100` branch divides CLI width by four using integer division
  and then adds four top-level coordinate reflections. Thus CLI `--width 140`
  becomes 35 RI orders and 140 top-level candidates. Table 9's standard
  TSP100 `W=35` would become 8 orders and 32 candidates if passed literally.
  This code/paper ambiguity, plus the cross-distribution-only dataset context,
  blocks a formal Uniform-100 choice.
- Reviser list order is executable order. A schedule `100 50 20` loads and
  runs Reviser-100, then Reviser-50, then Reviser-20. No schedule entry is
  ignored.
- CVRP input is `(depot, customer coordinates, raw demand, capacity)`. The
  partitioner uses `demand/capacity`, depot-relative polar coordinates, and
  sparse edge features. `Sampler` starts/ends routes at depot 0, masks visited
  customers and capacity violations, and uses greedy global decoding when
  `n_partition=1`.
- CVRP neural evaluation differs from `eval_cvrp.py`: `main.py` uses official
  revisers as neural sub-TSP solvers; `eval_cvrp.py` calls LKH-3. The paper's
  plain GLOP-G is the neural variant, while `GLOP-G (LKH-3)` is explicitly a
  separate hybrid. The manuscript describes GLOP as global plus autoregressive
  local policy, so the paper baseline must use the neural path.
- Official CVRP `main.py` forces width, `n_partition`, and original-instance
  batch size to one. Its README omits `--decode_strategy`, so the neural local
  revisers use the parser default `sampling`; GLOP-G describes greedy global
  partition decoding.

## TSP protocol evidence matrix

All rows use Random Insertion and original batch size 1. `Aug` and `Prune`
refer to local SHPP augmentation and first-reviser candidate pruning. The
checkpoint set follows `revision_lens` exactly.

| Size | Manuscript locations/labels | Official source | Plain GLOP `(lens; iters; W)` | More revisions `(lens; iters; W)` | Decode | Aug / Prune | Config class | Paper status / blocker |
|---:|---|---|---|---|---|---|---|---|
| 100 | main fewer+more; appendix plain GLOP | paper Table 9; README only gives cross-distribution more command | `100,50,20,10; 20,10,10,5; 35` | same; `W=140` | sampling | paper says augmentation off; more README also `no_prune`; plain pruning unspecified | E | `NEED_AUTHOR_DECISION`: not a Uniform-100 paper config; W=35 conflicts with the small-size branch |
| 500 | main fewer+more; appendix plain GLOP | paper Tables 1/9; README more command | `100,50,20; 20,25,5; 1` | same; `W=10` | greedy | on / on | B after BS1 | `NEED_AUTHOR_DECISION`: appendix budget unspecified |
| 1K | main fewer+more; appendix plain GLOP | paper Tables 1/9; README more command | `100,50,20; 20,25,5; 1` | same; `W=10` | greedy | on / on | B after BS1 | `NEED_AUTHOR_DECISION`: appendix budget unspecified |
| 2K | main fewer+more; appendix fewer | none | — | — | — | — | D | `NEED_AUTHOR_DECISION`: no official config; interpolation forbidden |
| 5K | main fewer+more; appendix fewer | none | — | — | — | — | D | `NEED_AUTHOR_DECISION`: no official config; interpolation forbidden |
| 10K | main fewer+more; appendix fewer | paper Tables 1/9; README more command | `100,50,20; 10,20,5; 1` | `100,50,20; 50,25,5; 1` | greedy | on / on | B after BS1 | `NEED_AUTHOR_DECISION`: manuscript does not state main/appendix relationship |

Classes are: A exact official, B official config with original BS changed to 1,
C legitimate size adaptation, D no official mapping, E author decision needed.
Changing `eval_batch_size` to one changes the DataLoader slice only. It retains
RI width, augmentation, revisions, pruning, and per-instance selection. The
same precomputed RI orders are sliced for each instance. It is therefore a
valid paper batching adaptation; width remains internal search parallelism.

## CVRP protocol evidence matrix

| Size | Manuscript locations/labels | Official source | Partitioner | Local revisers / iters | Global / local decode | Aug / Prune | W / partitions / original BS | Config class | Paper status / blocker |
|---:|---|---|---|---|---|---|---|---|---|
| 500 | appendix plain GLOP | none | none | none | none | none | none | D | `BLOCKED`: `K_SPARSE[500]` and `cvrp-500.pt` do not exist; using 1K is unsupported |
| 1K | main fewer+more; appendix plain GLOP | paper Tables 6/10; README neural command | `cvrp-1000.pt`, k=100 | `20 / 5` | greedy / sampling | on / on | `1 / 1 / 1` | A | `NEED_AUTHOR_DECISION`: official protocol has one budget, manuscript requests two |
| 2K | main fewer+more; appendix plain GLOP | paper Tables 6/10; README neural command | `cvrp-2000.pt`, k=200 | `50,20 / 5,5` | greedy / sampling | on / on | `1 / 1 / 1` | A | `NEED_AUTHOR_DECISION`: official protocol has one budget, manuscript requests two |

The partitioner network is not size-agnostic in the released execution path:
both model selection and graph sparsity index `problem_size`; inference indexes
the actual customer count again. CVRP500 therefore cannot follow CVRP1K merely
by supplying its checkpoint.

## Checkpoint inventory

CPU deserialization and exact strict state loading were performed without GPU
inference. All four revisers are
`nets.attention_local.AttentionModel`, contain 116 state entries and 1,301,888
parameters, and load with no missing or unexpected keys. Their `args.json`
identifies problem `local` and graph size matching the reviser size.

| Asset | Bytes | SHA256 | Strict model identity |
|---|---:|---|---|
| `Reviser-stage2/reviser_10/epoch-299.pt` | 24,394,377 | `41bd9e05d5f623a6a7978be0063354d75f6f8df62e0ed368789ca449f41922f4` | Reviser-10, strict |
| `Reviser-stage2/reviser_20/epoch-299.pt` | 25,034,377 | `6771bf6b955fe26004f378c1ab0a2068c3048d717325a62b85a279e0ec22a865` | Reviser-20, strict |
| `Reviser-stage2/reviser_50/epoch-299.pt` | 27,594,377 | `25189e74e1e0323ced3016e9c7495c2c8d8ae082961e8696dffff79f5db1d4a6` | Reviser-50, strict |
| `Reviser-stage2/reviser_100/epoch-299.pt` | 31,434,377 | `3810b460f210de35b5d4bd1f680f505ff4619823880652be1c7f35f320584451` | Reviser-100, strict |
| `Partitioner/cvrp/cvrp-1000.pt` | 1,951,194 | `3dd5ad73da831a69f65674cf30ebe910f2b74fcfeb0d57e42d0a0efb3139663b` | `nets.partition_net.Net`, k=100, depth=12, 148,513 parameters, strict |
| `Partitioner/cvrp/cvrp-2000.pt` | 1,951,194 | `988458678a06297edb767bb1c710074e3ad2d0a14f7f5b85934530f14c8ab89b` | `nets.partition_net.Net`, k=200, depth=12, 148,513 parameters, strict |
| `Partitioner/cvrp/cvrp-2000-cvrplib.pt` | 988,666 | `e174465fce2203f57255efcaa095ce056391b68545b9a50dc89456563f8fad53` | CVRPLIB-only k=300/depth=6 asset; outside manuscript synthetic rows |

The required Reviser-10/20/50/100 `args.json` files have respectively 1,004,
1,004, 1,004 and 1,022 bytes, with SHA256
`e21195ed71321b91ca2517e49b4a7556c27239ed1017d603e34d25a49742879c`,
`66171fc5178ee7fc2b8ddcda8a7c90e804a0e9c9eb960375f82df40cbc228ef6`,
`ae311d53fe1e36573a609cc7bab75be1f346a577576c36a1d30ff799cbdcc76c`,
and `b99400a52c2dd4b6bdbd221f17432ad65d0e9d585207032ee65126b78c0354d9`.
These identities remain pinned in `methods/glop/tsp/config.py`.

The partitioner strict-load audit used the exact
official `Net` topology; local PyG was unavailable, so a construction-only
BatchNorm stand-in reproduced PyG's registered state names. Forward execution
remains pending server verification.

## Formal timing boundary

Formal Time is mean wall-clock solve latency per original instance. After
models, dataset record, and neutral adapter output are ready, synchronize CUDA,
start the timer, and include RI initialization, top-level augmentation,
partition heatmap and greedy route construction for CVRP, SHPP preparation,
all revisions, pruning/candidate selection, final synchronization, D2H of the
selected coordinate solution, and exact node-ID decoding. Stop after the
canonical solution is available.

Exclude dataset/checkpoint loading, model construction, warm-up, artifact I/O,
independent validation, Kit validation, and summary aggregation. RI and CVRP
partition preparation belong to the solver and must not be hidden as dataset
adaptation. Official `main.py` times dataset construction/loading too, so its
timer cannot be reused for the paper measurement without moving the boundary.

## Solution, objective, and validation gates

TSP formal output must retain the coordinate tour from `reconnect`, map every
float32 coordinate bit pattern through an explicit identity/reflection
bijection, rotate node 0 to the front, and close the permutation. CVRP formal
output must retain the selected partition's sub-tour tensor before flattening,
map each non-depot coordinate exactly to its original customer ID, and use
depot padding as explicit route separators. Duplicate/missing customers,
ambiguous coordinate identities, nearest-neighbor matching, repair, or a
cost-only record fail the run.

Independent validators then require a closed continuous-Euclidean TSP tour, or
depot-return CVRP routes visiting each customer once within raw capacity. They
recompute objective on the original ML4CO coordinates. ML4CO-Kit independently
checks the same canonical solution and objective as a second gate.

For each instance `i`, store independent `objective_i` and benchmark
`reference_i`, then compute:

```
Drop = mean_i((objective_i - reference_i) / reference_i * 100)
Obj  = mean_i(objective_i)
Time = mean_i(runtime_i)
```

The manuscript references are Concorde for TSP100/500/1K, LKH(500) for
TSP2K/5K/10K, and HGS for all CVRP sizes.

## Dataset audit and remaining evidence

The local checkout contains only the manuscript-scope TSP100 file; its prior
audit established 1,280 TSP tasks and SHA256
`a2bfe99857b8072bdba051f6ae402b7e241f01b0462c5f379ed0aa03786406a0`.
The other eight manuscript-scope files are absent locally. This is not evidence
about the server. `scripts/audit_glop_paper_datasets.py` discovers and verifies
all nine sizes and records absolute path, realpath, bytes, SHA256, count, exact
task class/shape, coordinate/demand/capacity ranges, distance semantics, and an
independent index-0 reference objective plus Kit agreement. Its server output
is `PENDING_SERVER_VERIFICATION` until returned by the user.

Official sanity anchors are separate from ML4CO results. The original paper
reports plain/more TSP objectives of 17.07/16.91 at 500, 24.01/23.84 at 1K,
and 75.62/75.29 at 10K. It uses 128, 128, and 16 instances and reports total
runtime. For neural CVRP GLOP-G it reports 47.1, 63.5, 141.9, 191.7 at
1K/2K/5K/7K with per-instance times 0.4/1.2/1.7/2.4 seconds on RTX3090-class
paper hardware. These values are sanity anchors only.

## Hardware and next gate

The manuscript claims H800 80GB evaluation, while the approved reproduction
server is RTX4090. `MANUSCRIPT_HARDWARE_MISMATCH = UNRESOLVED`. Objective and
Drop can be reproducibly evaluated on the approved server; RTX4090 Time cannot
be represented as H800 Time. Before timing enters the manuscript, an author
must choose H800 reruns or revise the hardware statement.

No formal preflight is ready. Author decisions are required for the
main/appendix row mapping, TSP100, TSP2K/5K, CVRP fewer/more, CVRP500, and the
hardware statement. After those decisions, implement separate TSP/CVRP paper
adapters, exact decoders, per-instance timers, independent/Kit gates, and only
then prepare an RTX4090 preflight command.
