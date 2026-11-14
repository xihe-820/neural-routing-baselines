# TSP-2K runtime–quality rebuttal experiment

Deployment uses two independent worktrees from
`https://github.com/xihe-820/neural-routing-baselines.git`:

| Remote branch | Reviewed implementation commit | Solver implementation |
| --- | --- | --- |
| master | efcc54188586d2a52f6f053cda781b3444258833 | unchanged formal GLOP / UDC |
| feat/lehd | 5f536663e5c0b296c42fb6f9dc8e1259d7b4ec8e | existing SIL, explicit PRC constructor option |

The shared `rebuttal/tsp2k/` modules and regression tests are identical in both
worktrees. These commits are the reviewed bases preceding the warmup correction.
Use the exact corrected commit SHAs from the patch delivery as TSP2K_GU_SHA and
TSP2K_SIL_SHA in SERVER_RUNBOOK.md, then create detached worktrees. The local
rebuttal branch names are not remote deployment branches. The feat/lehd worktree
contains the aggregation/plot entrypoints used for all three methods.
No branch merge or production-registry change is required.

Reviewed manuscript: `D:\\Downloads\\ICLR-2027 RS4CO Rebuttal.md`, SHA256
`a0b97fa9b7f6c2843f6137604e04a450a39cf6c26475fbdd416dd20b123dedb6`.
Read in full locally. Reviewer Yj3H 3.1 requests runtime/quality curves; section
3.2 W1 is currently an empty response. No manuscript text was modified.

Initial source audit before publication: master=395070d, feat/lehd=cc4d017,
feat/sil=0a5036a after successful fetch. master is ancestor of feat/lehd.
The latter adds/changes 83 files, including unrelated baselines: merging it into
master would exceed this experiment. RS4CO HTTPS fetch failed authentication;
the connected GitHub commit API independently confirmed main at
8e0ed3ee1982b9357f7d6b87408a7ddf349c0594. Its old local checkout is preserved.

## Experiment

All methods use the same exact bytes of
`tsp2000_lkh_500_32.436.pkl`, 64 ordered original instances, a single visible
RTX 4090 and inference only. `audit.py` verifies dataset schema/count/references
and writes an explicit asset lock; each runner rechecks hashes before loading
models and after inference. Review that lock before GPU preflight.

- GLOP multipliers 1–6: [20m,25m,5m]; lens=[100,50,20], one shared RI order,
  identity top-level transform, greedy, local augmentation and pruning on,
  seed=1. 1x config is exactly the existing official_standard TSP2K config.
  Calls the unchanged `methods/glop/tsp/paper_eval.py::_solve_one`, model loader,
  seed/setup, shared-order builder, warmup and runtime accounting functions.
  Decoder/D2H remains inside that original timer. Order generation is charged
  once to dataset index zero; two RNG-neutral warmup instances.
- UDC x=1,2,5,10,20,50: alpha=50, sub_size=100, configured/effective POMO=2,
  seed=1234, BS1, official supplemental checkpoints with existing exact hashes.
  Calls `s3_eval.solve_tsp(x_stages=x)`; unchanged production seed, allocator,
  inter-instance memory cleanup and timing semantics. No warmup. Authoritative
  S1/S2/S3/S4 evidence and original budget-registry hash remain required.
  Only x differs from the formal TSP algorithm. Different x processes have
  independent continuous RNG streams; no common stochastic prefix is assumed.
- SIL PRC=1,2,5,10: random insertion=True, PRC=True, repair max=1000,
  POMO=1, beam=16, greedy, seed=123, setting_size=1000, TSP1K checkpoint.
  `build_tester(rebuttal_prc_budget=...)` validates this exact experiment and
  delegates to the existing official Tester constructor. Existing paper calls
  are unchanged, including the required mode/test and sqrt_embedding_dim keys.
  Fullset quality is one native original-instance BS64 solve. Independent
  BS1 timing uses indices [0,1,2], with its own freshly seeded tester and one
  RNG-neutral BS1 solve_one warmup, matching timing_probe.py. Quality executes
  directly without warmup, matching author_batch_eval.py; this also applies to
  calibration quality. Calibration uses indices [0,1] in both paths.
  Metadata records warmup separately for quality (0 batches) and timing (1 BS1 batch).
  Quality runtimes are null; no division of batch wall time is used.

Every actual tour is checked using original task coordinates, independently
recomputed objective, official-objective agreement, task.check_constraints,
task.evaluate and task.ref_sol reference. Drop is mean of per-instance percent
gaps. Timing excludes loading/adapter/validation/serialization, using each
unchanged solver's synchronization boundaries.

## Artifacts and acceptance

Output is external to source checkouts, in a new rebuttal-only tree.
Each point owns metadata.json, inference_records.jsonl and summary.json.
Identity includes full solver config, budget, dataset SHA/count/indices,
checkpoint/args SHA, clean actual project and official commits, source hashes,
environment/GPU, timing policy and lock hash. Failed runs keep FAILED metadata.
An existing output directory is always rejected: this experiment intentionally
does not resume RNG streams or reuse a partially completed native batch.
Use a new external run root to retry and retain prior failure evidence.

Statuses: MEASURED_FULLSET, MEASURED_TIMING_SUBSET, VERIFIED_HISTORICAL_REUSE,
PENDING and FAILED have distinct meanings. Calibration never becomes final
fullset or paper-ready. Records retain source=quality/timing; aggregate outputs
explicit quality_source/time_source. SIL values come from separate executions.
A complete GLOP/SIL grid is required; UDC may use an explicitly chosen subset of
the calibrated grid for fullset. All 16 initial candidates are retained for
calibration. No budget selection uses objective values.

The aggregate rechecks coverage, protocol fingerprints, numeric/semantic gates
and file hashes. Plotting rereads the underlying validated fullsets and rejects
synthetic/calibration evidence. CSV, PNG, PDF and SVG use actual measured points
only, preserve >30s observations, and introduce no extrapolated endpoints.

## Historical reuse audit

No TSP2K historical fullset GLOP or SIL artifact is present locally. No historical
point is currently declared reusable. Historical GLOP figures
33.845787 / 4.346346% / 4.952544s are reference notes, not curve input.

`reuse.py` audits explicit original source artifacts without editing them.
GLOP requires its PAPER_READY summary, all original chunk paths, exact coverage,
Kit records, prepared-data/checkpoint/args hashes, standard protocol, original
timing and shared RI provenance, and historical source files matching both
their recorded execution commit and the reviewed baseline.
SIL requires author BS64 PRC10 quality plus separate BS1 count3 timing,
their raw records, protocol/checkpoint/dataset/seed/source checks and existing
validators. Missing hashes, unknown source commits, rebindings with no matching
source snapshots, nonmatching timing samples or semantics produce NEEDS_RERUN.
Do not infer reuse from matching reported numbers.
The audit produces an eligibility report; the runner never silently substitutes
legacy results. Keep audited legacy sources and their execution commit if
using them in an independently reviewed final result import. Until that audit
is available, the prepared fullset commands explicitly remeasure every point.

See SERVER_RUNBOOK.md for server setup, inventory, preflight, calibration,
manual latency-only UDC selection, fullset and plots.
