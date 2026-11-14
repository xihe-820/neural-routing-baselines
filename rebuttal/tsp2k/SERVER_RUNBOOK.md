# Server runbook — TSP2K runtime/quality curves

Commands are run manually by the user in cp311_base. Source changes must first
be reviewed, committed and pushed by the user to both rebuttal branches.
This checkout step pins their actual remote SHAs; it cannot be run before the
branches are published. It never moves the original project checkout.
The one-time use of /inspire below is a configurable deployment example; source
code contains no server-specific paths.

## 1. Create two reviewed server worktrees

```bash
conda activate cp311_base
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TSP2K_SERVER_ROOT=/inspire/hdd/global_user/majiale-253108540229/zhang
export TSP2K_MAIN_REPO="$TSP2K_SERVER_ROOT/neural-routing-baselines"
export TSP2K_GU_ROOT="$TSP2K_SERVER_ROOT/rebuttal_worktrees/tsp2k-glop-udc"
export TSP2K_SIL_ROOT="$TSP2K_SERVER_ROOT/rebuttal_worktrees/tsp2k-sil"
export TSP2K_OUTPUT="$TSP2K_SERVER_ROOT/rebuttal_evidence/tsp2k"
mkdir -p "$TSP2K_SERVER_ROOT/rebuttal_worktrees" "$TSP2K_OUTPUT"
git -C "$TSP2K_MAIN_REPO" fetch origin rebuttal/tsp2k-glop-udc rebuttal/tsp2k-sil
export TSP2K_GU_SHA=$(git -C "$TSP2K_MAIN_REPO" rev-parse origin/rebuttal/tsp2k-glop-udc)
export TSP2K_SIL_SHA=$(git -C "$TSP2K_MAIN_REPO" rev-parse origin/rebuttal/tsp2k-sil)
test ! -e "$TSP2K_GU_ROOT"
test ! -e "$TSP2K_SIL_ROOT"
git -C "$TSP2K_MAIN_REPO" worktree add --detach "$TSP2K_GU_ROOT" "$TSP2K_GU_SHA"
git -C "$TSP2K_MAIN_REPO" worktree add --detach "$TSP2K_SIL_ROOT" "$TSP2K_SIL_SHA"
git -C "$TSP2K_GU_ROOT" status --short
git -C "$TSP2K_SIL_ROOT" status --short
nvidia-smi
python -m pip check
```

Check that no other GPU workload is running before preflight or calibration.
No environment install/downgrade command is needed.

## 2. Read-only asset and prior-evidence audit

```bash
cd "$TSP2K_SIL_ROOT"
python -B -m rebuttal.tsp2k.audit \
  --search-root "$TSP2K_SERVER_ROOT" \
  --search-root /inspire/hdd/global_user/majiale-253108540229/ML4CO-Bench-101 \
  --glop-udc-root "$TSP2K_GU_ROOT" \
  --sil-root "$TSP2K_SIL_ROOT" \
  --output "$TSP2K_OUTPUT/asset_lock.json"
python -B -m rebuttal.tsp2k.reuse \
  --asset-lock "$TSP2K_OUTPUT/asset_lock.json" \
  --all-candidates --output "$TSP2K_OUTPUT/reuse_audit.json"
```

The audit searches real directories and reports missing/conflicting bytes or
multiple clean checkouts. It does not download, copy or select different
checkpoints. If source locations are ambiguous, rerun with the reported exact
paths using --glop-upstream, --udc-upstream, --sil-upstream and/or
--sil-checkpoint. A successful lock records those paths, exact official commits,
every checkpoint/args SHA, dataset/reference identities, actual package versions
and authoritative UDC evidence. Check its SIL TSP1K checkpoint SHA against the
previous released-checkpoint audit before executing GPU jobs. Neither an unknown
checkpoint hash nor historical environment versions are reported as verified
release identity. Return the raw lock/reuse audit on any unresolved mismatch.

The reuse audit is optional for execution and never edits legacy artifacts.
If it reports multiple candidates, run --help and supply one explicit
--method/--summary or --quality-dir/--timing source from the printed candidate
list. No historical reuse has been established locally. The commands below
measure every point, preserving old results.

## 3. Review the exact planned commands, then GPU preflight

```bash
cd "$TSP2K_SIL_ROOT"
python -B -m rebuttal.tsp2k.run_calibration \
  --asset-lock "$TSP2K_OUTPUT/asset_lock.json" \
  --output-root "$TSP2K_OUTPUT/runs" --stage preflight
```

Review the printed 16 argv arrays and lock. The following command executes
all 16 budget points on fixed indices [0,1], one subprocess at a time, on the
one visible RTX4090. SIL quality here is a real two-instance batch; fullset
quality remains one native BS64 batch.

```bash
python -B -m rebuttal.tsp2k.run_calibration \
  --asset-lock "$TSP2K_OUTPUT/asset_lock.json" \
  --output-root "$TSP2K_OUTPUT/runs" --stage preflight --execute
```

Each point must finish KIT_VALIDATED with actual tours and all objective/Kit
gates. OOM or any semantic/provenance failure stops the serial sweep and retains
FAILED evidence. No reduced-batch fallback or silent resume is supported.

## 4. GPU runtime calibration, after preflight passes

```bash
python -B -m rebuttal.tsp2k.run_calibration \
  --asset-lock "$TSP2K_OUTPUT/asset_lock.json" \
  --output-root "$TSP2K_OUTPUT/runs" --stage calibration --execute
python -B -m rebuttal.tsp2k.aggregate \
  --artifact-root "$TSP2K_OUTPUT/runs/calibration" \
  --output "$TSP2K_OUTPUT/calibration.json"
```

All six GLOP, six UDC and four SIL candidates must be present. Read the actual
runtime_coverage and per-point measured mean_time_seconds. GLOP/SIL points are
retained as specified. User selects any UDC subset by latency alone and confirms
whether additional UDC x values are needed in a subsequent reviewed patch.
No automatic selection by objective is performed. >30s values are retained.

## 5. Fullset, only after user review of calibration

The command below uses the complete initial UDC grid explicitly. After reviewing
calibration, change only --udc-budgets to the selected subset of that measured
grid. The program requires a valid actual calibration report from the same lock.

```bash
python -B -m rebuttal.tsp2k.run_calibration \
  --asset-lock "$TSP2K_OUTPUT/asset_lock.json" \
  --output-root "$TSP2K_OUTPUT/runs" --stage fullset \
  --calibration-report "$TSP2K_OUTPUT/calibration.json" \
  --udc-budgets 1 2 5 10 20 50
```

Only the user starts fullset, after reviewing the printed commands:

```bash
python -B -m rebuttal.tsp2k.run_calibration \
  --asset-lock "$TSP2K_OUTPUT/asset_lock.json" \
  --output-root "$TSP2K_OUTPUT/runs" --stage fullset \
  --calibration-report "$TSP2K_OUTPUT/calibration.json" \
  --udc-budgets 1 2 5 10 20 50 --execute
```

GLOP/UDC each process indices 0..63 at BS1 with continuous seeded RNG.
SIL processes all 64 in one true author batch for quality and separately
reinitializes its tester for BS1 timing at indices [0,1,2] for every PRC.
All points retain actual runtime; batch-quality wall time is never a timing
column. No GPU command in this runbook has been executed by local development.

## 6. Verify artifacts and create final curves

```bash
python -B -m rebuttal.tsp2k.aggregate \
  --artifact-root "$TSP2K_OUTPUT/runs/fullset" \
  --output "$TSP2K_OUTPUT/curves.json" --require-fullset
MPLCONFIGDIR="$TSP2K_OUTPUT/matplotlib_cache" \
python -B -m rebuttal.tsp2k.plot \
  --report "$TSP2K_OUTPUT/curves.json" \
  --output-prefix "$TSP2K_OUTPUT/curves"
```

Outputs: curves.json, curves.csv, curves.png, curves.pdf and curves.svg.
The plot verifies the source measurements again, excludes calibration and
synthetic fixtures, and uses only measured points without a fabricated zero or
30-second endpoint. Methods with a missing budget/instance or mismatched
dataset/checkpoint identity fail closed.

Output directories are exclusive and intentionally non-resumable. To retry
after a failed stage, use a fresh TSP2K_OUTPUT and rerun its preflight/calibration;
retain the previous tree for diagnosis. Source repositories, historical
paper-ready artifacts, checkpoints and datasets are never overwritten.
