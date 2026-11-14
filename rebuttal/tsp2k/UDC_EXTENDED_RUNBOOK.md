# Extended UDC TSP2K curve — isolated server execution

The existing GLOP/SIL fullsets keep their original detached worktrees and outputs:

- GLOP: master implementation 14a3ebd6d3d2a6c7d371487bc1895a1ebb639fe7.
- SIL: feat/lehd implementation c51d1edae0fbaadf7d8af1993da6ca3517404c9a.

This extension uses two new branches, never updates those running worktrees:

- rebuttal/tsp2k-udc-extended-gu: new UDC execution.
- rebuttal/tsp2k-udc-extended-sil: cross-method verification and plotting.

Before Step 1 export UDC_EXT_GU_SHA and UDC_EXT_SIL_SHA to the exact new
40-character commits in this patch's delivery. They are explicit fixed SHAs,
not values inferred from moving branch tips. Wait until the GPU is free before
any --execute command; existing GLOP/SIL jobs must finish first.

## 1. New detached worktrees and evidence root

```bash
set -e
conda activate cp311_base
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TSP2K_SERVER_ROOT=/inspire/hdd/global_user/majiale-253108540229/zhang
export TSP2K_MAIN_REPO="$TSP2K_SERVER_ROOT/neural-routing-baselines"
: "${UDC_EXT_GU_SHA:?Export the exact extended-GU SHA from the delivery}"
: "${UDC_EXT_SIL_SHA:?Export the exact extended-SIL SHA from the delivery}"
[[ "$UDC_EXT_GU_SHA" =~ ^[0-9a-f]{40}$ ]]
[[ "$UDC_EXT_SIL_SHA" =~ ^[0-9a-f]{40}$ ]]
export UDC_EXT_GU_ROOT="$TSP2K_SERVER_ROOT/rebuttal_worktrees/udc-extended-gu-${UDC_EXT_GU_SHA:0:8}"
export UDC_EXT_SIL_ROOT="$TSP2K_SERVER_ROOT/rebuttal_worktrees/udc-extended-sil-${UDC_EXT_SIL_SHA:0:8}"
export UDC_EXT_OUTPUT="$TSP2K_SERVER_ROOT/rebuttal_evidence/tsp2k_udc_extended_v1"
git -C "$TSP2K_MAIN_REPO" fetch https://github.com/xihe-820/neural-routing-baselines.git \
  rebuttal/tsp2k-udc-extended-gu rebuttal/tsp2k-udc-extended-sil
git -C "$TSP2K_MAIN_REPO" merge-base --is-ancestor 14a3ebd6d3d2a6c7d371487bc1895a1ebb639fe7 "$UDC_EXT_GU_SHA"
git -C "$TSP2K_MAIN_REPO" merge-base --is-ancestor c51d1edae0fbaadf7d8af1993da6ca3517404c9a "$UDC_EXT_SIL_SHA"
test ! -e "$UDC_EXT_GU_ROOT"
test ! -e "$UDC_EXT_SIL_ROOT"
test ! -e "$UDC_EXT_OUTPUT"
mkdir -p "$TSP2K_SERVER_ROOT/rebuttal_worktrees" "$UDC_EXT_OUTPUT"
git -C "$TSP2K_MAIN_REPO" worktree add --detach "$UDC_EXT_GU_ROOT" "$UDC_EXT_GU_SHA"
git -C "$TSP2K_MAIN_REPO" worktree add --detach "$UDC_EXT_SIL_ROOT" "$UDC_EXT_SIL_SHA"
git -C "$UDC_EXT_GU_ROOT" status --short
git -C "$UDC_EXT_SIL_ROOT" status --short
nvidia-smi
python -m pip check
```

No command resets, cleans or moves the original project or either running
worktree. A preexisting destination fails; choose a fresh evidence revision
instead of removing previous artifacts.

## 2. Resolve the original lock, then new read-only asset audit

```bash
cd "$UDC_EXT_SIL_ROOT"
export TSP2K_REFERENCE_LOCK=$(python -B -m rebuttal.tsp2k.manifest find-reference-lock \
  --search-root "$TSP2K_SERVER_ROOT/rebuttal_evidence")
test -f "$TSP2K_REFERENCE_LOCK"
python -B -m rebuttal.tsp2k.audit \
  --search-root "$TSP2K_SERVER_ROOT" \
  --search-root /inspire/hdd/global_user/majiale-253108540229/ML4CO-Bench-101 \
  --glop-udc-root "$UDC_EXT_GU_ROOT" --sil-root "$UDC_EXT_SIL_ROOT" \
  --udc-upstream "$TSP2K_SERVER_ROOT/baselines/NCO_code" \
  --reference-lock "$TSP2K_REFERENCE_LOCK" \
  --output "$UDC_EXT_OUTPUT/asset_lock.json"
```

Lock discovery accepts only the original three project identities. If multiple
matches exist it lists them and stops; explicitly export TSP2K_REFERENCE_LOCK
to the intended original lock from that list, then rerun the audit command.
The audit preserves all checkouts and assets. In particular it uses the explicit
baselines/NCO_code checkout and does not delete the other identical checkout.

The new lock verifies original dataset SHA/count/all 64 reference objectives,
original official checkpoints and upstream commits, authoritative UDC
S1/S2/S3/S4 evidence, both new clean project commits, and actual RTX4090/package
environment. The old lock remains a hash-bound reference, not the execution lock.
Review the resulting JSON before preflight. Inventory queries GPU identity only;
it performs no model load/forward/inference.

## 3. UDC-only preflight — all 11 points, indices [0,1]

```bash
cd "$UDC_EXT_GU_ROOT"
python -B methods/udc/rebuttal_eval.py --budget 300 --phase calibration --dump-config
python -B -m rebuttal.tsp2k.udc_extended \
  --asset-lock "$UDC_EXT_OUTPUT/asset_lock.json" \
  --output-root "$UDC_EXT_OUTPUT/runs" --stage preflight
```

Review the printed argv arrays; all commands invoke only methods/udc/rebuttal_eval.py.
Once the RTX4090 is free, the user can start:

```bash
python -B -m rebuttal.tsp2k.udc_extended \
  --asset-lock "$UDC_EXT_OUTPUT/asset_lock.json" \
  --output-root "$UDC_EXT_OUTPUT/runs" --stage preflight --execute
python -B -m rebuttal.tsp2k.udc_extended \
  --asset-lock "$UDC_EXT_OUTPUT/asset_lock.json" \
  --output-root "$UDC_EXT_OUTPUT/runs" --report-stage preflight \
  --report-output "$UDC_EXT_OUTPUT/preflight.json"
```

Each budget runs in a separate subprocess, serially. Failure stops immediately
and retains FAILED evidence. No existing output or interrupted RNG trajectory
is reused. The original six points are rerun under this same new project SHA.

## 4. UDC-only calibration — all 11 points, indices [0,1]

```bash
python -B -m rebuttal.tsp2k.udc_extended \
  --asset-lock "$UDC_EXT_OUTPUT/asset_lock.json" \
  --output-root "$UDC_EXT_OUTPUT/runs" --stage calibration --execute
python -B -m rebuttal.tsp2k.udc_extended \
  --asset-lock "$UDC_EXT_OUTPUT/asset_lock.json" \
  --output-root "$UDC_EXT_OUTPUT/runs" --report-stage calibration \
  --report-output "$UDC_EXT_OUTPUT/calibration.json"
```

Calibration is gated by the current 11-point preflight. The report rereads
actual artifacts and CPU-revalidates tours and references. Inspect actual
minimum_seconds, maximum_seconds and over_30_second_budgets. x300 approaching
30 seconds is unverified until this step succeeds. Select fullset budgets by
measured latency only. Every >30s measurement remains in the evidence.

## 5. User-reviewed UDC fullset — 64 instances at true BS1

The complete grid below is explicit. After calibration review, edit only
UDC_EXT_SELECTED to the latency-selected subset; no code selects by quality.
Do not execute this stage before that review.

```bash
export UDC_EXT_SELECTED="1 2 5 10 20 50 100 150 200 250 300"
python -B -m rebuttal.tsp2k.udc_extended \
  --asset-lock "$UDC_EXT_OUTPUT/asset_lock.json" \
  --output-root "$UDC_EXT_OUTPUT/runs" --stage fullset \
  --calibration-report "$UDC_EXT_OUTPUT/calibration.json" \
  --budgets $UDC_EXT_SELECTED
```

After reviewing the exact commands, the user may execute:

```bash
python -B -m rebuttal.tsp2k.udc_extended \
  --asset-lock "$UDC_EXT_OUTPUT/asset_lock.json" \
  --output-root "$UDC_EXT_OUTPUT/runs" --stage fullset \
  --calibration-report "$UDC_EXT_OUTPUT/calibration.json" \
  --budgets $UDC_EXT_SELECTED --execute
```

The dispatcher and direct runner both enforce the current commit/new lock,
transitively bound preflight and calibration, exact phase/count/output path.
The fullset artifact fingerprints contain experiment_stage, udc_evidence_root
and stage_evidence; direct solver CLI invocation cannot bypass these gates.

Only x_stages changes. alpha=50, sub_size=100, configured/effective POMO=2,
seed=1234, BS1, original checkpoints, CUDA allocator/memory hygiene, continuous
RNG and synchronized production timer stay fixed. No fallback reduces alpha,
population, batch, iterations or algorithm steps on OOM.

## 6. Explicitly join old GLOP/SIL and new UDC fullsets

Wait until both the original GLOP/SIL fullsets and new UDC fullsets finish.
The default original runner layout is derived from the exact reference lock.
If the user selected a different original runs root, explicitly set the two
source-root variables to those actual directories; there is no fallback search.

```bash
cd "$UDC_EXT_SIL_ROOT"
export TSP2K_LEGACY_EVIDENCE=$(dirname "$TSP2K_REFERENCE_LOCK")
export TSP2K_LEGACY_GLOP_FULLSET="$TSP2K_LEGACY_EVIDENCE/runs/fullset/GLOP"
export TSP2K_LEGACY_SIL_FULLSET="$TSP2K_LEGACY_EVIDENCE/runs/fullset/SIL"
python -B -m rebuttal.tsp2k.manifest build \
  --glop-root "$TSP2K_LEGACY_GLOP_FULLSET" \
  --sil-root "$TSP2K_LEGACY_SIL_FULLSET" \
  --udc-root "$UDC_EXT_OUTPUT/runs/fullset/UDC" \
  --udc-budgets $UDC_EXT_SELECTED --udc-commit "$UDC_EXT_GU_SHA" \
  --source-repository "$UDC_EXT_SIL_ROOT" \
  --output "$UDC_EXT_OUTPUT/artifact_manifest.json"
python -B -m rebuttal.tsp2k.aggregate \
  --manifest "$UDC_EXT_OUTPUT/artifact_manifest.json" \
  --output "$UDC_EXT_OUTPUT/curves.json" --require-fullset
MPLCONFIGDIR="$UDC_EXT_OUTPUT/matplotlib_cache" \
python -B -m rebuttal.tsp2k.plot \
  --report "$UDC_EXT_OUTPUT/curves.json" --output-prefix "$UDC_EXT_OUTPUT/curves"
```

Outputs: artifact_manifest.json, curves_verification.json, curves.json,
curves.csv, curves.png, curves.pdf and curves.svg.

The manifest pins each method/budget/path/expected commit/metadata SHA.
GLOP and SIL retain their original commits and old lock. New UDC uses its own
single new commit and new lock. Per-method source, checkpoint, environment and
lock identities must agree. Protocol, record/summary hashes, 64 ordered indices,
all references, CPU independent objectives, feasibility and Kit are verified.
Source hashes are checked against Git objects at the recorded execution commit,
not against unrelated new source bytes. GLOP RI timing is charged only at index 0;
SIL quality remains native BS64, with separate BS1 timing at [0,1,2].
Legacy read_artifact semantics and paper budgets are unchanged.

The plot re-verifies the explicit manifest and its raw sources, rejects
synthetic/calibration/handwritten values, and retains measured points above 30s.
No interpolation, estimated runtime or artificial endpoint is exported.

## Local verification boundary

The supplied 16 calibration numbers are user-reported two-instance background;
none is inserted into final artifacts or curves. Local tests are CPU/source
tests and temporary synthetic fixtures. Neither x100–x300 runtime/memory nor
the final 64-instance curves have been verified by local development.
