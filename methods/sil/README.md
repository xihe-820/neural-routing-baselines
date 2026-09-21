# SIL server handoff

Local development stops before any official smoke, ML4CO GPU smoke, preflight,
or fullset. No local output is paper-ready.

## Phase 0: checkout, source, checkpoints, and datasets

```bash
conda activate cp311_base
export SERVER_ROOT=/inspire/hdd/global_user/majiale-253108540229/zhang
export BASELINE_PROJECT_ROOT="$SERVER_ROOT/neural-routing-baselines"
cd "$BASELINE_PROJECT_ROOT"
pwd
git rev-parse --show-toplevel
git status --short
test -z "$(git status --porcelain)"
git fetch origin feat/sil
if git show-ref --verify --quiet refs/heads/feat/sil; then
  git switch feat/sil
else
  git switch --track -c feat/sil origin/feat/sil
fi
git pull --ff-only origin feat/sil
export SIL_PROJECT_COMMIT=<approved-review-fix-commit>
test "$(git rev-parse HEAD)" = "$SIL_PROJECT_COMMIT"
test -z "$(git status --porcelain)"

export SIL_UPSTREAM="$BASELINE_PROJECT_ROOT/external/SIL"
test -d "$SIL_UPSTREAM/.git" || git clone https://github.com/CIAM-Group/SIL.git "$SIL_UPSTREAM"
git -C "$SIL_UPSTREAM" fetch origin
git -C "$SIL_UPSTREAM" checkout --detach 9ec783e90a1631f7b95f84eb20f8f9751cb45c10
test "$(git -C "$SIL_UPSTREAM" rev-parse HEAD)" = 9ec783e90a1631f7b95f84eb20f8f9751cb45c10
test -z "$(git -C "$SIL_UPSTREAM" status --porcelain)"

export SIL_CHECKPOINT_ROOT="$SERVER_ROOT/checkpoints/SIL"
mkdir -p "$SIL_CHECKPOINT_ROOT"
gdown 1nT2_JkghMjasHpZY2i_dhrGnhvbOlxTV -O "$SIL_CHECKPOINT_ROOT/checkpoint-tsp1k.pt"
gdown 1v2WHGKi8T-lTqE7NJsHRBno8AyKUY0n3 -O "$SIL_CHECKPOINT_ROOT/checkpoint-tsp5k.pt"
gdown 1UTKPLAOqw2Lco-Zx2HZ64qYn2O3GrrQ5 -O "$SIL_CHECKPOINT_ROOT/checkpoint-tsp10k.pt"
gdown 1yNUPciWXF-gSnsnRxXpMjO5rqAt0OOHj -O "$SIL_CHECKPOINT_ROOT/checkpoint-cvrp1k.pt"
sha256sum "$SIL_CHECKPOINT_ROOT"/*.pt | tee "$SIL_CHECKPOINT_ROOT/SHA256SUMS"
python -B -c 'import glob,torch; [(print(p,sorted(torch.load(p,map_location="cpu",weights_only=False))) ) for p in sorted(glob.glob("'"$SIL_CHECKPOINT_ROOT"'/*.pt"))]'
```

Export the four observed checkpoint hashes. Do not edit the tracked registry on
the server. Set the exact existing ML4CO pickle paths and their independently
audited hashes; the runner has no fallback path.

```bash
export SIL_TSP1K_SHA=<observed-tsp1k-checkpoint-sha256>
export SIL_TSP5K_SHA=<observed-tsp5k-checkpoint-sha256>
export SIL_TSP10K_SHA=<observed-tsp10k-checkpoint-sha256>
export SIL_CVRP1K_SHA=<observed-cvrp1k-checkpoint-sha256>
export SIL_ARTIFACT_ROOT="$SERVER_ROOT/artifacts/neural-routing-baselines/sil/$(git rev-parse --short HEAD)"

export SIL_TSP1000_DATASET=<exact-ML4CO-TSP1000-pickle>
export SIL_TSP2000_DATASET=<exact-ML4CO-TSP2000-pickle>
export SIL_TSP5000_DATASET=<exact-ML4CO-TSP5000-pickle>
export SIL_TSP10000_DATASET=<exact-ML4CO-TSP10000-pickle>
export SIL_CVRP1000_DATASET=<exact-ML4CO-CVRP1000-pickle>
export SIL_CVRP2000_DATASET=<exact-ML4CO-CVRP2000-pickle>
export SIL_TSP1000_DATASET_SHA=<audited-sha256>
export SIL_TSP2000_DATASET_SHA=<audited-sha256>
export SIL_TSP5000_DATASET_SHA=<audited-sha256>
export SIL_TSP10000_DATASET_SHA=<audited-sha256>
export SIL_CVRP1000_DATASET_SHA=<audited-sha256>
export SIL_CVRP2000_DATASET_SHA=<audited-sha256>
mkdir -p "$SIL_ARTIFACT_ROOT"
sha256sum "$SIL_TSP1000_DATASET" "$SIL_TSP2000_DATASET" "$SIL_TSP5000_DATASET" "$SIL_TSP10000_DATASET" "$SIL_CVRP1000_DATASET" "$SIL_CVRP2000_DATASET"
test "$(basename "$SIL_TSP1000_DATASET")" = tsp1000_concorde_23.118.pkl
test "$(basename "$SIL_TSP2000_DATASET")" = tsp2000_lkh_500_32.436.pkl
test "$(basename "$SIL_TSP5000_DATASET")" = tsp5000_lkh_500_50.968.pkl
test "$(basename "$SIL_TSP10000_DATASET")" = tsp10000_lkh_500_71.782.pkl
test "$(basename "$SIL_CVRP1000_DATASET")" = cvrp1000_hgs-360s_41.171.pkl
test "$(basename "$SIL_CVRP2000_DATASET")" = cvrp2000_hgs-360s_57.181.pkl
```

## Senior-approved baseline reproduction: result batch and BS1 timing probe

```
SIL_RESULT_PROTOCOL = OFFICIAL_STYLE_BATCHED
SIL_TIMING_PROTOCOL = BS1_SMALL_SAMPLE
```

Quality uses one real pinned-Tester batch at the frozen author-style size; its
wall time is diagnostic only and cannot be used as BS1 `Time`. The separate
timing probe invokes only original-instance BS=1 calls and validates every
timed solution after the timed region. Both require an RTX 4090.

| Problem | Dataset count | Author result batch |
|---|---:|---:|
| TSP1000 / TSP2000 / TSP5000 / TSP10000 | 128 / 64 / 32 / 16 | 128 / 64 / 16 / 16 |
| CVRP1000 / CVRP2000 | 100 / 100 | 100 / 100 |

`TSP2000` and `CVRP2000` remain `senior_approved_adaptation` and use
`adapted_from_author_1k_batch_cap`. Do not lower a batch automatically after
OOM. An explicit `--batch-size` override requires `--batch-override-reason`
and is recorded as an override.

```bash
run_sil_author_result () {
  problem="$1"; size="$2"; budget="$3"; dataset="$4"; dataset_sha="$5"; checkpoint="$6"; checkpoint_sha="$7"
  python -B methods/sil/author_batch_eval.py \
    --problem "$problem" --problem-size "$size" --budget "$budget" \
    --dataset "$dataset" --expected-dataset-sha256 "$dataset_sha" \
    --upstream "$SIL_UPSTREAM" --checkpoint "$checkpoint" \
    --expected-checkpoint-sha256 "$checkpoint_sha" --device cuda:0 \
    --output-dir "$SIL_ARTIFACT_ROOT/author_batch/${problem}${size}/${budget}"
}

run_sil_bs1_timing () {
  problem="$1"; size="$2"; budget="$3"; dataset="$4"; dataset_sha="$5"; checkpoint="$6"; checkpoint_sha="$7"
  python -B methods/sil/timing_probe.py \
    --problem "$problem" --problem-size "$size" --budget "$budget" \
    --dataset "$dataset" --expected-dataset-sha256 "$dataset_sha" \
    --upstream "$SIL_UPSTREAM" --checkpoint "$checkpoint" \
    --expected-checkpoint-sha256 "$checkpoint_sha" --device cuda:0 \
    --output "$SIL_ARTIFACT_ROOT/bs1_timing/${problem}${size}/${budget}.json"
}

# Example shortest pair. Repeat the result command for every formal size/protocol.
run_sil_author_result tsp 1000 greedy "$SIL_TSP1000_DATASET" "$SIL_TSP1000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-tsp1k.pt" "$SIL_TSP1K_SHA"
run_sil_bs1_timing tsp 1000 greedy "$SIL_TSP1000_DATASET" "$SIL_TSP1000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-tsp1k.pt" "$SIL_TSP1K_SHA"
```

Default timing samples are Greedy=5, PRC50=3, and PRC500=1. The old BS1
fullset path below remains a strict diagnostic/legacy audit; the
senior-approved baseline reproduction protocol supersedes its former
full-dataset-BS1 quality requirement.

## Phase 1: official native smoke

```bash
export SIL_OFFICIAL_TSP1K_DATASET=<SIL-release-MCTS_tsp1000_test_concorde.txt>
export SIL_OFFICIAL_CVRP1K_DATASET=<SIL-release-test_cvrp1000_hgs_n128_C250.txt>
export SIL_OFFICIAL_TSP1K_DATASET_SHA=<audited-sha256>
export SIL_OFFICIAL_CVRP1K_DATASET_SHA=<audited-sha256>
python -B methods/sil/official_smoke.py --problem tsp --problem-size 1000 --upstream "$SIL_UPSTREAM" --checkpoint "$SIL_CHECKPOINT_ROOT/checkpoint-tsp1k.pt" --expected-checkpoint-sha256 "$SIL_TSP1K_SHA" --official-dataset "$SIL_OFFICIAL_TSP1K_DATASET" --expected-dataset-sha256 "$SIL_OFFICIAL_TSP1K_DATASET_SHA" --count 2 --output "$SIL_ARTIFACT_ROOT/official_smoke/tsp1k.json"
python -B methods/sil/official_smoke.py --problem cvrp --problem-size 1000 --upstream "$SIL_UPSTREAM" --checkpoint "$SIL_CHECKPOINT_ROOT/checkpoint-cvrp1k.pt" --expected-checkpoint-sha256 "$SIL_CVRP1K_SHA" --official-dataset "$SIL_OFFICIAL_CVRP1K_DATASET" --expected-dataset-sha256 "$SIL_OFFICIAL_CVRP1K_DATASET_SHA" --count 2 --output "$SIL_ARTIFACT_ROOT/official_smoke/cvrp1k.json"
```

Require `status=PASS`. These runs establish only cp311_base/Torch/CUDA/RTX4090,
checkpoint strict-load, and official native-data compatibility.

## Phase 2: our-data formal-protocol smoke

Run one official-inference smoke for `greedy` at every formal size, followed by
one `fewer=PRC50` smoke at every size. All runs use actual solution capture,
independent validation, and ML4CO-Kit validation.

```bash
run_sil_smoke () {
  problem="$1"; size="$2"; protocol="$3"; dataset="$4"; dataset_sha="$5"; checkpoint="$6"; checkpoint_sha="$7"; count="$8"
  python -B "methods/sil/$problem/paper_eval.py" --problem-size "$size" --budget "$protocol" --dataset "$dataset" --expected-dataset-sha256 "$dataset_sha" --upstream "$SIL_UPSTREAM" --checkpoint "$checkpoint" --expected-checkpoint-sha256 "$checkpoint_sha" --scope our-smoke --offset 0 --count "$count" --output-dir "$SIL_ARTIFACT_ROOT/our_smoke/${problem}${size}/$protocol"
}

run_sil_smoke tsp 1000 greedy "$SIL_TSP1000_DATASET" "$SIL_TSP1000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-tsp1k.pt" "$SIL_TSP1K_SHA" 1
run_sil_smoke tsp 2000 greedy "$SIL_TSP2000_DATASET" "$SIL_TSP2000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-tsp1k.pt" "$SIL_TSP1K_SHA" 1
run_sil_smoke tsp 5000 greedy "$SIL_TSP5000_DATASET" "$SIL_TSP5000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-tsp5k.pt" "$SIL_TSP5K_SHA" 1
run_sil_smoke tsp 10000 greedy "$SIL_TSP10000_DATASET" "$SIL_TSP10000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-tsp10k.pt" "$SIL_TSP10K_SHA" 1
run_sil_smoke cvrp 1000 greedy "$SIL_CVRP1000_DATASET" "$SIL_CVRP1000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-cvrp1k.pt" "$SIL_CVRP1K_SHA" 1
run_sil_smoke cvrp 2000 greedy "$SIL_CVRP2000_DATASET" "$SIL_CVRP2000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-cvrp1k.pt" "$SIL_CVRP1K_SHA" 1

run_sil_smoke tsp 1000 fewer "$SIL_TSP1000_DATASET" "$SIL_TSP1000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-tsp1k.pt" "$SIL_TSP1K_SHA" 2
run_sil_smoke tsp 2000 fewer "$SIL_TSP2000_DATASET" "$SIL_TSP2000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-tsp1k.pt" "$SIL_TSP1K_SHA" 2
run_sil_smoke tsp 5000 fewer "$SIL_TSP5000_DATASET" "$SIL_TSP5000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-tsp5k.pt" "$SIL_TSP5K_SHA" 1
run_sil_smoke tsp 10000 fewer "$SIL_TSP10000_DATASET" "$SIL_TSP10000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-tsp10k.pt" "$SIL_TSP10K_SHA" 1
run_sil_smoke cvrp 1000 fewer "$SIL_CVRP1000_DATASET" "$SIL_CVRP1000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-cvrp1k.pt" "$SIL_CVRP1K_SHA" 2
run_sil_smoke cvrp 2000 fewer "$SIL_CVRP2000_DATASET" "$SIL_CVRP2000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-cvrp1k.pt" "$SIL_CVRP1K_SHA" 2
```

Require all twelve summaries to be `KIT_VALIDATED`. Greedy uses official pure
greedy (`budget=0`, no random insertion, no kNN). Fewer uses the frozen PRC50
pipeline. CVRP must preserve raw demands, true per-instance capacity, and exactly
one model-side normalization.

## Phase 3: same-size more preflight

Run `more=PRC500` on every formal size. The evidence is intentionally
size-specific because checkpoints, memory use, and runtime differ by size.

```bash
run_sil_more_preflight () {
  problem="$1"; size="$2"; dataset="$3"; dataset_sha="$4"; checkpoint="$5"; checkpoint_sha="$6"; count="$7"
  python -B "methods/sil/$problem/paper_eval.py" --problem-size "$size" --budget more --dataset "$dataset" --expected-dataset-sha256 "$dataset_sha" --upstream "$SIL_UPSTREAM" --checkpoint "$checkpoint" --expected-checkpoint-sha256 "$checkpoint_sha" --scope preflight --offset 0 --count "$count" --output-dir "$SIL_ARTIFACT_ROOT/preflight/${problem}${size}/more"
}
run_sil_more_preflight tsp 1000 "$SIL_TSP1000_DATASET" "$SIL_TSP1000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-tsp1k.pt" "$SIL_TSP1K_SHA" 2
run_sil_more_preflight tsp 2000 "$SIL_TSP2000_DATASET" "$SIL_TSP2000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-tsp1k.pt" "$SIL_TSP1K_SHA" 2
run_sil_more_preflight tsp 5000 "$SIL_TSP5000_DATASET" "$SIL_TSP5000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-tsp5k.pt" "$SIL_TSP5K_SHA" 1
run_sil_more_preflight tsp 10000 "$SIL_TSP10000_DATASET" "$SIL_TSP10000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-tsp10k.pt" "$SIL_TSP10K_SHA" 1
run_sil_more_preflight cvrp 1000 "$SIL_CVRP1000_DATASET" "$SIL_CVRP1000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-cvrp1k.pt" "$SIL_CVRP1K_SHA" 2
run_sil_more_preflight cvrp 2000 "$SIL_CVRP2000_DATASET" "$SIL_CVRP2000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-cvrp1k.pt" "$SIL_CVRP1K_SHA" 2
```

Require all six summaries to be `KIT_VALIDATED` with official, independent, and
Kit objective agreement.

## Legacy Phase 4: 18-cell BS1 fullset audit

Each cell requires evidence from the same project commit, source provenance,
problem, size, protocol, dataset, checkpoint, and upstream. Greedy and fewer use
Phase 2 evidence; more uses Phase 3 evidence.

```bash
dataset_count () {
  problem="$1"; dataset="$2"
  python -B -c 'import ml4co_kit as k,sys; w=(k.TSPWrapper() if sys.argv[1]=="tsp" else k.CVRPWrapper()); w.from_pickle(sys.argv[2]); print(len(w.task_list))' "$problem" "$dataset"
}
export SIL_TSP1000_COUNT="$(dataset_count tsp "$SIL_TSP1000_DATASET")"
export SIL_TSP2000_COUNT="$(dataset_count tsp "$SIL_TSP2000_DATASET")"
export SIL_TSP5000_COUNT="$(dataset_count tsp "$SIL_TSP5000_DATASET")"
export SIL_TSP10000_COUNT="$(dataset_count tsp "$SIL_TSP10000_DATASET")"
export SIL_CVRP1000_COUNT="$(dataset_count cvrp "$SIL_CVRP1000_DATASET")"
export SIL_CVRP2000_COUNT="$(dataset_count cvrp "$SIL_CVRP2000_DATASET")"

run_sil_fullset () {
  problem="$1"; size="$2"; protocol="$3"; dataset="$4"; dataset_sha="$5"; checkpoint="$6"; checkpoint_sha="$7"; count="$8"
  if test "$protocol" = more; then
    evidence="$SIL_ARTIFACT_ROOT/preflight/${problem}${size}/more/summary.json"
  else
    evidence="$SIL_ARTIFACT_ROOT/our_smoke/${problem}${size}/$protocol/summary.json"
  fi
  python -B "methods/sil/$problem/paper_eval.py" --problem-size "$size" --budget "$protocol" --dataset "$dataset" --expected-dataset-sha256 "$dataset_sha" --upstream "$SIL_UPSTREAM" --checkpoint "$checkpoint" --expected-checkpoint-sha256 "$checkpoint_sha" --preflight-evidence "$evidence" --scope fullset --offset 0 --count "$count" --output-dir "$SIL_ARTIFACT_ROOT/fullset/${problem}${size}/$protocol" --resume
}
for protocol in greedy fewer more; do
  run_sil_fullset tsp 1000 "$protocol" "$SIL_TSP1000_DATASET" "$SIL_TSP1000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-tsp1k.pt" "$SIL_TSP1K_SHA" "$SIL_TSP1000_COUNT"
  run_sil_fullset tsp 2000 "$protocol" "$SIL_TSP2000_DATASET" "$SIL_TSP2000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-tsp1k.pt" "$SIL_TSP1K_SHA" "$SIL_TSP2000_COUNT"
  run_sil_fullset tsp 5000 "$protocol" "$SIL_TSP5000_DATASET" "$SIL_TSP5000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-tsp5k.pt" "$SIL_TSP5K_SHA" "$SIL_TSP5000_COUNT"
  run_sil_fullset tsp 10000 "$protocol" "$SIL_TSP10000_DATASET" "$SIL_TSP10000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-tsp10k.pt" "$SIL_TSP10K_SHA" "$SIL_TSP10000_COUNT"
  run_sil_fullset cvrp 1000 "$protocol" "$SIL_CVRP1000_DATASET" "$SIL_CVRP1000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-cvrp1k.pt" "$SIL_CVRP1K_SHA" "$SIL_CVRP1000_COUNT"
  run_sil_fullset cvrp 2000 "$protocol" "$SIL_CVRP2000_DATASET" "$SIL_CVRP2000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-cvrp1k.pt" "$SIL_CVRP1K_SHA" "$SIL_CVRP2000_COUNT"
done
```

The legacy fullset runs above retain their historical `PAPER_READY` behavior.
They are strict BS1 diagnostic evidence, not the senior-approved baseline
reproduction quality path. Author-batch artifacts provide `Obj`/`Gap`; the
separate BS1 timing probes provide `Time`.
