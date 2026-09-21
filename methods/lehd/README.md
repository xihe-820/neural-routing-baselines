# LEHD server handoff

Run these commands manually on the server after the local patch is reviewed and
committed. They do not modify official source or the conda environment.

## Phase 0: checkout and identity

```bash
conda activate cp311_base
export SERVER_ROOT=/inspire/hdd/global_user/majiale-253108540229/zhang
export BASELINE_PROJECT_ROOT="$SERVER_ROOT/neural-routing-baselines"
cd "$BASELINE_PROJECT_ROOT"
git fetch origin feat/lehd
if git show-ref --verify --quiet refs/heads/feat/lehd; then
  git switch feat/lehd
else
  git switch --track -c feat/lehd origin/feat/lehd
fi
git pull --ff-only origin feat/lehd
export LEHD_PROJECT_COMMIT=<approved-lehd-integration-commit>
test "$(git rev-parse HEAD)" = "$LEHD_PROJECT_COMMIT"
test -z "$(git status --porcelain)"

export LEHD_UPSTREAM="$BASELINE_PROJECT_ROOT/external/NCO_code"
test -d "$LEHD_UPSTREAM/.git" || git clone https://github.com/CIAM-Group/NCO_code.git "$LEHD_UPSTREAM"
git -C "$LEHD_UPSTREAM" fetch origin
git -C "$LEHD_UPSTREAM" checkout --detach 274df3c4975384592b60fe7f79fbb2441ce11c15
test "$(git -C "$LEHD_UPSTREAM" rev-parse HEAD)" = 274df3c4975384592b60fe7f79fbb2441ce11c15
test -z "$(git -C "$LEHD_UPSTREAM" status --porcelain)"

export LEHD_TSP_CHECKPOINT="$LEHD_UPSTREAM/single_objective/LEHD/TSP/result/20230509_153705_train/checkpoint-150.pt"
export LEHD_CVRP_CHECKPOINT="$LEHD_UPSTREAM/single_objective/LEHD/CVRP/result/20230817_235537_train/checkpoint-40.pt"
sha256sum "$LEHD_TSP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT"
export LEHD_TSP_CHECKPOINT_SHA=<observed-checkpoint-150-sha256>
export LEHD_CVRP_CHECKPOINT_SHA=<observed-checkpoint-40-sha256>
test "$(sha256sum "$LEHD_TSP_CHECKPOINT" | cut -d' ' -f1)" = "$LEHD_TSP_CHECKPOINT_SHA"
test "$(sha256sum "$LEHD_CVRP_CHECKPOINT" | cut -d' ' -f1)" = "$LEHD_CVRP_CHECKPOINT_SHA"
python -B -c 'import torch,sys; p=torch.load(sys.argv[1],map_location="cpu",weights_only=False); assert "model_state_dict" in p; print(sorted(p))' "$LEHD_TSP_CHECKPOINT"
python -B -c 'import torch,sys; p=torch.load(sys.argv[1],map_location="cpu",weights_only=False); assert "model_state_dict" in p; print(sorted(p))' "$LEHD_CVRP_CHECKPOINT"

export LEHD_ARTIFACT_ROOT="$SERVER_ROOT/artifacts/neural-routing-baselines/lehd/$(git rev-parse --short HEAD)"
mkdir -p "$LEHD_ARTIFACT_ROOT"
```

Set the exact nine ML4CO dataset files and independently audited SHA256 values:

```bash
export LEHD_TSP100_DATASET=<path>/tsp100_concorde_7.756.pkl
export LEHD_TSP500_DATASET=<path>/tsp500_concorde_16.546.pkl
export LEHD_TSP1000_DATASET=<path>/tsp1000_concorde_23.118.pkl
export LEHD_CVRP50_DATASET=<path>/cvrp50_hgs-1s_10.366.pkl
export LEHD_CVRP100_DATASET=<path>/cvrp100_hgs-20s_15.563.pkl
export LEHD_CVRP200_DATASET=<path>/cvrp200_hgs-60s_19.630.pkl
export LEHD_CVRP500_DATASET=<path>/cvrp500_hgs-300s_37.154.pkl
export LEHD_CVRP1000_DATASET=<path>/cvrp1000_hgs-360s_41.171.pkl
export LEHD_CVRP2000_DATASET=<path>/cvrp2000_hgs-360s_57.181.pkl
export LEHD_TSP100_DATASET_SHA=<audited-sha256>
export LEHD_TSP500_DATASET_SHA=<audited-sha256>
export LEHD_TSP1000_DATASET_SHA=<audited-sha256>
export LEHD_CVRP50_DATASET_SHA=<audited-sha256>
export LEHD_CVRP100_DATASET_SHA=<audited-sha256>
export LEHD_CVRP200_DATASET_SHA=<audited-sha256>
export LEHD_CVRP500_DATASET_SHA=<audited-sha256>
export LEHD_CVRP1000_DATASET_SHA=<audited-sha256>
export LEHD_CVRP2000_DATASET_SHA=<audited-sha256>
sha256sum "$LEHD_TSP100_DATASET" "$LEHD_TSP500_DATASET" "$LEHD_TSP1000_DATASET" "$LEHD_CVRP50_DATASET" "$LEHD_CVRP100_DATASET" "$LEHD_CVRP200_DATASET" "$LEHD_CVRP500_DATASET" "$LEHD_CVRP1000_DATASET" "$LEHD_CVRP2000_DATASET"
```

## Senior-approved baseline reproduction: result batch and BS1 timing probe

```
LEHD_RESULT_PROTOCOL = OFFICIAL_STYLE_BATCHED
LEHD_TIMING_PROTOCOL = BS1_SMALL_SAMPLE
```

Current project mapping: Greedy=RRC0, fewer=RRC20, more=RRC50. RRC20 is a
senior-approved project adaptation. RRC50 is an official setting reclassified
from the former project label `fewer` to `more`. Existing RRC500 artifacts are
legacy extra-budget evidence and are not current formal results.

Quality uses one real pinned-Tester batch at the frozen author-style size; its
wall time is diagnostic only and cannot be used as BS1 `Time`. The separate
timing probe invokes only original-instance BS=1 calls and validates every
timed solution after the timed region. Both require an RTX 4090.

| Problem | Dataset count | Author result batch |
|---|---:|---:|
| TSP100 / TSP500 / TSP1000 | 1280 / 128 / 128 | 1280 / 128 / 128 |
| CVRP50 / CVRP100 | 10000 / 10000 | 10000 / 10000 |
| CVRP200 / CVRP500 / CVRP1000 / CVRP2000 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |

`CVRP50` and `CVRP2000` are recorded as senior-approved project adaptations;
their batch origins are respectively the nearest author small-size whole-set
and large-scale-cap conventions. Do not lower a batch automatically after OOM.
An explicit `--batch-size` override requires `--batch-override-reason` and is
recorded as an override.

```bash
run_lehd_author_result () {
  problem="$1"; size="$2"; protocol="$3"; dataset="$4"; dataset_sha="$5"; checkpoint="$6"; checkpoint_sha="$7"
  python -B methods/lehd/author_batch_eval.py \
    --problem "$problem" --problem-size "$size" --protocol "$protocol" \
    --dataset "$dataset" --expected-dataset-sha256 "$dataset_sha" \
    --upstream "$LEHD_UPSTREAM" --checkpoint "$checkpoint" \
    --expected-checkpoint-sha256 "$checkpoint_sha" --device cuda:0 \
    --output-dir "$LEHD_ARTIFACT_ROOT/author_batch/${problem}${size}/${protocol}"
}

run_lehd_bs1_timing () {
  problem="$1"; size="$2"; protocol="$3"; dataset="$4"; dataset_sha="$5"; checkpoint="$6"; checkpoint_sha="$7"
  python -B methods/lehd/timing_probe.py \
    --problem "$problem" --problem-size "$size" --protocol "$protocol" \
    --dataset "$dataset" --expected-dataset-sha256 "$dataset_sha" \
    --upstream "$LEHD_UPSTREAM" --checkpoint "$checkpoint" \
    --expected-checkpoint-sha256 "$checkpoint_sha" --device cuda:0 \
    --output "$LEHD_ARTIFACT_ROOT/bs1_timing/${problem}${size}/${protocol}.json"
}

# Example shortest pair. Repeat the result command for every formal size/protocol.
run_lehd_author_result tsp 100 greedy "$LEHD_TSP100_DATASET" "$LEHD_TSP100_DATASET_SHA" "$LEHD_TSP_CHECKPOINT" "$LEHD_TSP_CHECKPOINT_SHA"
run_lehd_bs1_timing tsp 100 greedy "$LEHD_TSP100_DATASET" "$LEHD_TSP100_DATASET_SHA" "$LEHD_TSP_CHECKPOINT" "$LEHD_TSP_CHECKPOINT_SHA"
```

Default timing samples are Greedy=5, RRC20=3, and RRC50=3. The old BS1
fullset path below remains a strict diagnostic/legacy audit; the
senior-approved baseline reproduction protocol supersedes its former
full-dataset-BS1 quality requirement.

## Verified legacy RRC50 rebind

The old root is read-only. This command validates every RRC50 record, aggregate,
asset hash, batch identity, source commit, and timing sample before writing
current `more=RRC50` artifacts. It records the old solver commit separately and
states that no solver execution occurred during rebind.

```bash
export LEHD_LEGACY_ROOT="$SERVER_ROOT/artifacts/neural-routing-baselines/lehd/6630301"
python -B methods/lehd/rebind_rrc50.py \
  --legacy-root "$LEHD_LEGACY_ROOT" \
  --output-root "$LEHD_ARTIFACT_ROOT"
```

## Phase 1: official native TSP1K/CVRP1K smoke

```bash
export LEHD_NATIVE_TSP1K="$LEHD_UPSTREAM/single_objective/LEHD/TSP/data/test_TSP1000_n128.txt"
export LEHD_NATIVE_CVRP1K="$LEHD_UPSTREAM/single_objective/LEHD/CVRP/data/vrp1000_test_lkh.txt"
export LEHD_NATIVE_TSP1K_SHA="$(sha256sum "$LEHD_NATIVE_TSP1K" | cut -d' ' -f1)"
export LEHD_NATIVE_CVRP1K_SHA="$(sha256sum "$LEHD_NATIVE_CVRP1K" | cut -d' ' -f1)"
python -B methods/lehd/official_smoke.py --problem tsp --problem-size 1000 --upstream "$LEHD_UPSTREAM" --checkpoint "$LEHD_TSP_CHECKPOINT" --expected-checkpoint-sha256 "$LEHD_TSP_CHECKPOINT_SHA" --official-dataset "$LEHD_NATIVE_TSP1K" --expected-dataset-sha256 "$LEHD_NATIVE_TSP1K_SHA" --count 2 --output "$LEHD_ARTIFACT_ROOT/official_smoke/tsp1k.json"
python -B methods/lehd/official_smoke.py --problem cvrp --problem-size 1000 --upstream "$LEHD_UPSTREAM" --checkpoint "$LEHD_CVRP_CHECKPOINT" --expected-checkpoint-sha256 "$LEHD_CVRP_CHECKPOINT_SHA" --official-dataset "$LEHD_NATIVE_CVRP1K" --expected-dataset-sha256 "$LEHD_NATIVE_CVRP1K_SHA" --count 2 --output "$LEHD_ARTIFACT_ROOT/official_smoke/cvrp1k.json"
```

Both artifacts must report `PASS`. This proves native parsing, strict checkpoint
load, CUDA, and official inference compatibility only.

## Phase 2: nine-size Greedy and RRC20 smoke

```bash
run_lehd_smoke () {
  problem="$1"; size="$2"; protocol="$3"; dataset="$4"; dataset_sha="$5"; checkpoint="$6"; checkpoint_sha="$7"; count="$8"
  python -B "methods/lehd/$problem/paper_eval.py" --problem-size "$size" --protocol "$protocol" --dataset "$dataset" --expected-dataset-sha256 "$dataset_sha" --upstream "$LEHD_UPSTREAM" --checkpoint "$checkpoint" --expected-checkpoint-sha256 "$checkpoint_sha" --scope our-smoke --offset 0 --count "$count" --output-dir "$LEHD_ARTIFACT_ROOT/our_smoke/${problem}${size}/$protocol"
}
run_lehd_smoke tsp 100 greedy "$LEHD_TSP100_DATASET" "$LEHD_TSP100_DATASET_SHA" "$LEHD_TSP_CHECKPOINT" "$LEHD_TSP_CHECKPOINT_SHA" 1
run_lehd_smoke tsp 500 greedy "$LEHD_TSP500_DATASET" "$LEHD_TSP500_DATASET_SHA" "$LEHD_TSP_CHECKPOINT" "$LEHD_TSP_CHECKPOINT_SHA" 1
run_lehd_smoke tsp 1000 greedy "$LEHD_TSP1000_DATASET" "$LEHD_TSP1000_DATASET_SHA" "$LEHD_TSP_CHECKPOINT" "$LEHD_TSP_CHECKPOINT_SHA" 1
run_lehd_smoke cvrp 50 greedy "$LEHD_CVRP50_DATASET" "$LEHD_CVRP50_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA" 1
run_lehd_smoke cvrp 100 greedy "$LEHD_CVRP100_DATASET" "$LEHD_CVRP100_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA" 1
run_lehd_smoke cvrp 200 greedy "$LEHD_CVRP200_DATASET" "$LEHD_CVRP200_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA" 1
run_lehd_smoke cvrp 500 greedy "$LEHD_CVRP500_DATASET" "$LEHD_CVRP500_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA" 1
run_lehd_smoke cvrp 1000 greedy "$LEHD_CVRP1000_DATASET" "$LEHD_CVRP1000_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA" 1
run_lehd_smoke cvrp 2000 greedy "$LEHD_CVRP2000_DATASET" "$LEHD_CVRP2000_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA" 1
run_lehd_smoke tsp 100 fewer "$LEHD_TSP100_DATASET" "$LEHD_TSP100_DATASET_SHA" "$LEHD_TSP_CHECKPOINT" "$LEHD_TSP_CHECKPOINT_SHA" 2
run_lehd_smoke tsp 500 fewer "$LEHD_TSP500_DATASET" "$LEHD_TSP500_DATASET_SHA" "$LEHD_TSP_CHECKPOINT" "$LEHD_TSP_CHECKPOINT_SHA" 2
run_lehd_smoke tsp 1000 fewer "$LEHD_TSP1000_DATASET" "$LEHD_TSP1000_DATASET_SHA" "$LEHD_TSP_CHECKPOINT" "$LEHD_TSP_CHECKPOINT_SHA" 2
run_lehd_smoke cvrp 50 fewer "$LEHD_CVRP50_DATASET" "$LEHD_CVRP50_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA" 2
run_lehd_smoke cvrp 100 fewer "$LEHD_CVRP100_DATASET" "$LEHD_CVRP100_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA" 2
run_lehd_smoke cvrp 200 fewer "$LEHD_CVRP200_DATASET" "$LEHD_CVRP200_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA" 2
run_lehd_smoke cvrp 500 fewer "$LEHD_CVRP500_DATASET" "$LEHD_CVRP500_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA" 2
run_lehd_smoke cvrp 1000 fewer "$LEHD_CVRP1000_DATASET" "$LEHD_CVRP1000_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA" 2
run_lehd_smoke cvrp 2000 fewer "$LEHD_CVRP2000_DATASET" "$LEHD_CVRP2000_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA" 1
```

Require all 18 summaries to be `KIT_VALIDATED`.

## Phase 3: nine same-size RRC50 preflights

```bash
run_lehd_more () {
  problem="$1"; size="$2"; dataset="$3"; dataset_sha="$4"; checkpoint="$5"; checkpoint_sha="$6"
  python -B "methods/lehd/$problem/paper_eval.py" --problem-size "$size" --protocol more --dataset "$dataset" --expected-dataset-sha256 "$dataset_sha" --upstream "$LEHD_UPSTREAM" --checkpoint "$checkpoint" --expected-checkpoint-sha256 "$checkpoint_sha" --scope preflight --offset 0 --count 1 --output-dir "$LEHD_ARTIFACT_ROOT/preflight/${problem}${size}/more"
}
run_lehd_more tsp 100 "$LEHD_TSP100_DATASET" "$LEHD_TSP100_DATASET_SHA" "$LEHD_TSP_CHECKPOINT" "$LEHD_TSP_CHECKPOINT_SHA"
run_lehd_more tsp 500 "$LEHD_TSP500_DATASET" "$LEHD_TSP500_DATASET_SHA" "$LEHD_TSP_CHECKPOINT" "$LEHD_TSP_CHECKPOINT_SHA"
run_lehd_more tsp 1000 "$LEHD_TSP1000_DATASET" "$LEHD_TSP1000_DATASET_SHA" "$LEHD_TSP_CHECKPOINT" "$LEHD_TSP_CHECKPOINT_SHA"
run_lehd_more cvrp 50 "$LEHD_CVRP50_DATASET" "$LEHD_CVRP50_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA"
run_lehd_more cvrp 100 "$LEHD_CVRP100_DATASET" "$LEHD_CVRP100_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA"
run_lehd_more cvrp 200 "$LEHD_CVRP200_DATASET" "$LEHD_CVRP200_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA"
run_lehd_more cvrp 500 "$LEHD_CVRP500_DATASET" "$LEHD_CVRP500_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA"
run_lehd_more cvrp 1000 "$LEHD_CVRP1000_DATASET" "$LEHD_CVRP1000_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA"
run_lehd_more cvrp 2000 "$LEHD_CVRP2000_DATASET" "$LEHD_CVRP2000_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA"
```

Require all nine summaries to be `KIT_VALIDATED`. These are current
`more=RRC50`, not the historical RRC500 protocol.

## Appendix parallel evaluation

Measured rows use complete datasets, one real official `solve_batch` call per
batch, CUDA-synchronized solver-only time, and validation outside timing. The
formal measured matrix is TSP100/500/1000 BS128 and CVRP50/100/200 BS100 for
both RRC20 and RRC50. The artifact stores raw values; the current LaTeX table
renders Obj, Drop, Total, and Time to three decimal places, with Total expressed
in human-readable seconds/minutes/hours.

```bash
run_lehd_parallel () {
  problem="$1"; size="$2"; protocol="$3"; batch="$4"; dataset="$5"; dataset_sha="$6"; checkpoint="$7"; checkpoint_sha="$8"
  python -B methods/lehd/parallel_eval.py \
    --problem "$problem" --problem-size "$size" --protocol "$protocol" --batch-size "$batch" \
    --dataset "$dataset" --expected-dataset-sha256 "$dataset_sha" \
    --upstream "$LEHD_UPSTREAM" --checkpoint "$checkpoint" \
    --expected-checkpoint-sha256 "$checkpoint_sha" --device cuda:0 \
    --output "$LEHD_ARTIFACT_ROOT/parallel/${problem}${size}/${protocol}/bs${batch}"
}

run_lehd_parallel tsp 100 fewer 128 "$LEHD_TSP100_DATASET" "$LEHD_TSP100_DATASET_SHA" "$LEHD_TSP_CHECKPOINT" "$LEHD_TSP_CHECKPOINT_SHA"
run_lehd_parallel tsp 500 fewer 128 "$LEHD_TSP500_DATASET" "$LEHD_TSP500_DATASET_SHA" "$LEHD_TSP_CHECKPOINT" "$LEHD_TSP_CHECKPOINT_SHA"
run_lehd_parallel tsp 1000 fewer 128 "$LEHD_TSP1000_DATASET" "$LEHD_TSP1000_DATASET_SHA" "$LEHD_TSP_CHECKPOINT" "$LEHD_TSP_CHECKPOINT_SHA"
run_lehd_parallel tsp 100 more 128 "$LEHD_TSP100_DATASET" "$LEHD_TSP100_DATASET_SHA" "$LEHD_TSP_CHECKPOINT" "$LEHD_TSP_CHECKPOINT_SHA"
run_lehd_parallel tsp 500 more 128 "$LEHD_TSP500_DATASET" "$LEHD_TSP500_DATASET_SHA" "$LEHD_TSP_CHECKPOINT" "$LEHD_TSP_CHECKPOINT_SHA"
run_lehd_parallel tsp 1000 more 128 "$LEHD_TSP1000_DATASET" "$LEHD_TSP1000_DATASET_SHA" "$LEHD_TSP_CHECKPOINT" "$LEHD_TSP_CHECKPOINT_SHA"
run_lehd_parallel cvrp 50 fewer 100 "$LEHD_CVRP50_DATASET" "$LEHD_CVRP50_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA"
run_lehd_parallel cvrp 100 fewer 100 "$LEHD_CVRP100_DATASET" "$LEHD_CVRP100_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA"
run_lehd_parallel cvrp 200 fewer 100 "$LEHD_CVRP200_DATASET" "$LEHD_CVRP200_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA"
run_lehd_parallel cvrp 50 more 100 "$LEHD_CVRP50_DATASET" "$LEHD_CVRP50_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA"
run_lehd_parallel cvrp 100 more 100 "$LEHD_CVRP100_DATASET" "$LEHD_CVRP100_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA"
run_lehd_parallel cvrp 200 more 100 "$LEHD_CVRP200_DATASET" "$LEHD_CVRP200_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA"
```

BS1 rows are generated only with the explicit derived flag. `Obj`/`Drop` come
from the full quality artifact, `Time` from all three BS1 probe samples, and
`Total=Time*dataset_count`; the result never claims a measured serial sweep.

```bash
run_lehd_derived_bs1 () {
  problem="$1"; size="$2"; protocol="$3"
  python -B methods/lehd/parallel_eval.py \
    --problem "$problem" --problem-size "$size" --protocol "$protocol" --batch-size 1 \
    --derive-bs1-from-verified-artifacts \
    --quality-source "$LEHD_ARTIFACT_ROOT/author_batch/${problem}${size}/${protocol}" \
    --timing-source "$LEHD_ARTIFACT_ROOT/bs1_timing/${problem}${size}/${protocol}.json" \
    --output "$LEHD_ARTIFACT_ROOT/parallel/${problem}${size}/${protocol}/bs1"
}

for protocol in fewer more; do
  run_lehd_derived_bs1 tsp 100 "$protocol"
  run_lehd_derived_bs1 tsp 500 "$protocol"
  run_lehd_derived_bs1 tsp 1000 "$protocol"
  run_lehd_derived_bs1 cvrp 50 "$protocol"
  run_lehd_derived_bs1 cvrp 100 "$protocol"
  run_lehd_derived_bs1 cvrp 200 "$protocol"
done
```

## Legacy Phase 4: 27-cell BS1 fullset audit

```bash
dataset_count () {
  problem="$1"; dataset="$2"
  python -B -c 'import ml4co_kit as k,sys; w=(k.TSPWrapper() if sys.argv[1]=="tsp" else k.CVRPWrapper()); w.from_pickle(sys.argv[2]); print(len(w.task_list))' "$problem" "$dataset"
}
run_lehd_fullset () {
  problem="$1"; size="$2"; protocol="$3"; dataset="$4"; dataset_sha="$5"; checkpoint="$6"; checkpoint_sha="$7"
  count="$(dataset_count "$problem" "$dataset")"
  if test "$protocol" = more; then evidence="$LEHD_ARTIFACT_ROOT/preflight/${problem}${size}/more/summary.json"; else evidence="$LEHD_ARTIFACT_ROOT/our_smoke/${problem}${size}/$protocol/summary.json"; fi
  python -B "methods/lehd/$problem/paper_eval.py" --problem-size "$size" --protocol "$protocol" --dataset "$dataset" --expected-dataset-sha256 "$dataset_sha" --upstream "$LEHD_UPSTREAM" --checkpoint "$checkpoint" --expected-checkpoint-sha256 "$checkpoint_sha" --preflight-evidence "$evidence" --scope fullset --offset 0 --count "$count" --output-dir "$LEHD_ARTIFACT_ROOT/fullset/${problem}${size}/$protocol" --resume
}
for protocol in greedy fewer more; do
  run_lehd_fullset tsp 100 "$protocol" "$LEHD_TSP100_DATASET" "$LEHD_TSP100_DATASET_SHA" "$LEHD_TSP_CHECKPOINT" "$LEHD_TSP_CHECKPOINT_SHA"
  run_lehd_fullset tsp 500 "$protocol" "$LEHD_TSP500_DATASET" "$LEHD_TSP500_DATASET_SHA" "$LEHD_TSP_CHECKPOINT" "$LEHD_TSP_CHECKPOINT_SHA"
  run_lehd_fullset tsp 1000 "$protocol" "$LEHD_TSP1000_DATASET" "$LEHD_TSP1000_DATASET_SHA" "$LEHD_TSP_CHECKPOINT" "$LEHD_TSP_CHECKPOINT_SHA"
  run_lehd_fullset cvrp 50 "$protocol" "$LEHD_CVRP50_DATASET" "$LEHD_CVRP50_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA"
  run_lehd_fullset cvrp 100 "$protocol" "$LEHD_CVRP100_DATASET" "$LEHD_CVRP100_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA"
  run_lehd_fullset cvrp 200 "$protocol" "$LEHD_CVRP200_DATASET" "$LEHD_CVRP200_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA"
  run_lehd_fullset cvrp 500 "$protocol" "$LEHD_CVRP500_DATASET" "$LEHD_CVRP500_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA"
  run_lehd_fullset cvrp 1000 "$protocol" "$LEHD_CVRP1000_DATASET" "$LEHD_CVRP1000_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA"
  run_lehd_fullset cvrp 2000 "$protocol" "$LEHD_CVRP2000_DATASET" "$LEHD_CVRP2000_DATASET_SHA" "$LEHD_CVRP_CHECKPOINT" "$LEHD_CVRP_CHECKPOINT_SHA"
done
```

The legacy fullset runs above retain their historical `PAPER_READY` behavior.
They are strict BS1 diagnostic evidence, not the senior-approved baseline
reproduction quality path. Author-batch artifacts provide `Obj`/`Gap`; the
separate BS1 timing probes provide `Time`.
