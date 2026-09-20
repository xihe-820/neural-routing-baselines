# SIL server handoff

Local development stops before any official smoke, ML4CO GPU smoke, preflight,
or fullset. No local output is paper-ready.

## Setup, source, and checkpoints

```bash
conda activate cp311_base
export SERVER_ROOT=/inspire/hdd/global_user/majiale-253108540229/zhang
export BASELINE_PROJECT_ROOT="$SERVER_ROOT/neural-routing-baselines"
cd "$BASELINE_PROJECT_ROOT"
pwd
git rev-parse --show-toplevel
git status --short
git rev-parse HEAD
git pull origin feat/sil

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

export SIL_TSP500_DATASET=<exact-ML4CO-TSP500-pickle>
export SIL_TSP1000_DATASET=<exact-ML4CO-TSP1000-pickle>
export SIL_TSP2000_DATASET=<exact-ML4CO-TSP2000-pickle>
export SIL_TSP5000_DATASET=<exact-ML4CO-TSP5000-pickle>
export SIL_TSP10000_DATASET=<exact-ML4CO-TSP10000-pickle>
export SIL_CVRP500_DATASET=<exact-ML4CO-CVRP500-pickle>
export SIL_CVRP1000_DATASET=<exact-ML4CO-CVRP1000-pickle>
export SIL_CVRP2000_DATASET=<exact-ML4CO-CVRP2000-pickle>
export SIL_TSP500_DATASET_SHA=<audited-sha256>
export SIL_TSP1000_DATASET_SHA=<audited-sha256>
export SIL_TSP2000_DATASET_SHA=<audited-sha256>
export SIL_TSP5000_DATASET_SHA=<audited-sha256>
export SIL_TSP10000_DATASET_SHA=<audited-sha256>
export SIL_CVRP500_DATASET_SHA=<audited-sha256>
export SIL_CVRP1000_DATASET_SHA=<audited-sha256>
export SIL_CVRP2000_DATASET_SHA=<audited-sha256>
mkdir -p "$SIL_ARTIFACT_ROOT"
sha256sum "$SIL_TSP500_DATASET" "$SIL_TSP1000_DATASET" "$SIL_TSP2000_DATASET" "$SIL_TSP5000_DATASET" "$SIL_TSP10000_DATASET" "$SIL_CVRP500_DATASET" "$SIL_CVRP1000_DATASET" "$SIL_CVRP2000_DATASET"
```

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

## Phase 2: ML4CO count-2 smoke

```bash
python -B methods/sil/tsp/paper_eval.py --problem-size 500 --budget fewer --dataset "$SIL_TSP500_DATASET" --expected-dataset-sha256 "$SIL_TSP500_DATASET_SHA" --upstream "$SIL_UPSTREAM" --checkpoint "$SIL_CHECKPOINT_ROOT/checkpoint-tsp1k.pt" --expected-checkpoint-sha256 "$SIL_TSP1K_SHA" --scope our-smoke --offset 0 --count 2 --output-dir "$SIL_ARTIFACT_ROOT/our_smoke/tsp500/fewer"
python -B methods/sil/tsp/paper_eval.py --problem-size 1000 --budget fewer --dataset "$SIL_TSP1000_DATASET" --expected-dataset-sha256 "$SIL_TSP1000_DATASET_SHA" --upstream "$SIL_UPSTREAM" --checkpoint "$SIL_CHECKPOINT_ROOT/checkpoint-tsp1k.pt" --expected-checkpoint-sha256 "$SIL_TSP1K_SHA" --scope our-smoke --offset 0 --count 2 --output-dir "$SIL_ARTIFACT_ROOT/our_smoke/tsp1000/fewer"
python -B methods/sil/tsp/paper_eval.py --problem-size 2000 --budget fewer --dataset "$SIL_TSP2000_DATASET" --expected-dataset-sha256 "$SIL_TSP2000_DATASET_SHA" --upstream "$SIL_UPSTREAM" --checkpoint "$SIL_CHECKPOINT_ROOT/checkpoint-tsp1k.pt" --expected-checkpoint-sha256 "$SIL_TSP1K_SHA" --scope our-smoke --offset 0 --count 2 --output-dir "$SIL_ARTIFACT_ROOT/our_smoke/tsp2000/fewer"
python -B methods/sil/cvrp/paper_eval.py --problem-size 500 --budget fewer --dataset "$SIL_CVRP500_DATASET" --expected-dataset-sha256 "$SIL_CVRP500_DATASET_SHA" --upstream "$SIL_UPSTREAM" --checkpoint "$SIL_CHECKPOINT_ROOT/checkpoint-cvrp1k.pt" --expected-checkpoint-sha256 "$SIL_CVRP1K_SHA" --scope our-smoke --offset 0 --count 2 --output-dir "$SIL_ARTIFACT_ROOT/our_smoke/cvrp500/fewer"
python -B methods/sil/cvrp/paper_eval.py --problem-size 1000 --budget fewer --dataset "$SIL_CVRP1000_DATASET" --expected-dataset-sha256 "$SIL_CVRP1000_DATASET_SHA" --upstream "$SIL_UPSTREAM" --checkpoint "$SIL_CHECKPOINT_ROOT/checkpoint-cvrp1k.pt" --expected-checkpoint-sha256 "$SIL_CVRP1K_SHA" --scope our-smoke --offset 0 --count 2 --output-dir "$SIL_ARTIFACT_ROOT/our_smoke/cvrp1000/fewer"
python -B methods/sil/cvrp/paper_eval.py --problem-size 2000 --budget fewer --dataset "$SIL_CVRP2000_DATASET" --expected-dataset-sha256 "$SIL_CVRP2000_DATASET_SHA" --upstream "$SIL_UPSTREAM" --checkpoint "$SIL_CHECKPOINT_ROOT/checkpoint-cvrp1k.pt" --expected-checkpoint-sha256 "$SIL_CVRP1K_SHA" --scope our-smoke --offset 0 --count 2 --output-dir "$SIL_ARTIFACT_ROOT/our_smoke/cvrp2000/fewer"
```

Require six `KIT_VALIDATED` summaries and inspect both records. TSP500 must use
the initial full-attention path; TSP2000 must initially trigger k=1000. CVRP must
record raw demands, each instance's true capacity, and exactly one normalization.

## Phase 3: fewer/more preflight

```bash
for budget in fewer more; do
  python -B methods/sil/tsp/paper_eval.py --problem-size 1000 --budget "$budget" --dataset "$SIL_TSP1000_DATASET" --expected-dataset-sha256 "$SIL_TSP1000_DATASET_SHA" --upstream "$SIL_UPSTREAM" --checkpoint "$SIL_CHECKPOINT_ROOT/checkpoint-tsp1k.pt" --expected-checkpoint-sha256 "$SIL_TSP1K_SHA" --scope preflight --offset 0 --count 2 --output-dir "$SIL_ARTIFACT_ROOT/preflight/tsp1000/$budget"
  python -B methods/sil/cvrp/paper_eval.py --problem-size 1000 --budget "$budget" --dataset "$SIL_CVRP1000_DATASET" --expected-dataset-sha256 "$SIL_CVRP1000_DATASET_SHA" --upstream "$SIL_UPSTREAM" --checkpoint "$SIL_CHECKPOINT_ROOT/checkpoint-cvrp1k.pt" --expected-checkpoint-sha256 "$SIL_CVRP1K_SHA" --scope preflight --offset 0 --count 2 --output-dir "$SIL_ARTIFACT_ROOT/preflight/cvrp1000/$budget"
done
```

`fewer` is paper-reported PRC50 and `more` is paper-reported PRC500. Both use
random insertion and the same PRC pipeline. Require feasible solutions,
official/independent/Kit agreement, and a credible runtime increase. More need
not improve every instance.

## Phase 4: formal fullset

Only after all earlier gates pass, export each audited wrapper's complete task
count as `SIL_<PROBLEM><SIZE>_COUNT`, then execute the 16 cells. `--resume`
accepts only identical provenance/configuration and restores the saved RNG state.

```bash
dataset_count () {
  problem="$1"; dataset="$2"
  python -B -c 'import ml4co_kit as k,sys; w=(k.TSPWrapper() if sys.argv[1]=="tsp" else k.CVRPWrapper()); w.from_pickle(sys.argv[2]); print(len(w.task_list))' "$problem" "$dataset"
}
export SIL_TSP500_COUNT="$(dataset_count tsp "$SIL_TSP500_DATASET")"
export SIL_TSP1000_COUNT="$(dataset_count tsp "$SIL_TSP1000_DATASET")"
export SIL_TSP2000_COUNT="$(dataset_count tsp "$SIL_TSP2000_DATASET")"
export SIL_TSP5000_COUNT="$(dataset_count tsp "$SIL_TSP5000_DATASET")"
export SIL_TSP10000_COUNT="$(dataset_count tsp "$SIL_TSP10000_DATASET")"
export SIL_CVRP500_COUNT="$(dataset_count cvrp "$SIL_CVRP500_DATASET")"
export SIL_CVRP1000_COUNT="$(dataset_count cvrp "$SIL_CVRP1000_DATASET")"
export SIL_CVRP2000_COUNT="$(dataset_count cvrp "$SIL_CVRP2000_DATASET")"

run_sil_fullset () {
  problem="$1"; size="$2"; budget="$3"; dataset="$4"; dataset_sha="$5"; checkpoint="$6"; checkpoint_sha="$7"; count="$8"
  python -B "methods/sil/$problem/paper_eval.py" --problem-size "$size" --budget "$budget" --dataset "$dataset" --expected-dataset-sha256 "$dataset_sha" --upstream "$SIL_UPSTREAM" --checkpoint "$checkpoint" --expected-checkpoint-sha256 "$checkpoint_sha" --scope fullset --offset 0 --count "$count" --output-dir "$SIL_ARTIFACT_ROOT/fullset/${problem}${size}/$budget" --resume
}
for budget in fewer more; do
  run_sil_fullset tsp 500 "$budget" "$SIL_TSP500_DATASET" "$SIL_TSP500_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-tsp1k.pt" "$SIL_TSP1K_SHA" "$SIL_TSP500_COUNT"
  run_sil_fullset tsp 1000 "$budget" "$SIL_TSP1000_DATASET" "$SIL_TSP1000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-tsp1k.pt" "$SIL_TSP1K_SHA" "$SIL_TSP1000_COUNT"
  run_sil_fullset tsp 2000 "$budget" "$SIL_TSP2000_DATASET" "$SIL_TSP2000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-tsp1k.pt" "$SIL_TSP1K_SHA" "$SIL_TSP2000_COUNT"
  run_sil_fullset tsp 5000 "$budget" "$SIL_TSP5000_DATASET" "$SIL_TSP5000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-tsp5k.pt" "$SIL_TSP5K_SHA" "$SIL_TSP5000_COUNT"
  run_sil_fullset tsp 10000 "$budget" "$SIL_TSP10000_DATASET" "$SIL_TSP10000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-tsp10k.pt" "$SIL_TSP10K_SHA" "$SIL_TSP10000_COUNT"
  run_sil_fullset cvrp 500 "$budget" "$SIL_CVRP500_DATASET" "$SIL_CVRP500_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-cvrp1k.pt" "$SIL_CVRP1K_SHA" "$SIL_CVRP500_COUNT"
  run_sil_fullset cvrp 1000 "$budget" "$SIL_CVRP1000_DATASET" "$SIL_CVRP1000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-cvrp1k.pt" "$SIL_CVRP1K_SHA" "$SIL_CVRP1000_COUNT"
  run_sil_fullset cvrp 2000 "$budget" "$SIL_CVRP2000_DATASET" "$SIL_CVRP2000_DATASET_SHA" "$SIL_CHECKPOINT_ROOT/checkpoint-cvrp1k.pt" "$SIL_CVRP1K_SHA" "$SIL_CVRP2000_COUNT"
done
```

A usable cell has `summary.json status=PAPER_READY`, every ordered dataset index
exactly once, and every record `KIT_VALIDATED`. `mean_instance_gap_percent` is
the mean of per-instance gaps. Time is mean BS1 solver latency; Total is the sum.
