# SERVER_RUNBOOK

These commands are run manually by the user. They do not require Codex SSH access, modify official source, or install packages. Use an approved project commit that contains the integration being tested.

## 1. Repository checkout and paths

```bash
read -r -p 'Baseline project absolute path: ' BASELINE_PROJECT_ROOT
read -r -p 'Approved project commit: ' BASELINE_PROJECT_COMMIT
read -r -p 'Official checkout root: ' BASELINE_UPSTREAM_ROOT
export BASELINE_PROJECT_ROOT BASELINE_PROJECT_COMMIT BASELINE_UPSTREAM_ROOT
export ML4CO_DATA_ROOT=/inspire/hdd/global_user/majiale-253108540229/ML4CO-Bench-101
export BASELINE_ARTIFACT_ROOT="$BASELINE_PROJECT_ROOT/artifacts/server"
export PYTHONDONTWRITEBYTECODE=1

if [ -d "$BASELINE_PROJECT_ROOT/.git" ]; then
  git -C "$BASELINE_PROJECT_ROOT" checkout master
  git -C "$BASELINE_PROJECT_ROOT" pull --ff-only origin master
else
  git clone https://github.com/xihe-820/neural-routing-baselines.git "$BASELINE_PROJECT_ROOT"
fi
git -C "$BASELINE_PROJECT_ROOT" checkout --detach "$BASELINE_PROJECT_COMMIT"
test "$(git -C "$BASELINE_PROJECT_ROOT" rev-parse HEAD)" = "$BASELINE_PROJECT_COMMIT"
mkdir -p "$BASELINE_ARTIFACT_ROOT"
cd "$BASELINE_PROJECT_ROOT"
conda activate cp311_base
```

If activation fails, return the error without creating or changing an environment.

## 2. Environment audit

```bash
python -B scripts/audit_environment.py --label server-user-executed \
  --output "$BASELINE_ARTIFACT_ROOT/environment/server_environment.json"
```

## 3. Dataset, upstream and checkpoint identity

```bash
python -B scripts/audit_datasets.py --dataset-root "$ML4CO_DATA_ROOT" \
  --expected-manifest manifests/public_datasets.json --label server-user-executed \
  --output "$BASELINE_ARTIFACT_ROOT/audit/server_datasets.json"

python -B scripts/audit_assets.py --upstream-root "$BASELINE_UPSTREAM_ROOT" \
  --output "$BASELINE_ARTIFACT_ROOT/audit/server_upstreams.json"

python -B scripts/audit_assets.py --upstream-root "$BASELINE_UPSTREAM_ROOT" \
  --compare-manifest manifests/checkpoints.yaml --load-checkpoints \
  --output "$BASELINE_ARTIFACT_ROOT/audit/server_checkpoints.json"
```

Do not continue if the requested dataset SHA, upstream URL/commit/clean state, checkpoint SHA, checkpoint payload load or model identity fails.

For the GLOP paper scope, audit TSP 100/500/1K/2K/5K/10K and CVRP
500/1K/2K without inference:

```bash
python -B scripts/audit_glop_paper_datasets.py --dataset-root "$ML4CO_DATA_ROOT" \
  --output "$BASELINE_ARTIFACT_ROOT/audit/glop_paper_datasets.json"
```

## 4. MVMoE paths

```bash
read -r -p 'Exact CVRP50 benchmark pickle: ' CVRP50_DATASET
read -r -p 'Exact CVRP100 benchmark pickle: ' CVRP100_DATASET
read -r -p 'Exact CVRPTW50 benchmark pickle: ' CVRPTW50_DATASET
export CVRPTW100_DATASET="$ML4CO_DATA_ROOT/test_dataset/cvrptw/cvrptw100_pyvrp-20s_25.431.pkl"
export MVMOE_UPSTREAM="$BASELINE_UPSTREAM_ROOT/Routing-MVMoE"
export MVMOE_N50_CHECKPOINT="$MVMOE_UPSTREAM/pretrained/mvmoe_4e_n50/epoch-5000.pt"
export MVMOE_N100_CHECKPOINT="$MVMOE_UPSTREAM/pretrained/mvmoe_4e_n100/epoch-5000.pt"
export CVRP50_DATASET CVRP100_DATASET CVRPTW50_DATASET CVRPTW100_DATASET
export MVMOE_UPSTREAM MVMOE_N50_CHECKPOINT MVMOE_N100_CHECKPOINT
```

The absolute CVRPTW100 path is a deployment value in this runbook. Program source receives every path through CLI arguments.

## 5. MVMoE CVRP50 and CVRP100

```bash
python -B methods/mvmoe/cvrp/prepare_instances.py \
  --dataset "$CVRP50_DATASET" --problem-size 50 --offset 0 --count 5 \
  --output "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrp50/input_first5.npz"
python -B methods/mvmoe/cvrp/run.py --problem-size 50 \
  --input "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrp50/input_first5.npz" \
  --upstream "$MVMOE_UPSTREAM" --checkpoint "$MVMOE_N50_CHECKPOINT" \
  --output "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrp50/official_search_aug8_first5.json" \
  --aug-factor 8 --seed 2024 --device cuda:0

python -B methods/mvmoe/cvrp/prepare_instances.py \
  --dataset "$CVRP100_DATASET" --problem-size 100 --offset 0 --count 5 \
  --output "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrp100/input_first5.npz"
python -B methods/mvmoe/cvrp/run.py --problem-size 100 \
  --input "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrp100/input_first5.npz" \
  --upstream "$MVMOE_UPSTREAM" --checkpoint "$MVMOE_N100_CHECKPOINT" \
  --output "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrp100/official_search_aug8_first5.json" \
  --aug-factor 8 --seed 2024 --device cuda:0
```

## 6. MVMoE CVRPTW50 and CVRPTW100

The adapter rejects a nonzero depot lower bound or mixed depot windows. The runner assigns the validated depot window before `VRPTWEnv.load_problems`.

```bash
python -B methods/mvmoe/cvrptw/prepare_instances.py \
  --dataset "$CVRPTW50_DATASET" --problem-size 50 --offset 0 --count 5 \
  --output "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrptw50/input_first5.npz"
python -B methods/mvmoe/cvrptw/run.py --problem-size 50 \
  --input "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrptw50/input_first5.npz" \
  --upstream "$MVMOE_UPSTREAM" --checkpoint "$MVMOE_N50_CHECKPOINT" \
  --output "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrptw50/official_search_aug8_first5.json" \
  --aug-factor 8 --seed 2024 --device cuda:0

python -B methods/mvmoe/cvrptw/prepare_instances.py \
  --dataset "$CVRPTW100_DATASET" --problem-size 100 --offset 0 --count 5 \
  --output "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrptw100/input_first5.npz"
python -B methods/mvmoe/cvrptw/run.py --problem-size 100 \
  --input "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrptw100/input_first5.npz" \
  --upstream "$MVMOE_UPSTREAM" --checkpoint "$MVMOE_N100_CHECKPOINT" \
  --output "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrptw100/official_search_aug8_first5.json" \
  --aug-factor 8 --seed 2024 --device cuda:0
```

Each runner records the actual selected route and immediately applies the independent validator to the original benchmark fields.

## 7. ML4CO-Kit validation

```bash
python -B methods/mvmoe/cvrp/validate_with_kit.py \
  --input "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrp50/official_search_aug8_first5.json" \
  --dataset "$CVRP50_DATASET" \
  --output "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrp50/official_search_aug8_first5_validated.json"
python -B methods/mvmoe/cvrp/validate_with_kit.py \
  --input "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrp100/official_search_aug8_first5.json" \
  --dataset "$CVRP100_DATASET" \
  --output "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrp100/official_search_aug8_first5_validated.json"
python -B methods/mvmoe/cvrptw/validate_with_kit.py --problem-size 50 \
  --input "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrptw50/official_search_aug8_first5.json" \
  --dataset "$CVRPTW50_DATASET" \
  --output "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrptw50/official_search_aug8_first5_validated.json"
python -B methods/mvmoe/cvrptw/validate_with_kit.py --problem-size 100 \
  --input "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrptw100/official_search_aug8_first5.json" \
  --dataset "$CVRPTW100_DATASET" \
  --output "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrptw100/official_search_aug8_first5_validated.json"
```

Every row must have `evidence_status=LOCAL_VERIFIED` and all four completion gates true.

## Paper evaluation / MVMoE

These commands create batch-one paper evidence under the ignored
`artifacts/paper/mvmoe` directory. Keep the fixed project checkout clean.

```bash
export MVMOE_PAPER_ROOT="$BASELINE_PROJECT_ROOT/artifacts/paper/mvmoe"
mkdir -p "$MVMOE_PAPER_ROOT"
```

The CVRP preflight commands remain unchanged. For the scaled migration, run the
two CVRPTW preflights before any CVRPTW production chunk. Each evaluator prints
method, problem, size, original batch size, POMO size, augmentation, both asset
hashes, GPU, objective, reference, gap, runtime and feasibility per instance.

```bash
python -B methods/mvmoe/cvrp/prepare_instances.py --dataset "$CVRP50_DATASET" \
  --problem-size 50 --offset 0 --count 2 \
  --output "$MVMOE_PAPER_ROOT/cvrp50/preflight/input.npz"
python -B methods/mvmoe/cvrp/paper_eval.py --problem-size 50 \
  --input "$MVMOE_PAPER_ROOT/cvrp50/preflight/input.npz" \
  --upstream "$MVMOE_UPSTREAM" --checkpoint "$MVMOE_N50_CHECKPOINT" \
  --output-dir "$MVMOE_PAPER_ROOT/cvrp50/preflight/chunk_00000_00002" \
  --warmup-instances 2 --device cuda:0

python -B methods/mvmoe/cvrp/prepare_instances.py --dataset "$CVRP100_DATASET" \
  --problem-size 100 --offset 0 --count 2 \
  --output "$MVMOE_PAPER_ROOT/cvrp100/preflight/input.npz"
python -B methods/mvmoe/cvrp/paper_eval.py --problem-size 100 \
  --input "$MVMOE_PAPER_ROOT/cvrp100/preflight/input.npz" \
  --upstream "$MVMOE_UPSTREAM" --checkpoint "$MVMOE_N100_CHECKPOINT" \
  --output-dir "$MVMOE_PAPER_ROOT/cvrp100/preflight/chunk_00000_00002" \
  --warmup-instances 2 --device cuda:0

python -B methods/mvmoe/cvrptw/prepare_instances.py --dataset "$CVRPTW50_DATASET" \
  --problem-size 50 --offset 0 --count 2 \
  --output "$MVMOE_PAPER_ROOT/cvrptw50_scaled/preflight/input.npz"
python -B methods/mvmoe/cvrptw/paper_eval.py --problem-size 50 \
  --input "$MVMOE_PAPER_ROOT/cvrptw50_scaled/preflight/input.npz" \
  --upstream "$MVMOE_UPSTREAM" --checkpoint "$MVMOE_N50_CHECKPOINT" \
  --output-dir "$MVMOE_PAPER_ROOT/cvrptw50_scaled/preflight/chunk_00000_00002" \
  --warmup-instances 2 --device cuda:0

python -B methods/mvmoe/cvrptw/prepare_instances.py --dataset "$CVRPTW100_DATASET" \
  --problem-size 100 --offset 0 --count 2 \
  --output "$MVMOE_PAPER_ROOT/cvrptw100_scaled/preflight/input.npz"
python -B methods/mvmoe/cvrptw/paper_eval.py --problem-size 100 \
  --input "$MVMOE_PAPER_ROOT/cvrptw100_scaled/preflight/input.npz" \
  --upstream "$MVMOE_UPSTREAM" --checkpoint "$MVMOE_N100_CHECKPOINT" \
  --output-dir "$MVMOE_PAPER_ROOT/cvrptw100_scaled/preflight/chunk_00000_00002" \
  --warmup-instances 2 --device cuda:0
```

The existing `cvrptw50` and `cvrptw100` directories are the verified unscaled
control. The `*_scaled` directories are the canonical scaled paper run. Before
any full-set run, compare each scaled preflight record for indices 0 and 1 with
the existing scaling-audit record: canonical solution, best augmentation/POMO
indices, original objective, gap and scaler must match (apart from serialized
floating-point representation). Any route mismatch blocks production.

```bash
python -B scripts/compare_mvmoe_cvrptw_scaled_preflight.py \
  --paper-chunk "$MVMOE_PAPER_ROOT/cvrptw50_scaled/preflight/chunk_00000_00002" \
  --scaling-audit-dir "$BASELINE_PROJECT_ROOT/artifacts/audit/mvmoe_cvrptw_scaling/cvrptw50"
python -B scripts/compare_mvmoe_cvrptw_scaled_preflight.py \
  --paper-chunk "$MVMOE_PAPER_ROOT/cvrptw100_scaled/preflight/chunk_00000_00002" \
  --scaling-audit-dir "$BASELINE_PROJECT_ROOT/artifacts/audit/mvmoe_cvrptw_scaling/cvrptw100"
```

After the required preflights and both scaled comparison gates pass, the future
production commands are below. Re-running an exact command resumes completed
records; changed provenance or configuration fails.

```bash
for offset in $(seq 0 1000 9000); do
  stop=$((offset + 1000))
  python -B methods/mvmoe/cvrp/prepare_instances.py --dataset "$CVRP50_DATASET" \
    --problem-size 50 --offset "$offset" --count 1000 \
    --output "$MVMOE_PAPER_ROOT/cvrp50/production/input_${offset}_${stop}.npz"
  python -B methods/mvmoe/cvrp/paper_eval.py --problem-size 50 \
    --input "$MVMOE_PAPER_ROOT/cvrp50/production/input_${offset}_${stop}.npz" \
    --upstream "$MVMOE_UPSTREAM" --checkpoint "$MVMOE_N50_CHECKPOINT" \
    --output-dir "$MVMOE_PAPER_ROOT/cvrp50/production/chunk_${offset}_${stop}" \
    --warmup-instances 2 --device cuda:0
done

for offset in $(seq 0 1000 9000); do
  stop=$((offset + 1000))
  python -B methods/mvmoe/cvrp/prepare_instances.py --dataset "$CVRP100_DATASET" \
    --problem-size 100 --offset "$offset" --count 1000 \
    --output "$MVMOE_PAPER_ROOT/cvrp100/production/input_${offset}_${stop}.npz"
  python -B methods/mvmoe/cvrp/paper_eval.py --problem-size 100 \
    --input "$MVMOE_PAPER_ROOT/cvrp100/production/input_${offset}_${stop}.npz" \
    --upstream "$MVMOE_UPSTREAM" --checkpoint "$MVMOE_N100_CHECKPOINT" \
    --output-dir "$MVMOE_PAPER_ROOT/cvrp100/production/chunk_${offset}_${stop}" \
    --warmup-instances 2 --device cuda:0
done

for offset in $(seq 0 250 750); do
  stop=$((offset + 250))
  python -B methods/mvmoe/cvrptw/prepare_instances.py --dataset "$CVRPTW50_DATASET" \
    --problem-size 50 --offset "$offset" --count 250 \
    --output "$MVMOE_PAPER_ROOT/cvrptw50_scaled/production/input_${offset}_${stop}.npz"
  python -B methods/mvmoe/cvrptw/paper_eval.py --problem-size 50 \
    --input "$MVMOE_PAPER_ROOT/cvrptw50_scaled/production/input_${offset}_${stop}.npz" \
    --upstream "$MVMOE_UPSTREAM" --checkpoint "$MVMOE_N50_CHECKPOINT" \
    --output-dir "$MVMOE_PAPER_ROOT/cvrptw50_scaled/production/chunk_${offset}_${stop}" \
    --warmup-instances 2 --device cuda:0
done

for offset in $(seq 0 250 750); do
  stop=$((offset + 250))
  python -B methods/mvmoe/cvrptw/prepare_instances.py --dataset "$CVRPTW100_DATASET" \
    --problem-size 100 --offset "$offset" --count 250 \
    --output "$MVMOE_PAPER_ROOT/cvrptw100_scaled/production/input_${offset}_${stop}.npz"
  python -B methods/mvmoe/cvrptw/paper_eval.py --problem-size 100 \
    --input "$MVMOE_PAPER_ROOT/cvrptw100_scaled/production/input_${offset}_${stop}.npz" \
    --upstream "$MVMOE_UPSTREAM" --checkpoint "$MVMOE_N100_CHECKPOINT" \
    --output-dir "$MVMOE_PAPER_ROOT/cvrptw100_scaled/production/chunk_${offset}_${stop}" \
    --warmup-instances 2 --device cuda:0
done
```

Load each full dataset once to apply the secondary Kit gate across its chunks,
then create one strict full-set summary per paper row.

```bash
python -B scripts/validate_paper_results_with_kit.py --dataset "$CVRP50_DATASET" \
  --chunk-dirs "$MVMOE_PAPER_ROOT"/cvrp50/production/chunk_*
python -B scripts/validate_paper_results_with_kit.py --dataset "$CVRP100_DATASET" \
  --chunk-dirs "$MVMOE_PAPER_ROOT"/cvrp100/production/chunk_*
python -B scripts/validate_paper_results_with_kit.py --dataset "$CVRPTW50_DATASET" \
  --chunk-dirs "$MVMOE_PAPER_ROOT"/cvrptw50_scaled/production/chunk_*
python -B scripts/validate_paper_results_with_kit.py --dataset "$CVRPTW100_DATASET" \
  --chunk-dirs "$MVMOE_PAPER_ROOT"/cvrptw100_scaled/production/chunk_*

python -B scripts/summarize_paper_results.py \
  --chunk-dirs "$MVMOE_PAPER_ROOT"/cvrp50/production/chunk_* \
  --output "$MVMOE_PAPER_ROOT/cvrp50/summary.json"
python -B scripts/summarize_paper_results.py \
  --chunk-dirs "$MVMOE_PAPER_ROOT"/cvrp100/production/chunk_* \
  --output "$MVMOE_PAPER_ROOT/cvrp100/summary.json"
python -B scripts/summarize_paper_results.py \
  --chunk-dirs "$MVMOE_PAPER_ROOT"/cvrptw50_scaled/production/chunk_* \
  --output "$MVMOE_PAPER_ROOT/cvrptw50_scaled/summary.json"
python -B scripts/summarize_paper_results.py \
  --chunk-dirs "$MVMOE_PAPER_ROOT"/cvrptw100_scaled/production/chunk_* \
  --output "$MVMOE_PAPER_ROOT/cvrptw100_scaled/summary.json"
```

Only the four complete summaries may report `PAPER_READY`; preflight chunks are
never paper results.

### MVMoE CVRPTW native BS10 extension

The original scaled BS1 evidence remains valid. The commands below create a
separate BS10 tree and use the same checkpoint, per-instance scaler, Aug8,
POMO=N, argmax selection, independent validator, and Kit gate. Each call to the
official environment receives ten original instances at once.

```bash
for size in 50 100; do
  dataset_var="CVRPTW${size}_DATASET"
  checkpoint_var="MVMOE_N${size}_CHECKPOINT"
  dataset="${!dataset_var}"
  checkpoint="${!checkpoint_var}"
  python -B methods/mvmoe/cvrptw/prepare_instances.py \
    --dataset "$dataset" --problem-size "$size" --offset 0 --count 10 \
    --output "$MVMOE_PAPER_ROOT/cvrptw${size}_scaled/bs10/preflight/input.npz"
  python -B methods/mvmoe/cvrptw/paper_eval.py \
    --problem-size "$size" --batch-size 10 \
    --input "$MVMOE_PAPER_ROOT/cvrptw${size}_scaled/bs10/preflight/input.npz" \
    --upstream "$MVMOE_UPSTREAM" --checkpoint "$checkpoint" \
    --output-dir "$MVMOE_PAPER_ROOT/cvrptw${size}_scaled/bs10/preflight/chunk_00000_00010" \
    --warmup-batches 0 --device cuda:0
done
```

Only after both one-batch preflights pass, prepare and run four aligned
250-instance chunks per size. Re-running an identical command resumes at the
next complete ten-instance batch; a partial batch fails closed.

```bash
for size in 50 100; do
  dataset_var="CVRPTW${size}_DATASET"
  checkpoint_var="MVMOE_N${size}_CHECKPOINT"
  dataset="${!dataset_var}"
  checkpoint="${!checkpoint_var}"
  for offset in 0 250 500 750; do
    stop=$((offset + 250))
    python -B methods/mvmoe/cvrptw/prepare_instances.py \
      --dataset "$dataset" --problem-size "$size" --offset "$offset" --count 250 \
      --output "$MVMOE_PAPER_ROOT/cvrptw${size}_scaled/bs10/production/input_${offset}_${stop}.npz"
    python -B methods/mvmoe/cvrptw/paper_eval.py \
      --problem-size "$size" --batch-size 10 \
      --input "$MVMOE_PAPER_ROOT/cvrptw${size}_scaled/bs10/production/input_${offset}_${stop}.npz" \
      --upstream "$MVMOE_UPSTREAM" --checkpoint "$checkpoint" \
      --output-dir "$MVMOE_PAPER_ROOT/cvrptw${size}_scaled/bs10/production/chunk_${offset}_${stop}" \
      --warmup-batches 2 --device cuda:0
  done
  python -B scripts/validate_paper_results_with_kit.py --dataset "$dataset" \
    --chunk-dirs "$MVMOE_PAPER_ROOT"/cvrptw${size}_scaled/bs10/production/chunk_*
  python -B scripts/summarize_paper_results.py \
    --chunk-dirs "$MVMOE_PAPER_ROOT"/cvrptw${size}_scaled/bs10/production/chunk_* \
    --output "$MVMOE_PAPER_ROOT/cvrptw${size}_scaled/bs10/summary.json"
done
```

The BS10 summary must report 1000 instances, 100 native batches,
`time_mean_batch_seconds`, and `time_total_seconds`. Do not divide the batch
latency by ten.

## 8. NeuOpt CVRP50 and CVRP100

Run both sizes from the same integration. These commands keep the official checkout read-only and use the official 1000-step configuration.

If `tensorboard_logger` is absent, the runner enables a repository-owned import-only guard shim for `no_tb` inference; any attempt to instantiate its `Logger` fails immediately.

```bash
export NEUOPT_UPSTREAM="$BASELINE_UPSTREAM_ROOT/NeuOpt"
export NEUOPT_N50_CHECKPOINT="$NEUOPT_UPSTREAM/pre-trained/cvrp50.pt"
export NEUOPT_N100_CHECKPOINT="$NEUOPT_UPSTREAM/pre-trained/cvrp100.pt"

test "$(git -C "$NEUOPT_UPSTREAM" rev-parse HEAD)" = "ccf6b5f0f6a8fda2792b4be11d4ec35390a8139b"
test -z "$(git -C "$NEUOPT_UPSTREAM" status --porcelain)"
test "$(sha256sum "$CVRP50_DATASET" | cut -d' ' -f1)" = "eea12fbefe9c1bcc008d56ecfc1c50dadd64ac774f3547774c9fade8a7baa6c2"
test "$(sha256sum "$CVRP100_DATASET" | cut -d' ' -f1)" = "bb47d5a113848e5a404edefc562d5d2ef6b0ade1aafc287828bdf60364e23532"
test "$(sha256sum "$NEUOPT_N50_CHECKPOINT" | cut -d' ' -f1)" = "1cd201ca47888e51068a157389460641c81d71054f064d9c8ea1743312289e3a"
test "$(sha256sum "$NEUOPT_N100_CHECKPOINT" | cut -d' ' -f1)" = "502a5904182306c1a3f65f7b1a8503a609ff2a044db054abcf93690af36594fb"

python -B methods/neuopt/cvrp/prepare_instances.py \
  --dataset "$CVRP50_DATASET" --problem-size 50 --offset 0 --count 5 \
  --output "$BASELINE_ARTIFACT_ROOT/neuopt_cvrp50/input_first5.npz"
python -B methods/neuopt/cvrp/run.py --problem-size 50 \
  --input "$BASELINE_ARTIFACT_ROOT/neuopt_cvrp50/input_first5.npz" \
  --upstream "$NEUOPT_UPSTREAM" --checkpoint "$NEUOPT_N50_CHECKPOINT" \
  --output "$BASELINE_ARTIFACT_ROOT/neuopt_cvrp50/official_first5.json" --device cuda:0
python -B methods/neuopt/cvrp/validate_with_kit.py --problem-size 50 \
  --input "$BASELINE_ARTIFACT_ROOT/neuopt_cvrp50/official_first5.json" \
  --dataset "$CVRP50_DATASET" \
  --output "$BASELINE_ARTIFACT_ROOT/neuopt_cvrp50/official_first5_validated.json"

python -B methods/neuopt/cvrp/prepare_instances.py \
  --dataset "$CVRP100_DATASET" --problem-size 100 --offset 0 --count 5 \
  --output "$BASELINE_ARTIFACT_ROOT/neuopt_cvrp100/input_first5.npz"
python -B methods/neuopt/cvrp/run.py --problem-size 100 \
  --input "$BASELINE_ARTIFACT_ROOT/neuopt_cvrp100/input_first5.npz" \
  --upstream "$NEUOPT_UPSTREAM" --checkpoint "$NEUOPT_N100_CHECKPOINT" \
  --output "$BASELINE_ARTIFACT_ROOT/neuopt_cvrp100/official_first5.json" --device cuda:0
python -B methods/neuopt/cvrp/validate_with_kit.py --problem-size 100 \
  --input "$BASELINE_ARTIFACT_ROOT/neuopt_cvrp100/official_first5.json" \
  --dataset "$CVRP100_DATASET" \
  --output "$BASELINE_ARTIFACT_ROOT/neuopt_cvrp100/official_first5_validated.json"

test "$(git rev-parse HEAD)" = "$BASELINE_PROJECT_COMMIT"
test -z "$(git -C "$NEUOPT_UPSTREAM" status --porcelain)"
sha256sum "$BASELINE_ARTIFACT_ROOT/neuopt_cvrp50/official_first5_validated.json" \
  "$BASELINE_ARTIFACT_ROOT/neuopt_cvrp100/official_first5_validated.json"
```

Every result row must report all four completion gates as true. Return both validated artifacts, their SHA256 lines, the environment audit and both repository clean-state outputs.

### NeuOpt final paper production

The latest frozen protocol is CVRP50/100, D2A=1, T=20/50. BS1 supplies
Complete Results; BS100 supplies Parallel Table 8. BS1 uses the guarded
repository-owned in-memory shape shim, while BS100 requires the unmodified
pinned decoder. Both paths time only `record=False`; the same-RNG
`record=True` replay and all validation are untimed. If `tensorboard_logger`
is absent, the import-only guard described above remains valid for `no_tb`.

The older D2A=5/T=1k/5k calibration is **LEGACY / DO NOT USE FOR FINAL
PAPER**. Its existing artifacts and readers remain historical evidence.

Prepare exact full-set neutral inputs once. All later offsets address these
same 0..9999 dataset indices.

```bash
export NEUOPT_PAPER_ROOT="$BASELINE_ARTIFACT_ROOT/paper/neuopt_final"
mkdir -p "$NEUOPT_PAPER_ROOT/cvrp50" "$NEUOPT_PAPER_ROOT/cvrp100"

python -B methods/neuopt/cvrp/prepare_instances.py \
  --dataset "$CVRP50_DATASET" --problem-size 50 --offset 0 --count 10000 \
  --output "$NEUOPT_PAPER_ROOT/cvrp50/fullset_input.npz"
python -B methods/neuopt/cvrp/prepare_instances.py \
  --dataset "$CVRP100_DATASET" --problem-size 100 --offset 0 --count 10000 \
  --output "$NEUOPT_PAPER_ROOT/cvrp100/fullset_input.npz"
```

Run the four mandatory T20 correctness preflights. Reusing an output directory
resumes only an exact matching identity; use a new directory for any changed
code or protocol.

```bash
python -B methods/neuopt/cvrp/paper_eval.py --problem-size 50 \
  --input "$NEUOPT_PAPER_ROOT/cvrp50/fullset_input.npz" \
  --dataset "$CVRP50_DATASET" --upstream "$NEUOPT_UPSTREAM" \
  --checkpoint "$NEUOPT_N50_CHECKPOINT" --d2a 1 --T-max 20 --batch-size 1 \
  --offset 0 --count 1 --warmup-batches 1 --device cuda:0 \
  --output-dir "$NEUOPT_PAPER_ROOT/cvrp50/preflight_t20_bs1"

python -B methods/neuopt/cvrp/paper_eval.py --problem-size 50 \
  --input "$NEUOPT_PAPER_ROOT/cvrp50/fullset_input.npz" \
  --dataset "$CVRP50_DATASET" --upstream "$NEUOPT_UPSTREAM" \
  --checkpoint "$NEUOPT_N50_CHECKPOINT" --d2a 1 --T-max 20 --batch-size 100 \
  --offset 0 --count 100 --warmup-batches 1 --device cuda:0 \
  --output-dir "$NEUOPT_PAPER_ROOT/cvrp50/preflight_t20_bs100"

python -B methods/neuopt/cvrp/paper_eval.py --problem-size 100 \
  --input "$NEUOPT_PAPER_ROOT/cvrp100/fullset_input.npz" \
  --dataset "$CVRP100_DATASET" --upstream "$NEUOPT_UPSTREAM" \
  --checkpoint "$NEUOPT_N100_CHECKPOINT" --d2a 1 --T-max 20 --batch-size 1 \
  --offset 0 --count 1 --warmup-batches 1 --device cuda:0 \
  --output-dir "$NEUOPT_PAPER_ROOT/cvrp100/preflight_t20_bs1"

python -B methods/neuopt/cvrp/paper_eval.py --problem-size 100 \
  --input "$NEUOPT_PAPER_ROOT/cvrp100/fullset_input.npz" \
  --dataset "$CVRP100_DATASET" --upstream "$NEUOPT_UPSTREAM" \
  --checkpoint "$NEUOPT_N100_CHECKPOINT" --d2a 1 --T-max 20 --batch-size 100 \
  --offset 0 --count 100 --warmup-batches 1 --device cuda:0 \
  --output-dir "$NEUOPT_PAPER_ROOT/cvrp100/preflight_t20_bs100"

python -B - \
  "$NEUOPT_PAPER_ROOT/cvrp50/preflight_t20_bs1/metadata.json" \
  "$NEUOPT_PAPER_ROOT/cvrp50/preflight_t20_bs100/metadata.json" \
  "$NEUOPT_PAPER_ROOT/cvrp100/preflight_t20_bs1/metadata.json" \
  "$NEUOPT_PAPER_ROOT/cvrp100/preflight_t20_bs100/metadata.json" <<'PY'
import json, pathlib, sys
for name in sys.argv[1:]:
    metadata = json.loads(pathlib.Path(name).read_text())
    identity = metadata["resume_identity"]
    protocol = identity["paper_protocol"]
    compatibility = identity["decoder_compatibility"]
    batch_size = protocol["original_batch_size"]
    assert metadata["state"] == "KIT_VALIDATED"
    assert protocol["D2A"] == protocol["val_m"] == 1
    assert compatibility["internal_decoder_batch_size"] == batch_size
    assert compatibility["bs1_shape_shim"] is (batch_size == 1)
    assert compatibility["official_source_modified"] is False
    assert identity["upstream"]["dirty"] is False
PY
```

After all four preflights pass, run BS100 full sets first. Each output is one
resumable 1,000-instance chunk containing ten native BS100 rollouts.

```bash
for SIZE in 50 100; do
  if test "$SIZE" = 50; then
    DATASET="$CVRP50_DATASET"; CHECKPOINT="$NEUOPT_N50_CHECKPOINT"
  else
    DATASET="$CVRP100_DATASET"; CHECKPOINT="$NEUOPT_N100_CHECKPOINT"
  fi
  INPUT="$NEUOPT_PAPER_ROOT/cvrp$SIZE/fullset_input.npz"
  for T_MAX in 20 50; do
    for OFFSET in 0 1000 2000 3000 4000 5000 6000 7000 8000 9000; do
      printf -v CHUNK 'chunk_%05d_%05d' "$OFFSET" "$((OFFSET + 1000))"
      python -B methods/neuopt/cvrp/paper_eval.py --problem-size "$SIZE" \
        --input "$INPUT" --dataset "$DATASET" --upstream "$NEUOPT_UPSTREAM" \
        --checkpoint "$CHECKPOINT" --d2a 1 --T-max "$T_MAX" --batch-size 100 \
        --offset "$OFFSET" --count 1000 --warmup-batches 1 --device cuda:0 \
        --output-dir "$NEUOPT_PAPER_ROOT/cvrp$SIZE/t$T_MAX/bs100/$CHUNK"
    done
  done
done
```

Aggregate exact BS100 coverage. The explicit loop constructs the ten expected
directories; the aggregator itself rejects missing, duplicate, mixed, or
non-finalized evidence.

```bash
for SIZE in 50 100; do
  for T_MAX in 20 50; do
    CHUNKS=()
    for OFFSET in 0 1000 2000 3000 4000 5000 6000 7000 8000 9000; do
      printf -v CHUNK 'chunk_%05d_%05d' "$OFFSET" "$((OFFSET + 1000))"
      CHUNKS+=("$NEUOPT_PAPER_ROOT/cvrp$SIZE/t$T_MAX/bs100/$CHUNK")
    done
    python -B methods/neuopt/cvrp/paper_aggregate.py \
      --chunk-dirs "${CHUNKS[@]}" --expected-offset 0 --expected-count 10000 \
      --scope fullset \
      --output "$NEUOPT_PAPER_ROOT/cvrp$SIZE/t$T_MAX/bs100_fullset_summary.json"
  done
done
```

Then run the same fixed first100 indices at BS1. These summaries are explicitly
estimated timing subsets and include mean, median, min, max, standard
deviation, sample count, and exact indices.

```bash
for SIZE in 50 100; do
  if test "$SIZE" = 50; then
    DATASET="$CVRP50_DATASET"; CHECKPOINT="$NEUOPT_N50_CHECKPOINT"
  else
    DATASET="$CVRP100_DATASET"; CHECKPOINT="$NEUOPT_N100_CHECKPOINT"
  fi
  for T_MAX in 20 50; do
    BS1_DIR="$NEUOPT_PAPER_ROOT/cvrp$SIZE/t$T_MAX/bs1_first100"
    python -B methods/neuopt/cvrp/paper_eval.py --problem-size "$SIZE" \
      --input "$NEUOPT_PAPER_ROOT/cvrp$SIZE/fullset_input.npz" \
      --dataset "$DATASET" --upstream "$NEUOPT_UPSTREAM" \
      --checkpoint "$CHECKPOINT" --d2a 1 --T-max "$T_MAX" --batch-size 1 \
      --offset 0 --count 100 --warmup-batches 1 --device cuda:0 \
      --output-dir "$BS1_DIR"
    python -B methods/neuopt/cvrp/paper_aggregate.py \
      --chunk-dirs "$BS1_DIR" --expected-offset 0 --expected-count 100 \
      --scope timing_subset \
      --output "$NEUOPT_PAPER_ROOT/cvrp$SIZE/t$T_MAX/bs1_first100_summary.json"
  done
done
```

Build the explicit hybrid candidates. Each report extracts the exact first100
from BS100 full-set records and compares its quality with the BS1 first100.
Review the deltas manually. If the reviewer considers them abnormal, rerun the
same command with `--hybrid-quality-warning`; the code applies no threshold.

```bash
for SIZE in 50 100; do
  for T_MAX in 20 50; do
    python -B methods/neuopt/cvrp/hybrid_summary.py \
      --quality-summary "$NEUOPT_PAPER_ROOT/cvrp$SIZE/t$T_MAX/bs100_fullset_summary.json" \
      --time-summary "$NEUOPT_PAPER_ROOT/cvrp$SIZE/t$T_MAX/bs1_first100_summary.json" \
      --output "$NEUOPT_PAPER_ROOT/cvrp$SIZE/t$T_MAX/hybrid_candidate.json"
  done
done
```

Optional BS1 full sets are not part of the default production sequence. Run
them only after user review, using these resumable 1,000-instance chunks and
the same strict aggregation command.

```bash
for SIZE in 50 100; do
  if test "$SIZE" = 50; then
    DATASET="$CVRP50_DATASET"; CHECKPOINT="$NEUOPT_N50_CHECKPOINT"
  else
    DATASET="$CVRP100_DATASET"; CHECKPOINT="$NEUOPT_N100_CHECKPOINT"
  fi
  for T_MAX in 20 50; do
    for OFFSET in 0 1000 2000 3000 4000 5000 6000 7000 8000 9000; do
      printf -v CHUNK 'chunk_%05d_%05d' "$OFFSET" "$((OFFSET + 1000))"
      python -B methods/neuopt/cvrp/paper_eval.py --problem-size "$SIZE" \
        --input "$NEUOPT_PAPER_ROOT/cvrp$SIZE/fullset_input.npz" \
        --dataset "$DATASET" --upstream "$NEUOPT_UPSTREAM" \
        --checkpoint "$CHECKPOINT" --d2a 1 --T-max "$T_MAX" --batch-size 1 \
        --offset "$OFFSET" --count 1000 --warmup-batches 1 --device cuda:0 \
        --output-dir "$NEUOPT_PAPER_ROOT/cvrp$SIZE/t$T_MAX/bs1_fullset/$CHUNK"
    done
  done
done
```

After an optional BS1 full set finishes, aggregate its ten exact chunks:

```bash
for SIZE in 50 100; do
  for T_MAX in 20 50; do
    CHUNKS=()
    for OFFSET in 0 1000 2000 3000 4000 5000 6000 7000 8000 9000; do
      printf -v CHUNK 'chunk_%05d_%05d' "$OFFSET" "$((OFFSET + 1000))"
      CHUNKS+=("$NEUOPT_PAPER_ROOT/cvrp$SIZE/t$T_MAX/bs1_fullset/$CHUNK")
    done
    python -B methods/neuopt/cvrp/paper_aggregate.py \
      --chunk-dirs "${CHUNKS[@]}" --expected-offset 0 --expected-count 10000 \
      --scope fullset \
      --output "$NEUOPT_PAPER_ROOT/cvrp$SIZE/t$T_MAX/bs1_fullset_summary.json"
  done
done
```

### NeuOpt TSP100 BS1 timing calibration

This calibration is evidence for human budget selection; it does not freeze a
paper T or run a full set. Use the same first 20 benchmark instances for every
candidate and new, non-existing output directories.

```bash
export NEUOPT_TSP100_DATASET="$ML4CO_DATA_ROOT/tsp100_concorde_7.756.pkl"
export NEUOPT_TSP100_CHECKPOINT="$NEUOPT_UPSTREAM/pre-trained/tsp100.pt"
export NEUOPT_TSP_CALIBRATION_ROOT="$BASELINE_ARTIFACT_ROOT/paper/neuopt/tsp100/calibration"

for T_MAX in 1 2 5 10 20; do
  python -B methods/neuopt/tsp/paper_eval.py \
    --problem-size 100 --T-max "$T_MAX" --batch-size 1 --d2a 1 \
    --stall-limit 10 --k 4 --device cuda:0 \
    --dataset "$NEUOPT_TSP100_DATASET" --upstream "$NEUOPT_UPSTREAM" \
    --checkpoint "$NEUOPT_TSP100_CHECKPOINT" \
    --output-dir "$NEUOPT_TSP_CALIBRATION_ROOT/t$T_MAX"
done

python -B methods/neuopt/tsp/runtime_calibration.py \
  --candidate-dirs \
    "$NEUOPT_TSP_CALIBRATION_ROOT/t1" \
    "$NEUOPT_TSP_CALIBRATION_ROOT/t2" \
    "$NEUOPT_TSP_CALIBRATION_ROOT/t5" \
    "$NEUOPT_TSP_CALIBRATION_ROOT/t10" \
    "$NEUOPT_TSP_CALIBRATION_ROOT/t20" \
  --output "$NEUOPT_TSP_CALIBRATION_ROOT/calibration_report.json"
```

### NeuOpt TSP100 frozen production

The human-frozen TSP100 budgets are `fewer: D2A=1,T=1` and
`more: D2A=1,T=5`. They are identical for BS1, BS16, and BS128. Production
artifacts use a separate root from calibration. Each preflight below runs one
true native batch; BS128 OOM fails without splitting the batch. Child output
directories must be new, except that an exact authenticated `IN_PROGRESS` run
may resume.

```bash
cd "$BASELINE_PROJECT_ROOT"
test -z "$(git status --porcelain)"
test "$(git -C "$NEUOPT_UPSTREAM" rev-parse HEAD)" = "ccf6b5f0f6a8fda2792b4be11d4ec35390a8139b"
test -z "$(git -C "$NEUOPT_UPSTREAM" status --porcelain)"
export NEUOPT_TSP100_DATASET="$ML4CO_DATA_ROOT/tsp100_concorde_7.756.pkl"
export NEUOPT_TSP100_CHECKPOINT="$NEUOPT_UPSTREAM/pre-trained/tsp100.pt"
export NEUOPT_TSP_PRODUCTION_ROOT="$BASELINE_ARTIFACT_ROOT/paper/neuopt/tsp100/production"
mkdir -p "$NEUOPT_TSP_PRODUCTION_ROOT"
```

A. TSP100 fewer, BS1 preflight:

```bash
python -B methods/neuopt/tsp/production_eval.py \
  --scope preflight --problem-size 100 --budget fewer --T-max 1 \
  --batch-size 1 --d2a 1 --stall-limit 10 --k 4 --warmup-batches 1 \
  --device cuda:0 --dataset "$NEUOPT_TSP100_DATASET" \
  --upstream "$NEUOPT_UPSTREAM" --checkpoint "$NEUOPT_TSP100_CHECKPOINT" \
  --output-dir "$NEUOPT_TSP_PRODUCTION_ROOT/preflight/fewer/bs1"
```

B. TSP100 more, BS1 preflight:

```bash
python -B methods/neuopt/tsp/production_eval.py \
  --scope preflight --problem-size 100 --budget more --T-max 5 \
  --batch-size 1 --d2a 1 --stall-limit 10 --k 4 --warmup-batches 1 \
  --device cuda:0 --dataset "$NEUOPT_TSP100_DATASET" \
  --upstream "$NEUOPT_UPSTREAM" --checkpoint "$NEUOPT_TSP100_CHECKPOINT" \
  --output-dir "$NEUOPT_TSP_PRODUCTION_ROOT/preflight/more/bs1"
```

C. TSP100 fewer, BS16 preflight:

```bash
python -B methods/neuopt/tsp/production_eval.py \
  --scope preflight --problem-size 100 --budget fewer --T-max 1 \
  --batch-size 16 --d2a 1 --stall-limit 10 --k 4 --warmup-batches 1 \
  --device cuda:0 --dataset "$NEUOPT_TSP100_DATASET" \
  --upstream "$NEUOPT_UPSTREAM" --checkpoint "$NEUOPT_TSP100_CHECKPOINT" \
  --output-dir "$NEUOPT_TSP_PRODUCTION_ROOT/preflight/fewer/bs16"
```

D. TSP100 more, BS16 preflight:

```bash
python -B methods/neuopt/tsp/production_eval.py \
  --scope preflight --problem-size 100 --budget more --T-max 5 \
  --batch-size 16 --d2a 1 --stall-limit 10 --k 4 --warmup-batches 1 \
  --device cuda:0 --dataset "$NEUOPT_TSP100_DATASET" \
  --upstream "$NEUOPT_UPSTREAM" --checkpoint "$NEUOPT_TSP100_CHECKPOINT" \
  --output-dir "$NEUOPT_TSP_PRODUCTION_ROOT/preflight/more/bs16"
```

E. TSP100 fewer, BS128 memory preflight:

```bash
python -B methods/neuopt/tsp/production_eval.py \
  --scope preflight --problem-size 100 --budget fewer --T-max 1 \
  --batch-size 128 --d2a 1 --stall-limit 10 --k 4 --warmup-batches 1 \
  --device cuda:0 --dataset "$NEUOPT_TSP100_DATASET" \
  --upstream "$NEUOPT_UPSTREAM" --checkpoint "$NEUOPT_TSP100_CHECKPOINT" \
  --output-dir "$NEUOPT_TSP_PRODUCTION_ROOT/preflight/fewer/bs128"
```

F. TSP100 more, BS128 memory preflight:

```bash
python -B methods/neuopt/tsp/production_eval.py \
  --scope preflight --problem-size 100 --budget more --T-max 5 \
  --batch-size 128 --d2a 1 --stall-limit 10 --k 4 --warmup-batches 1 \
  --device cuda:0 --dataset "$NEUOPT_TSP100_DATASET" \
  --upstream "$NEUOPT_UPSTREAM" --checkpoint "$NEUOPT_TSP100_CHECKPOINT" \
  --output-dir "$NEUOPT_TSP_PRODUCTION_ROOT/preflight/more/bs128"
```

Only after all six preflights are `PREFLIGHT_VALIDATED`, run the six exact
1280-instance full sets:

```bash
python -B methods/neuopt/tsp/production_eval.py \
  --scope fullset --problem-size 100 --budget fewer --T-max 1 \
  --batch-size 1 --d2a 1 --stall-limit 10 --k 4 --warmup-batches 1 \
  --device cuda:0 --dataset "$NEUOPT_TSP100_DATASET" \
  --upstream "$NEUOPT_UPSTREAM" --checkpoint "$NEUOPT_TSP100_CHECKPOINT" \
  --output-dir "$NEUOPT_TSP_PRODUCTION_ROOT/fullset/fewer/bs1"

python -B methods/neuopt/tsp/production_eval.py \
  --scope fullset --problem-size 100 --budget more --T-max 5 \
  --batch-size 1 --d2a 1 --stall-limit 10 --k 4 --warmup-batches 1 \
  --device cuda:0 --dataset "$NEUOPT_TSP100_DATASET" \
  --upstream "$NEUOPT_UPSTREAM" --checkpoint "$NEUOPT_TSP100_CHECKPOINT" \
  --output-dir "$NEUOPT_TSP_PRODUCTION_ROOT/fullset/more/bs1"

python -B methods/neuopt/tsp/production_eval.py \
  --scope fullset --problem-size 100 --budget fewer --T-max 1 \
  --batch-size 16 --d2a 1 --stall-limit 10 --k 4 --warmup-batches 1 \
  --device cuda:0 --dataset "$NEUOPT_TSP100_DATASET" \
  --upstream "$NEUOPT_UPSTREAM" --checkpoint "$NEUOPT_TSP100_CHECKPOINT" \
  --output-dir "$NEUOPT_TSP_PRODUCTION_ROOT/fullset/fewer/bs16"

python -B methods/neuopt/tsp/production_eval.py \
  --scope fullset --problem-size 100 --budget more --T-max 5 \
  --batch-size 16 --d2a 1 --stall-limit 10 --k 4 --warmup-batches 1 \
  --device cuda:0 --dataset "$NEUOPT_TSP100_DATASET" \
  --upstream "$NEUOPT_UPSTREAM" --checkpoint "$NEUOPT_TSP100_CHECKPOINT" \
  --output-dir "$NEUOPT_TSP_PRODUCTION_ROOT/fullset/more/bs16"

python -B methods/neuopt/tsp/production_eval.py \
  --scope fullset --problem-size 100 --budget fewer --T-max 1 \
  --batch-size 128 --d2a 1 --stall-limit 10 --k 4 --warmup-batches 1 \
  --device cuda:0 --dataset "$NEUOPT_TSP100_DATASET" \
  --upstream "$NEUOPT_UPSTREAM" --checkpoint "$NEUOPT_TSP100_CHECKPOINT" \
  --output-dir "$NEUOPT_TSP_PRODUCTION_ROOT/fullset/fewer/bs128"

python -B methods/neuopt/tsp/production_eval.py \
  --scope fullset --problem-size 100 --budget more --T-max 5 \
  --batch-size 128 --d2a 1 --stall-limit 10 --k 4 --warmup-batches 1 \
  --device cuda:0 --dataset "$NEUOPT_TSP100_DATASET" \
  --upstream "$NEUOPT_UPSTREAM" --checkpoint "$NEUOPT_TSP100_CHECKPOINT" \
  --output-dir "$NEUOPT_TSP_PRODUCTION_ROOT/fullset/more/bs128"
```

Reproduce one paper result from each full-set artifact, then read all six into
the strict Complete Results/parallel matrix:

```bash
python -B methods/neuopt/tsp/paper_aggregate.py \
  --run-dir "$NEUOPT_TSP_PRODUCTION_ROOT/fullset/fewer/bs1" \
  --output "$NEUOPT_TSP_PRODUCTION_ROOT/results/fewer_bs1.json"
python -B methods/neuopt/tsp/paper_aggregate.py \
  --run-dir "$NEUOPT_TSP_PRODUCTION_ROOT/fullset/more/bs1" \
  --output "$NEUOPT_TSP_PRODUCTION_ROOT/results/more_bs1.json"
python -B methods/neuopt/tsp/paper_aggregate.py \
  --run-dir "$NEUOPT_TSP_PRODUCTION_ROOT/fullset/fewer/bs16" \
  --output "$NEUOPT_TSP_PRODUCTION_ROOT/results/fewer_bs16.json"
python -B methods/neuopt/tsp/paper_aggregate.py \
  --run-dir "$NEUOPT_TSP_PRODUCTION_ROOT/fullset/more/bs16" \
  --output "$NEUOPT_TSP_PRODUCTION_ROOT/results/more_bs16.json"
python -B methods/neuopt/tsp/paper_aggregate.py \
  --run-dir "$NEUOPT_TSP_PRODUCTION_ROOT/fullset/fewer/bs128" \
  --output "$NEUOPT_TSP_PRODUCTION_ROOT/results/fewer_bs128.json"
python -B methods/neuopt/tsp/paper_aggregate.py \
  --run-dir "$NEUOPT_TSP_PRODUCTION_ROOT/fullset/more/bs128" \
  --output "$NEUOPT_TSP_PRODUCTION_ROOT/results/more_bs128.json"

python -B methods/neuopt/tsp/paper_results.py \
  --result-files \
    "$NEUOPT_TSP_PRODUCTION_ROOT/results/fewer_bs1.json" \
    "$NEUOPT_TSP_PRODUCTION_ROOT/results/more_bs1.json" \
    "$NEUOPT_TSP_PRODUCTION_ROOT/results/fewer_bs16.json" \
    "$NEUOPT_TSP_PRODUCTION_ROOT/results/more_bs16.json" \
    "$NEUOPT_TSP_PRODUCTION_ROOT/results/fewer_bs128.json" \
    "$NEUOPT_TSP_PRODUCTION_ROOT/results/more_bs128.json" \
  --output "$NEUOPT_TSP_PRODUCTION_ROOT/results/paper_matrix.json"
```


## 9. GLOP TSP50 and TSP100

Use a fixed project commit containing this integration. The official asset root must contain Reviser-stage2/reviser_{10,20,50,100} from the official GLOP bundle.

```bash
test "$(git -C "$BASELINE_PROJECT_ROOT" rev-parse HEAD)" = "$BASELINE_PROJECT_COMMIT"
export GLOP_UPSTREAM="$BASELINE_UPSTREAM_ROOT/GLOP"
read -r -p 'Official GLOP pretrained root: ' GLOP_ASSET_ROOT
read -r -p 'Exact TSP50 benchmark pickle: ' TSP50_DATASET
read -r -p 'Exact TSP100 benchmark pickle: ' TSP100_DATASET
export GLOP_UPSTREAM GLOP_ASSET_ROOT TSP50_DATASET TSP100_DATASET
mkdir -p "$BASELINE_ARTIFACT_ROOT/audit"

test "$(git -C "$GLOP_UPSTREAM" rev-parse HEAD)" = "e540bc0153a0598e923e35116deeaecaf9c1cfff"
test -z "$(git -C "$GLOP_UPSTREAM" status --porcelain)"
test "$(sha256sum "$TSP50_DATASET" | cut -d' ' -f1)" = "1ede2b289d2e6fbfe614219a86d1ba6dfca2a726925a8fe8cef9c8196ee0b213"
test "$(sha256sum "$TSP100_DATASET" | cut -d' ' -f1)" = "a2bfe99857b8072bdba051f6ae402b7e241f01b0462c5f379ed0aa03786406a0"
test "$(sha256sum "$GLOP_ASSET_ROOT/Reviser-stage2/reviser_10/epoch-299.pt" | cut -d' ' -f1)" = "41bd9e05d5f623a6a7978be0063354d75f6f8df62e0ed368789ca449f41922f4"
test "$(sha256sum "$GLOP_ASSET_ROOT/Reviser-stage2/reviser_10/args.json" | cut -d' ' -f1)" = "e21195ed71321b91ca2517e49b4a7556c27239ed1017d603e34d25a49742879c"
test "$(sha256sum "$GLOP_ASSET_ROOT/Reviser-stage2/reviser_20/epoch-299.pt" | cut -d' ' -f1)" = "6771bf6b955fe26004f378c1ab0a2068c3048d717325a62b85a279e0ec22a865"
test "$(sha256sum "$GLOP_ASSET_ROOT/Reviser-stage2/reviser_20/args.json" | cut -d' ' -f1)" = "66171fc5178ee7fc2b8ddcda8a7c90e804a0e9c9eb960375f82df40cbc228ef6"
test "$(sha256sum "$GLOP_ASSET_ROOT/Reviser-stage2/reviser_50/epoch-299.pt" | cut -d' ' -f1)" = "25189e74e1e0323ced3016e9c7495c2c8d8ae082961e8696dffff79f5db1d4a6"
test "$(sha256sum "$GLOP_ASSET_ROOT/Reviser-stage2/reviser_50/args.json" | cut -d' ' -f1)" = "ae311d53fe1e36573a609cc7bab75be1f346a577576c36a1d30ff799cbdcc76c"
test "$(sha256sum "$GLOP_ASSET_ROOT/Reviser-stage2/reviser_100/epoch-299.pt" | cut -d' ' -f1)" = "3810b460f210de35b5d4bd1f680f505ff4619823880652be1c7f35f320584451"
test "$(sha256sum "$GLOP_ASSET_ROOT/Reviser-stage2/reviser_100/args.json" | cut -d' ' -f1)" = "b99400a52c2dd4b6bdbd221f17432ad65d0e9d585207032ee65126b78c0354d9"

python -B -c "import importlib.metadata as m, platform, torch, numpy, scipy, tqdm, random_insertion; print({'python': platform.python_version(), 'torch': torch.__version__, 'numpy': numpy.__version__, 'scipy': scipy.__version__, 'tqdm': tqdm.__version__, 'cuda': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'random_insertion': m.version('random-insertion'), 'random_insertion_module': random_insertion.__file__})" \
  | tee "$BASELINE_ARTIFACT_ROOT/audit/glop_dependency_environment.txt"

python -B methods/glop/tsp/prepare_instances.py \
  --dataset "$TSP50_DATASET" --problem-size 50 --offset 0 --count 5 \
  --output "$BASELINE_ARTIFACT_ROOT/glop_tsp50/input_first5.npz"
python -B methods/glop/tsp/run.py --problem-size 50 \
  --input "$BASELINE_ARTIFACT_ROOT/glop_tsp50/input_first5.npz" \
  --upstream "$GLOP_UPSTREAM" --asset-root "$GLOP_ASSET_ROOT" \
  --output "$BASELINE_ARTIFACT_ROOT/glop_tsp50/formal_first5.json" --device cuda:0
python -B methods/glop/tsp/validate_with_kit.py --problem-size 50 \
  --input "$BASELINE_ARTIFACT_ROOT/glop_tsp50/formal_first5.json" \
  --dataset "$TSP50_DATASET" \
  --output "$BASELINE_ARTIFACT_ROOT/glop_tsp50/formal_first5_validated.json"

python -B methods/glop/tsp/prepare_instances.py \
  --dataset "$TSP100_DATASET" --problem-size 100 --offset 0 --count 5 \
  --output "$BASELINE_ARTIFACT_ROOT/glop_tsp100/input_first5.npz"
python -B methods/glop/tsp/run.py --problem-size 100 \
  --input "$BASELINE_ARTIFACT_ROOT/glop_tsp100/input_first5.npz" \
  --upstream "$GLOP_UPSTREAM" --asset-root "$GLOP_ASSET_ROOT" \
  --output "$BASELINE_ARTIFACT_ROOT/glop_tsp100/formal_first5.json" --device cuda:0
python -B methods/glop/tsp/validate_with_kit.py --problem-size 100 \
  --input "$BASELINE_ARTIFACT_ROOT/glop_tsp100/formal_first5.json" \
  --dataset "$TSP100_DATASET" \
  --output "$BASELINE_ARTIFACT_ROOT/glop_tsp100/formal_first5_validated.json"

test "$(git -C "$BASELINE_PROJECT_ROOT" rev-parse HEAD)" = "$BASELINE_PROJECT_COMMIT"
test -z "$(git -C "$GLOP_UPSTREAM" status --porcelain)"
sha256sum "$BASELINE_ARTIFACT_ROOT/glop_tsp50/formal_first5_validated.json" \
  "$BASELINE_ARTIFACT_ROOT/glop_tsp100/formal_first5_validated.json"
```

Every result row must have all four completion gates true. Return both validated artifacts, the dependency/environment audit, both artifact SHA256 lines, and both clean-state outputs.

## 10. GLOP formal paper preflights

Use exact paper-target dataset paths. These commands create count-two evidence
under a decoder-fixed/new-protocol root and never append to the historical ff39
artifact directories. Formal TSP production remains one offset-zero full-set
chunk; formal CVRP remains one offset-zero sequential stream with completed-prefix
replay.

```bash
export GLOP_UPSTREAM="$BASELINE_UPSTREAM_ROOT/GLOP"
read -r -p 'Official GLOP pretrained root: ' GLOP_ASSET_ROOT
read -r -p 'tsp100_concorde_7.756.pkl path: ' GLOP_TSP100_DATASET
read -r -p 'tsp500_concorde_16.546.pkl path: ' GLOP_TSP500_DATASET
read -r -p 'tsp1000_concorde_23.118.pkl path: ' GLOP_TSP1K_DATASET
read -r -p 'tsp2000_lkh_500_32.436.pkl path: ' GLOP_TSP2K_DATASET
read -r -p 'tsp5000_lkh_500_50.968.pkl path: ' GLOP_TSP5K_DATASET
read -r -p 'tsp10000_lkh_500_71.782.pkl path: ' GLOP_TSP10K_DATASET
read -r -p 'cvrp500_hgs-300s_37.154.pkl path: ' GLOP_CVRP500_DATASET
read -r -p 'cvrp1000_hgs-360s_41.171.pkl path: ' GLOP_CVRP1K_DATASET
read -r -p 'cvrp2000_hgs-360s_57.181.pkl path: ' GLOP_CVRP2K_DATASET
export GLOP_ASSET_ROOT
export GLOP_PROTOCOL_ROOT="$BASELINE_ARTIFACT_ROOT/paper/glop/decoder_fixed_new_protocols"

test "$(basename "$GLOP_TSP100_DATASET")" = "tsp100_concorde_7.756.pkl"
test "$(basename "$GLOP_TSP500_DATASET")" = "tsp500_concorde_16.546.pkl"
test "$(basename "$GLOP_TSP1K_DATASET")" = "tsp1000_concorde_23.118.pkl"
test "$(basename "$GLOP_TSP2K_DATASET")" = "tsp2000_lkh_500_32.436.pkl"
test "$(basename "$GLOP_TSP5K_DATASET")" = "tsp5000_lkh_500_50.968.pkl"
test "$(basename "$GLOP_TSP10K_DATASET")" = "tsp10000_lkh_500_71.782.pkl"
test "$(basename "$GLOP_CVRP500_DATASET")" = "cvrp500_hgs-300s_37.154.pkl"
test "$(basename "$GLOP_CVRP1K_DATASET")" = "cvrp1000_hgs-360s_41.171.pkl"
test "$(basename "$GLOP_CVRP2K_DATASET")" = "cvrp2000_hgs-360s_57.181.pkl"

python -B scripts/audit_glop_paper_datasets.py \
  --dataset TSP 100 "$GLOP_TSP100_DATASET" \
  --dataset TSP 500 "$GLOP_TSP500_DATASET" \
  --dataset TSP 1000 "$GLOP_TSP1K_DATASET" \
  --dataset TSP 2000 "$GLOP_TSP2K_DATASET" \
  --dataset TSP 5000 "$GLOP_TSP5K_DATASET" \
  --dataset TSP 10000 "$GLOP_TSP10K_DATASET" \
  --dataset CVRP 500 "$GLOP_CVRP500_DATASET" \
  --dataset CVRP 1000 "$GLOP_CVRP1K_DATASET" \
  --dataset CVRP 2000 "$GLOP_CVRP2K_DATASET" \
  --output "$GLOP_PROTOCOL_ROOT/audit/glop_paper_datasets.json"
```

Prepare and run the new TSP100/2K/5K protocols:

```bash
for size in 100 2000 5000; do
  case "$size" in
    100) dataset="$GLOP_TSP100_DATASET" ;;
    2000) dataset="$GLOP_TSP2K_DATASET" ;;
    5000) dataset="$GLOP_TSP5K_DATASET" ;;
  esac
  for protocol in official_standard official_more; do
    run_root="$GLOP_PROTOCOL_ROOT/tsp$size/$protocol/count2"
    python -B methods/glop/tsp/prepare_instances.py \
      --dataset "$dataset" --problem-size "$size" --protocol "$protocol" \
      --offset 0 --count 2 --output "$run_root/input.npz"
    python -B methods/glop/tsp/paper_eval.py \
      --input "$run_root/input.npz" --problem-size "$size" \
      --protocol "$protocol" --upstream "$GLOP_UPSTREAM" \
      --asset-root "$GLOP_ASSET_ROOT" --output-dir "$run_root/inference" \
      --device cuda:0
    python -B methods/glop/validate_with_kit.py \
      --dataset "$dataset" --chunk-dirs "$run_root/inference"
  done
done
```

Prepare and run decoder-fixed CVRP500/1K/2K count-two protocols from index
zero. Both budgets get separate provenance and output directories:

```bash
for size in 500 1000 2000; do
  case "$size" in
    500) dataset="$GLOP_CVRP500_DATASET" ;;
    1000) dataset="$GLOP_CVRP1K_DATASET" ;;
    2000) dataset="$GLOP_CVRP2K_DATASET" ;;
  esac
  for protocol in official_standard project_more_revisions; do
    run_root="$GLOP_PROTOCOL_ROOT/cvrp$size/$protocol/count2"
    python -B methods/glop/cvrp/prepare_instances.py \
      --dataset "$dataset" --problem-size "$size" --protocol "$protocol" \
      --offset 0 --count 2 --output "$run_root/input.npz"
    python -B methods/glop/cvrp/paper_eval.py \
      --input "$run_root/input.npz" --problem-size "$size" \
      --protocol "$protocol" --upstream "$GLOP_UPSTREAM" \
      --asset-root "$GLOP_ASSET_ROOT" --output-dir "$run_root/inference" \
      --device cuda:0
    python -B methods/glop/validate_with_kit.py \
      --dataset "$dataset" --chunk-dirs "$run_root/inference"
  done
done
```

Existing ff39 TSP500/1K/10K and CVRP evidence is immutable control evidence.
Do not overwrite, append, or resummarize it. Before any new full-set run, return
the count-two metadata, inference records, Kit-validated records, and hashes.
TSP100/2K/5K and CVRP500 also require an independent official-shadow comparison.
For CVRP `project_more_revisions`, shadow equivalence means matching pinned
official primitives under the project-defined iteration budget; it does not
claim that this budget was released by GLOP.

## UDC Stage S3: ML4CO TSP500/CVRP500 semantic adapter

Run `our_2` first. The runner discovers exactly one pinned TSP500 and CVRP500
filename below the dataset root and refuses ambiguous copies.

```bash
conda activate cp311_base
cd /inspire/hdd/global_user/majiale-253108540229/zhang/neural-routing-baselines
S3_HEAD="$(git rev-parse --short=8 HEAD)"
python -B methods/udc/s3_eval.py \
  --project-root "$PWD" \
  --official-root /inspire/hdd/global_user/majiale-253108540229/zhang/baselines/NCO_code \
  --supplemental-root /inspire/hdd/global_user/majiale-253108540229/zhang/baselines/UDC_supplemental_unpack/UDC-master/UDC \
  --dataset-root /inspire/hdd/global_user/majiale-253108540229/ML4CO-Bench-101 \
  --s1-evidence /inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/udc/15d02f5b/s1_audit_v5.json \
  --s2-evidence /inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/udc/15d02f5b/s2_official_smoke_v3 \
  --count 2 \
  --output-dir "/inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/udc/$S3_HEAD/s3_ml4co_adapter/our_2"
```

Only after `our_2/metadata.json` reports `state=KIT_VALIDATED`, run the same
protocol on the first five tasks:

```bash
S3_HEAD="$(git rev-parse --short=8 HEAD)"
python -B methods/udc/s3_eval.py \
  --project-root "$PWD" \
  --official-root /inspire/hdd/global_user/majiale-253108540229/zhang/baselines/NCO_code \
  --supplemental-root /inspire/hdd/global_user/majiale-253108540229/zhang/baselines/UDC_supplemental_unpack/UDC-master/UDC \
  --dataset-root /inspire/hdd/global_user/majiale-253108540229/ML4CO-Bench-101 \
  --s1-evidence /inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/udc/15d02f5b/s1_audit_v5.json \
  --s2-evidence /inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/udc/15d02f5b/s2_official_smoke_v3 \
  --count 5 \
  --prior-our2 "/inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/udc/$S3_HEAD/s3_ml4co_adapter/our_2" \
  --output-dir "/inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/udc/$S3_HEAD/s3_ml4co_adapter/our_5"
```

## UDC Stage S4: one-size formal scale preflight

Run one process per size in strict ascending order. Each command runs dataset
index zero only. A family stops after CUDA OOM or semantic failure.

```bash
conda activate cp311_base
cd /inspire/hdd/global_user/majiale-253108540229/zhang/neural-routing-baselines
S4_HEAD="$(git rev-parse --short=8 HEAD)"
S4_ROOT="/inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/udc/$S4_HEAD/s4_scale_preflight"

python -B methods/udc/s4_scale_preflight.py \
  --problem tsp --size 100 \
  --project-root "$PWD" \
  --official-root /inspire/hdd/global_user/majiale-253108540229/zhang/baselines/NCO_code \
  --supplemental-root /inspire/hdd/global_user/majiale-253108540229/zhang/baselines/UDC_supplemental_unpack/UDC-master/UDC \
  --dataset-root /inspire/hdd/global_user/majiale-253108540229/ML4CO-Bench-101 \
  --s3-evidence /inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/udc/df082675/s3_ml4co_adapter/our_5 \
  --output-root "$S4_ROOT"
```

After each `PASS`, repeat the command by changing only `--problem` and `--size`:

- TSP: `100`, `500`, `1000`, `2000`, `5000`, `10000`
- CVRP: `200`, `500`, `1000`, `2000`

Every invocation updates `$S4_ROOT/summary.json` and writes immutable
`metadata.json` and `record.json` under its problem-size directory.

## UDC final paper production

Use the committed production implementation only after the project checkout and
pinned official checkout are clean. The output root is outside both repositories.
Do not reuse S3/S4 runtime as paper Time.

```bash
conda activate cp311_base
cd /inspire/hdd/global_user/majiale-253108540229/zhang/neural-routing-baselines
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PRODUCTION_HEAD="$(git rev-parse --short=8 HEAD)"
UDC_PRODUCTION_ROOT="/inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/udc/$PRODUCTION_HEAD/paper_production"
```

Set this exact allocator configuration before starting every pilot, safety-gate,
or production Python process. Do not add any other allocator knob. Aggregation
does not require CUDA or this environment variable.

Run the four fixed pilot processes. Each process independently seeds once and
then processes dataset indices 0, 1, and 2 continuously. The fourth completed
run creates `budget_freeze/pilot.json` and `budget_freeze/decision.json`.

```bash
python -B methods/udc/paper_production.py --mode pilot --problem tsp --budget fewer \
  --project-root "$PWD" --official-root /inspire/hdd/global_user/majiale-253108540229/zhang/baselines/NCO_code \
  --supplemental-root /inspire/hdd/global_user/majiale-253108540229/zhang/baselines/UDC_supplemental_unpack/UDC-master/UDC \
  --dataset-root /inspire/hdd/global_user/majiale-253108540229/ML4CO-Bench-101 \
  --s3-evidence /inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/udc/df082675/s3_ml4co_adapter/our_5 \
  --s4-evidence /inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/udc/99b85d29/s4_scale_preflight \
  --output-root "$UDC_PRODUCTION_ROOT"

python -B methods/udc/paper_production.py --mode pilot --problem tsp --budget more \
  --project-root "$PWD" --official-root /inspire/hdd/global_user/majiale-253108540229/zhang/baselines/NCO_code \
  --supplemental-root /inspire/hdd/global_user/majiale-253108540229/zhang/baselines/UDC_supplemental_unpack/UDC-master/UDC \
  --dataset-root /inspire/hdd/global_user/majiale-253108540229/ML4CO-Bench-101 \
  --s3-evidence /inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/udc/df082675/s3_ml4co_adapter/our_5 \
  --s4-evidence /inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/udc/99b85d29/s4_scale_preflight \
  --output-root "$UDC_PRODUCTION_ROOT"

python -B methods/udc/paper_production.py --mode pilot --problem cvrp --budget fewer \
  --project-root "$PWD" --official-root /inspire/hdd/global_user/majiale-253108540229/zhang/baselines/NCO_code \
  --supplemental-root /inspire/hdd/global_user/majiale-253108540229/zhang/baselines/UDC_supplemental_unpack/UDC-master/UDC \
  --dataset-root /inspire/hdd/global_user/majiale-253108540229/ML4CO-Bench-101 \
  --s3-evidence /inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/udc/df082675/s3_ml4co_adapter/our_5 \
  --s4-evidence /inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/udc/99b85d29/s4_scale_preflight \
  --output-root "$UDC_PRODUCTION_ROOT"

python -B methods/udc/paper_production.py --mode pilot --problem cvrp --budget more \
  --project-root "$PWD" --official-root /inspire/hdd/global_user/majiale-253108540229/zhang/baselines/NCO_code \
  --supplemental-root /inspire/hdd/global_user/majiale-253108540229/zhang/baselines/UDC_supplemental_unpack/UDC-master/UDC \
  --dataset-root /inspire/hdd/global_user/majiale-253108540229/ML4CO-Bench-101 \
  --s3-evidence /inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/udc/df082675/s3_ml4co_adapter/our_5 \
  --s4-evidence /inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/udc/99b85d29/s4_scale_preflight \
  --output-root "$UDC_PRODUCTION_ROOT"
```

After `decision.json` reports `FROZEN`, run both largest-size more safety gates:

```bash
python -B methods/udc/paper_production.py --mode safety-gate --problem tsp --size 10000 --budget more \
  --project-root "$PWD" --official-root /inspire/hdd/global_user/majiale-253108540229/zhang/baselines/NCO_code \
  --supplemental-root /inspire/hdd/global_user/majiale-253108540229/zhang/baselines/UDC_supplemental_unpack/UDC-master/UDC \
  --dataset-root /inspire/hdd/global_user/majiale-253108540229/ML4CO-Bench-101 \
  --s3-evidence /inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/udc/df082675/s3_ml4co_adapter/our_5 \
  --s4-evidence /inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/udc/99b85d29/s4_scale_preflight \
  --output-root "$UDC_PRODUCTION_ROOT"

python -B methods/udc/paper_production.py --mode safety-gate --problem cvrp --size 2000 --budget more \
  --project-root "$PWD" --official-root /inspire/hdd/global_user/majiale-253108540229/zhang/baselines/NCO_code \
  --supplemental-root /inspire/hdd/global_user/majiale-253108540229/zhang/baselines/UDC_supplemental_unpack/UDC-master/UDC \
  --dataset-root /inspire/hdd/global_user/majiale-253108540229/ML4CO-Bench-101 \
  --s3-evidence /inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/udc/df082675/s3_ml4co_adapter/our_5 \
  --s4-evidence /inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/udc/99b85d29/s4_scale_preflight \
  --output-root "$UDC_PRODUCTION_ROOT"
```

Run one formal fullset at a time. Re-running the exact command resumes from the
serialized RNG checkpoint; a completed run returns its existing summary.

```bash
python -B methods/udc/paper_production.py --mode production --problem tsp --size 1000 --budget fewer \
  --project-root "$PWD" --official-root /inspire/hdd/global_user/majiale-253108540229/zhang/baselines/NCO_code \
  --supplemental-root /inspire/hdd/global_user/majiale-253108540229/zhang/baselines/UDC_supplemental_unpack/UDC-master/UDC \
  --dataset-root /inspire/hdd/global_user/majiale-253108540229/ML4CO-Bench-101 \
  --s3-evidence /inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/udc/df082675/s3_ml4co_adapter/our_5 \
  --s4-evidence /inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/udc/99b85d29/s4_scale_preflight \
  --output-root "$UDC_PRODUCTION_ROOT"
```

Change only problem, size, and budget to cover the 20 frozen cells. After all
20 summaries are `KIT_VALIDATED`, build the paper table without invoking a solver:

```bash
python -B methods/udc/paper_production.py --mode aggregate \
  --output-root "$UDC_PRODUCTION_ROOT"
```

## 11. Remaining CVRPTW Baselines

This section covers only RF-TE, CaDA, and MoSES(CaDA), for CVRPTW50/100. It
never installs into `cp311_base`. Set each Python executable to an already
audited compatible environment; if the import/load audit fails, create and
freeze a separate method environment before continuing. The paths below do not
fall back to another dataset, checkpoint, or checkout.

```bash
export BASELINE_PROJECT_ROOT=/inspire/hdd/global_user/majiale-253108540229/zhang/neural-routing-baselines
export REMAINING_DATASET_ROOT=/inspire/hdd/global_user/majiale-253108540229/zhang/datasets/ML4CO-Bench-101-SL
export REMAINING_ARTIFACT_ROOT=/inspire/hdd/global_user/majiale-253108540229/zhang/artifacts/neural-routing-baselines/remaining_cvrptw
export REMAINING_AUDIT="$REMAINING_ARTIFACT_ROOT/audit/audit.json"
export ML4CO_PYTHON=/absolute/path/to/python-with-ml4co-kit
export RFTE_PYTHON=/absolute/path/to/audited-rfte-python
export CADA_PYTHON=/absolute/path/to/audited-cada-python
export MOSES_CADA_PYTHON=/absolute/path/to/audited-moses-cada-python
cd "$BASELINE_PROJECT_ROOT"
test -z "$(git status --porcelain)"
mkdir -p "$REMAINING_ARTIFACT_ROOT/audit"
```

Step 1, audit both datasets, all three clean pinned upstreams, both checkpoint
paths per method, hashes, and the invoking environment. This command writes the
JSON even when one method fails, so other methods remain auditable.

```bash
"$ML4CO_PYTHON" -B scripts/audit_remaining_cvrptw.py \
  --upstream-root "$BASELINE_PROJECT_ROOT/external" \
  --checkpoint-root "$BASELINE_PROJECT_ROOT/external" \
  --dataset-root "$REMAINING_DATASET_ROOT" \
  --output "$REMAINING_AUDIT"
```

Inspect the JSON before continuing. RF-TE must resolve to
`rf-transformer.ckpt`; CaDA must resolve to its size-specific
`checkpoint-300.pt`; MoSES must resolve only to the CaDA
`multilora_denseroute_sigmoid.ckpt`. A missing asset remains a method-local
blocker. Do not substitute an RF/MoE/POMO, softmax, softplus, or VR1route file.

Run CPU import/construction/audited-load checks with each selected method
Python before any GPU smoke. Repeat both sizes; a failure is local to that
method and must not suppress the other commands.

```bash
export RFTE50_SHA=$("$ML4CO_PYTHON" -c 'import json,os;print(json.load(open(os.environ["REMAINING_AUDIT"]))["methods"]["rfte"]["checkpoints"]["50"]["sha256"])')
export RFTE100_SHA=$("$ML4CO_PYTHON" -c 'import json,os;print(json.load(open(os.environ["REMAINING_AUDIT"]))["methods"]["rfte"]["checkpoints"]["100"]["sha256"])')
export CADA50_SHA=$("$ML4CO_PYTHON" -c 'import json,os;print(json.load(open(os.environ["REMAINING_AUDIT"]))["methods"]["cada"]["checkpoints"]["50"]["sha256"])')
export CADA100_SHA=$("$ML4CO_PYTHON" -c 'import json,os;print(json.load(open(os.environ["REMAINING_AUDIT"]))["methods"]["cada"]["checkpoints"]["100"]["sha256"])')
export MOSES50_SHA=1aa499f3fce5d3412c2544c9632bbb9709309a7299b4298b535fa7e9014ef803
export MOSES100_SHA=2eac9b038ae4655581aa73e4dbe8ad529aefd1963368c9a92d254b6269f8aabf

"$RFTE_PYTHON" -B scripts/audit_remaining_cvrptw_environment.py --method rfte --problem-size 50 --upstream "$BASELINE_PROJECT_ROOT/external/routefinder" --checkpoint "$BASELINE_PROJECT_ROOT/external/routefinder/checkpoints/50/rf-transformer.ckpt" --expected-checkpoint-sha256 "$RFTE50_SHA" --output "$REMAINING_ARTIFACT_ROOT/audit/rfte50_environment.json"
"$RFTE_PYTHON" -B scripts/audit_remaining_cvrptw_environment.py --method rfte --problem-size 100 --upstream "$BASELINE_PROJECT_ROOT/external/routefinder" --checkpoint "$BASELINE_PROJECT_ROOT/external/routefinder/checkpoints/100/rf-transformer.ckpt" --expected-checkpoint-sha256 "$RFTE100_SHA" --output "$REMAINING_ARTIFACT_ROOT/audit/rfte100_environment.json"
"$CADA_PYTHON" -B scripts/audit_remaining_cvrptw_environment.py --method cada --problem-size 50 --upstream "$BASELINE_PROJECT_ROOT/external/CaDA" --checkpoint "$BASELINE_PROJECT_ROOT/external/CaDA/50/result/2024-1111-1139/checkpoint-300.pt" --expected-checkpoint-sha256 "$CADA50_SHA" --output "$REMAINING_ARTIFACT_ROOT/audit/cada50_environment.json"
"$CADA_PYTHON" -B scripts/audit_remaining_cvrptw_environment.py --method cada --problem-size 100 --upstream "$BASELINE_PROJECT_ROOT/external/CaDA" --checkpoint "$BASELINE_PROJECT_ROOT/external/CaDA/100/result/2024-1121-1355/checkpoint-300.pt" --expected-checkpoint-sha256 "$CADA100_SHA" --output "$REMAINING_ARTIFACT_ROOT/audit/cada100_environment.json"
"$MOSES_CADA_PYTHON" -B scripts/audit_remaining_cvrptw_environment.py --method moses_cada --problem-size 50 --upstream "$BASELINE_PROJECT_ROOT/external/moses_vrp" --checkpoint "$BASELINE_PROJECT_ROOT/external/moses_vrp/pretrained_moses_model/cada/50/multilora_denseroute_sigmoid.ckpt" --expected-checkpoint-sha256 "$MOSES50_SHA" --output "$REMAINING_ARTIFACT_ROOT/audit/moses_cada50_environment.json"
"$MOSES_CADA_PYTHON" -B scripts/audit_remaining_cvrptw_environment.py --method moses_cada --problem-size 100 --upstream "$BASELINE_PROJECT_ROOT/external/moses_vrp" --checkpoint "$BASELINE_PROJECT_ROOT/external/moses_vrp/pretrained_moses_model/cada/100/multilora_denseroute_sigmoid.ckpt" --expected-checkpoint-sha256 "$MOSES100_SHA" --output "$REMAINING_ARTIFACT_ROOT/audit/moses_cada100_environment.json"
```

For both MoSES reports, require `status=PASS`, empty checkpoint
`missing_keys`/`unexpected_keys`, and an effective runtime protocol with policy
decode `greedy`, all-customer multistart, and the pinned CaDA multi-LoRA
settings. Checkpoint decode hyperparameters are diagnostic metadata; the formal
multistart behavior comes from pinned `test.py` passing `num_starts=N`.

Step 2, prepare the exact shared first-1, first-2, first-5, and full-set inputs.
Preparation verifies filename, SHA256, task class/schema, capacity, and exact
dataset count before writing original units.

```bash
for size in 50 100; do
  dataset="$REMAINING_DATASET_ROOT/cvrptw${size}_pyvrp-$([ "$size" = 50 ] && echo '10s_16.038' || echo '20s_25.431').pkl"
  mkdir -p "$REMAINING_ARTIFACT_ROOT/prepared/cvrptw${size}"
  "$ML4CO_PYTHON" -B scripts/prepare_remaining_cvrptw.py --dataset "$dataset" \
    --problem-size "$size" --offset 0 --count 1 \
    --output "$REMAINING_ARTIFACT_ROOT/prepared/cvrptw${size}/preflight.npz"
  "$ML4CO_PYTHON" -B scripts/prepare_remaining_cvrptw.py --dataset "$dataset" \
    --problem-size "$size" --offset 0 --count 2 \
    --output "$REMAINING_ARTIFACT_ROOT/prepared/cvrptw${size}/our_2.npz"
  "$ML4CO_PYTHON" -B scripts/prepare_remaining_cvrptw.py --dataset "$dataset" \
    --problem-size "$size" --offset 0 --count 5 \
    --output "$REMAINING_ARTIFACT_ROOT/prepared/cvrptw${size}/our_5.npz"
  "$ML4CO_PYTHON" -B scripts/prepare_remaining_cvrptw.py --dataset "$dataset" \
    --problem-size "$size" --offset 0 --count 1000 \
    --output "$REMAINING_ARTIFACT_ROOT/prepared/cvrptw${size}/production.npz"
done
```

Step 3, execute all six one-instance preflights. A failed RF-TE job does not
hide CaDA/MoSES results, but each failed method remains blocked from production.

```bash
"$ML4CO_PYTHON" -B scripts/run_remaining_cvrptw_preflights.py \
  --project-root "$BASELINE_PROJECT_ROOT" --audit-evidence "$REMAINING_AUDIT" \
  --prepared-root "$REMAINING_ARTIFACT_ROOT/prepared" \
  --dataset-root "$REMAINING_DATASET_ROOT" \
  --output-root "$REMAINING_ARTIFACT_ROOT/results" \
  --rfte-python "$RFTE_PYTHON" --cada-python "$CADA_PYTHON" \
  --moses-cada-python "$MOSES_CADA_PYTHON"
```

Step 4, run `our_2`, then Step 5, run `our_5`. The helper below is only shell
composition: every Python evaluator still enforces the exact method, size,
scope, upstream, checkpoint, dataset, GPU, and clean-tree gates.

```bash
run_remaining_cvrptw_scope () {
  method="$1"; size="$2"; scope="$3"; python_exe="$4"; upstream="$5"; checkpoint="$6"; checkpoint_sha="$7"
  "$python_exe" -B "methods/$method/cvrptw/paper_eval.py" \
    --scope "$scope" --problem-size "$size" \
    --input "$REMAINING_ARTIFACT_ROOT/prepared/cvrptw${size}/${scope}.npz" \
    --dataset "$REMAINING_DATASET_ROOT/cvrptw${size}_pyvrp-$([ "$size" = 50 ] && echo '10s_16.038' || echo '20s_25.431').pkl" \
    --upstream "$upstream" --checkpoint "$checkpoint" \
    --expected-checkpoint-sha256 "$checkpoint_sha" \
    --output-dir "$REMAINING_ARTIFACT_ROOT/results/$method/cvrptw${size}/${scope}" \
    $([ "$scope" = our_5 ] && printf '%s %s' --our2-evidence "$REMAINING_ARTIFACT_ROOT/results/$method/cvrptw${size}/our_2") \
    --warmup-instances 2 --device cuda:0
}

for scope in our_2 our_5; do
  run_remaining_cvrptw_scope rfte 50 "$scope" "$RFTE_PYTHON" "$BASELINE_PROJECT_ROOT/external/routefinder" "$BASELINE_PROJECT_ROOT/external/routefinder/checkpoints/50/rf-transformer.ckpt" "$RFTE50_SHA"
  run_remaining_cvrptw_scope rfte 100 "$scope" "$RFTE_PYTHON" "$BASELINE_PROJECT_ROOT/external/routefinder" "$BASELINE_PROJECT_ROOT/external/routefinder/checkpoints/100/rf-transformer.ckpt" "$RFTE100_SHA"
  run_remaining_cvrptw_scope cada 50 "$scope" "$CADA_PYTHON" "$BASELINE_PROJECT_ROOT/external/CaDA" "$BASELINE_PROJECT_ROOT/external/CaDA/50/result/2024-1111-1139/checkpoint-300.pt" "$CADA50_SHA"
  run_remaining_cvrptw_scope cada 100 "$scope" "$CADA_PYTHON" "$BASELINE_PROJECT_ROOT/external/CaDA" "$BASELINE_PROJECT_ROOT/external/CaDA/100/result/2024-1121-1355/checkpoint-300.pt" "$CADA100_SHA"
  run_remaining_cvrptw_scope moses_cada 50 "$scope" "$MOSES_CADA_PYTHON" "$BASELINE_PROJECT_ROOT/external/moses_vrp" "$BASELINE_PROJECT_ROOT/external/moses_vrp/pretrained_moses_model/cada/50/multilora_denseroute_sigmoid.ckpt" "$MOSES50_SHA"
  run_remaining_cvrptw_scope moses_cada 100 "$scope" "$MOSES_CADA_PYTHON" "$BASELINE_PROJECT_ROOT/external/moses_vrp" "$BASELINE_PROJECT_ROOT/external/moses_vrp/pretrained_moses_model/cada/100/multilora_denseroute_sigmoid.ckpt" "$MOSES100_SHA"
done
```

Verify all six `our_5/summary.json` files say `KIT_VALIDATED`. Do not run a
fullset before that. Step 6 is the production launcher template. It is dry-run
unless the final argument is `--execute`, and it revalidates the selected
method's hashed `our_5` evidence before invoking the solver. Run all six
invocations without `--execute` first. Review the printed commands, then repeat
one invocation at a time with `--execute`.

```bash
launch_remaining_cvrptw () {
  method="$1"; size="$2"; python_exe="$3"; upstream="$4"; checkpoint="$5"; checkpoint_sha="$6"; execute_flag="${7:-}"
  "$ML4CO_PYTHON" -B scripts/launch_remaining_cvrptw_production.py \
    --method "$method" --problem-size "$size" --python "$python_exe" \
    --input "$REMAINING_ARTIFACT_ROOT/prepared/cvrptw${size}/production.npz" \
    --dataset "$REMAINING_DATASET_ROOT/cvrptw${size}_pyvrp-$([ "$size" = 50 ] && echo '10s_16.038' || echo '20s_25.431').pkl" \
    --upstream "$upstream" --checkpoint "$checkpoint" \
    --expected-checkpoint-sha256 "$checkpoint_sha" \
    --our5-evidence "$REMAINING_ARTIFACT_ROOT/results/$method/cvrptw${size}/our_5" \
    --output-dir "$REMAINING_ARTIFACT_ROOT/results/$method/cvrptw${size}/production" \
    $execute_flag
}

launch_remaining_cvrptw rfte 50 "$RFTE_PYTHON" "$BASELINE_PROJECT_ROOT/external/routefinder" "$BASELINE_PROJECT_ROOT/external/routefinder/checkpoints/50/rf-transformer.ckpt" "$RFTE50_SHA"
launch_remaining_cvrptw rfte 100 "$RFTE_PYTHON" "$BASELINE_PROJECT_ROOT/external/routefinder" "$BASELINE_PROJECT_ROOT/external/routefinder/checkpoints/100/rf-transformer.ckpt" "$RFTE100_SHA"
launch_remaining_cvrptw cada 50 "$CADA_PYTHON" "$BASELINE_PROJECT_ROOT/external/CaDA" "$BASELINE_PROJECT_ROOT/external/CaDA/50/result/2024-1111-1139/checkpoint-300.pt" "$CADA50_SHA"
launch_remaining_cvrptw cada 100 "$CADA_PYTHON" "$BASELINE_PROJECT_ROOT/external/CaDA" "$BASELINE_PROJECT_ROOT/external/CaDA/100/result/2024-1121-1355/checkpoint-300.pt" "$CADA100_SHA"
launch_remaining_cvrptw moses_cada 50 "$MOSES_CADA_PYTHON" "$BASELINE_PROJECT_ROOT/external/moses_vrp" "$BASELINE_PROJECT_ROOT/external/moses_vrp/pretrained_moses_model/cada/50/multilora_denseroute_sigmoid.ckpt" "$MOSES50_SHA"
launch_remaining_cvrptw moses_cada 100 "$MOSES_CADA_PYTHON" "$BASELINE_PROJECT_ROOT/external/moses_vrp" "$BASELINE_PROJECT_ROOT/external/moses_vrp/pretrained_moses_model/cada/100/multilora_denseroute_sigmoid.ckpt" "$MOSES100_SHA"
```

Each production run is resumable only with an identical provenance fingerprint
and produces `metadata.json`, `checkpoint_state.json`,
`validated_records.jsonl`, `batch_timings.jsonl`, and `summary.json`.

Step 7, after all six production summaries are `PAPER_READY`, aggregate without
invoking any model:

```bash
"$ML4CO_PYTHON" -B scripts/summarize_remaining_cvrptw.py \
  --rfte-50 "$REMAINING_ARTIFACT_ROOT/results/rfte/cvrptw50/production" \
  --rfte-100 "$REMAINING_ARTIFACT_ROOT/results/rfte/cvrptw100/production" \
  --cada-50 "$REMAINING_ARTIFACT_ROOT/results/cada/cvrptw50/production" \
  --cada-100 "$REMAINING_ARTIFACT_ROOT/results/cada/cvrptw100/production" \
  --moses-cada-50 "$REMAINING_ARTIFACT_ROOT/results/moses_cada/cvrptw50/production" \
  --moses-cada-100 "$REMAINING_ARTIFACT_ROOT/results/moses_cada/cvrptw100/production" \
  --output "$REMAINING_ARTIFACT_ROOT/six_cell_summary.json"
```

CVRPTW200 and every CVRP cell are outside this phase. The production commands
must remain unexecuted until all six `our_5` gates pass.

### RF-TE and MoSES(CaDA) native BS1/BS10 formal runs

The batch-aware flow below supersedes the legacy RF-TE/MoSES commands above.
`--batch-size` is the original-instance batch size. BS10 preflight, small gate,
and validation gate therefore contain 10, 20, and 50 instances. The production
input remains the same instance-major 1000-row NPZ. CaDA is outside this BS10
phase.

```bash
for size in 50 100; do
  dataset="$REMAINING_DATASET_ROOT/cvrptw${size}_pyvrp-$([ "$size" = 50 ] && echo '10s_16.038' || echo '20s_25.431').pkl"
  for bs in 1 10; do
    for spec in "preflight:$bs" "small_gate:$((2*bs))" "validation_gate:$((5*bs))"; do
      scope=${spec%%:*}; count=${spec##*:}
      "$ML4CO_PYTHON" -B scripts/prepare_remaining_cvrptw.py \
        --dataset "$dataset" --problem-size "$size" --offset 0 --count "$count" \
        --output "$REMAINING_ARTIFACT_ROOT/prepared/cvrptw${size}/${scope}_bs${bs}.npz"
    done
  done
done

for bs in 1 10; do
  "$ML4CO_PYTHON" -B scripts/run_remaining_cvrptw_preflights.py \
    --project-root "$BASELINE_PROJECT_ROOT" --audit-evidence "$REMAINING_AUDIT" \
    --prepared-root "$REMAINING_ARTIFACT_ROOT/prepared" \
    --dataset-root "$REMAINING_DATASET_ROOT" \
    --output-root "$REMAINING_ARTIFACT_ROOT/results" \
    --rfte-python "$RFTE_PYTHON" --cada-python "$CADA_PYTHON" \
    --moses-cada-python "$MOSES_CADA_PYTHON" \
    --methods rfte moses_cada --batch-size "$bs"
done
```

Run the two complete-batch gates for each method, size, and batch size. This
function does not install packages or mutate an upstream checkout.

```bash
run_remaining_batch_gate () {
  method="$1"; size="$2"; bs="$3"; scope="$4"; python_exe="$5"
  upstream="$6"; checkpoint="$7"; checkpoint_sha="$8"
  evidence_args=()
  if [ "$scope" = validation_gate ]; then
    evidence_args=(--small-gate-evidence "$REMAINING_ARTIFACT_ROOT/results/$method/cvrptw${size}/bs${bs}/small_gate")
  fi
  "$python_exe" -B "methods/$method/cvrptw/paper_eval.py" \
    --scope "$scope" --problem-size "$size" --batch-size "$bs" \
    --input "$REMAINING_ARTIFACT_ROOT/prepared/cvrptw${size}/${scope}_bs${bs}.npz" \
    --dataset "$REMAINING_DATASET_ROOT/cvrptw${size}_pyvrp-$([ "$size" = 50 ] && echo '10s_16.038' || echo '20s_25.431').pkl" \
    --upstream "$upstream" --checkpoint "$checkpoint" \
    --expected-checkpoint-sha256 "$checkpoint_sha" \
    --output-dir "$REMAINING_ARTIFACT_ROOT/results/$method/cvrptw${size}/bs${bs}/$scope" \
    "${evidence_args[@]}" --warmup-batches 2 --device cuda:0
}

for bs in 1 10; do
  for scope in small_gate validation_gate; do
    run_remaining_batch_gate rfte 50 "$bs" "$scope" "$RFTE_PYTHON" "$BASELINE_PROJECT_ROOT/external/routefinder" "$BASELINE_PROJECT_ROOT/external/routefinder/checkpoints/50/rf-transformer.ckpt" "$RFTE50_SHA"
    run_remaining_batch_gate rfte 100 "$bs" "$scope" "$RFTE_PYTHON" "$BASELINE_PROJECT_ROOT/external/routefinder" "$BASELINE_PROJECT_ROOT/external/routefinder/checkpoints/100/rf-transformer.ckpt" "$RFTE100_SHA"
    run_remaining_batch_gate moses_cada 50 "$bs" "$scope" "$MOSES_CADA_PYTHON" "$BASELINE_PROJECT_ROOT/external/moses_vrp" "$BASELINE_PROJECT_ROOT/external/moses_vrp/pretrained_moses_model/cada/50/multilora_denseroute_sigmoid.ckpt" "$MOSES50_SHA"
    run_remaining_batch_gate moses_cada 100 "$bs" "$scope" "$MOSES_CADA_PYTHON" "$BASELINE_PROJECT_ROOT/external/moses_vrp" "$BASELINE_PROJECT_ROOT/external/moses_vrp/pretrained_moses_model/cada/100/multilora_denseroute_sigmoid.ckpt" "$MOSES100_SHA"
  done
done
```

After every matching validation gate is `KIT_VALIDATED`, dry-run and then
execute each production cell. `TIME` is mean native-batch latency and `TOTAL`
is the sum of the 1000 BS1 or 100 BS10 timings.

```bash
launch_remaining_batch_production () {
  method="$1"; size="$2"; bs="$3"; python_exe="$4"
  upstream="$5"; checkpoint="$6"; checkpoint_sha="$7"; execute_flag="${8:-}"
  "$ML4CO_PYTHON" -B scripts/launch_remaining_cvrptw_production.py \
    --method "$method" --problem-size "$size" --batch-size "$bs" \
    --python "$python_exe" \
    --input "$REMAINING_ARTIFACT_ROOT/prepared/cvrptw${size}/production.npz" \
    --dataset "$REMAINING_DATASET_ROOT/cvrptw${size}_pyvrp-$([ "$size" = 50 ] && echo '10s_16.038' || echo '20s_25.431').pkl" \
    --upstream "$upstream" --checkpoint "$checkpoint" \
    --expected-checkpoint-sha256 "$checkpoint_sha" \
    --validation-gate-evidence "$REMAINING_ARTIFACT_ROOT/results/$method/cvrptw${size}/bs${bs}/validation_gate" \
    --output-dir "$REMAINING_ARTIFACT_ROOT/results/$method/cvrptw${size}/bs${bs}/production" \
    $execute_flag
}

for bs in 1 10; do
  launch_remaining_batch_production rfte 50 "$bs" "$RFTE_PYTHON" "$BASELINE_PROJECT_ROOT/external/routefinder" "$BASELINE_PROJECT_ROOT/external/routefinder/checkpoints/50/rf-transformer.ckpt" "$RFTE50_SHA"
  launch_remaining_batch_production rfte 100 "$bs" "$RFTE_PYTHON" "$BASELINE_PROJECT_ROOT/external/routefinder" "$BASELINE_PROJECT_ROOT/external/routefinder/checkpoints/100/rf-transformer.ckpt" "$RFTE100_SHA"
  launch_remaining_batch_production moses_cada 50 "$bs" "$MOSES_CADA_PYTHON" "$BASELINE_PROJECT_ROOT/external/moses_vrp" "$BASELINE_PROJECT_ROOT/external/moses_vrp/pretrained_moses_model/cada/50/multilora_denseroute_sigmoid.ckpt" "$MOSES50_SHA"
  launch_remaining_batch_production moses_cada 100 "$bs" "$MOSES_CADA_PYTHON" "$BASELINE_PROJECT_ROOT/external/moses_vrp" "$BASELINE_PROJECT_ROOT/external/moses_vrp/pretrained_moses_model/cada/100/multilora_denseroute_sigmoid.ckpt" "$MOSES100_SHA"
done
```

Each new artifact contains `metadata.json`, `validated_records.jsonl`,
`batch_timings.jsonl`, and `summary.json`. A partial native batch blocks resume.

## 12. Regression tests

```bash
ML4CO_REFERENCE_TESTS=1 python -B -m unittest discover -s tests -v
```

## 13. GLOP TSP Parallel Table

This evaluator is separate from the frozen BS1 formal artifacts. It runs only
TSP100/500/1000 with native original-instance batch sizes 16 or 128. Never
split an OOM batch, lower its batch size, or reuse an existing output directory.

```bash
conda activate cp311_base
export BASELINE_PROJECT_ROOT=/inspire/hdd/global_user/majiale-253108540229/zhang/neural-routing-baselines
export GLOP_UPSTREAM="$BASELINE_PROJECT_ROOT/external/GLOP"
export GLOP_PARALLEL_DATASET_ROOT=/inspire/hdd/global_user/majiale-253108540229/zhang/datasets/ML4CO-Bench-101-SL
read -r -p 'Official GLOP pretrained root: ' GLOP_ASSET_ROOT
read -r -p 'TSP100 standard BS1 PAPER_READY summary.json: ' GLOP_BS1_TSP100_STANDARD
read -r -p 'TSP100 more BS1 PAPER_READY summary.json: ' GLOP_BS1_TSP100_MORE
read -r -p 'TSP500 standard BS1 PAPER_READY summary.json: ' GLOP_BS1_TSP500_STANDARD
read -r -p 'TSP500 more BS1 PAPER_READY summary.json: ' GLOP_BS1_TSP500_MORE
read -r -p 'TSP1000 standard BS1 PAPER_READY summary.json: ' GLOP_BS1_TSP1000_STANDARD
read -r -p 'TSP1000 more BS1 PAPER_READY summary.json: ' GLOP_BS1_TSP1000_MORE
export GLOP_PARALLEL_ROOT="/inspire/hdd/global_user/majiale-253108540229/zhang/verification_evidence/neural-routing-baselines/glop/$(git -C "$BASELINE_PROJECT_ROOT" rev-parse --short=8 HEAD)/parallel_table"
cd "$BASELINE_PROJECT_ROOT"
test -z "$(git status --porcelain)"
test "$(git -C "$GLOP_UPSTREAM" rev-parse HEAD)" = e540bc0153a0598e923e35116deeaecaf9c1cfff
test -z "$(git -C "$GLOP_UPSTREAM" status --porcelain)"
```

Prepare one exact complete input per size/protocol. The two batch sizes for a
given size/protocol read the same immutable prepared input.

```bash
for size in 100 500 1000; do
  case "$size" in
    100) dataset="$GLOP_PARALLEL_DATASET_ROOT/tsp100_concorde_7.756.pkl"; count=1280 ;;
    500) dataset="$GLOP_PARALLEL_DATASET_ROOT/tsp500_concorde_16.546.pkl"; count=128 ;;
    1000) dataset="$GLOP_PARALLEL_DATASET_ROOT/tsp1000_concorde_23.118.pkl"; count=128 ;;
  esac
  for protocol in official_standard official_more; do
    input="$GLOP_PARALLEL_ROOT/inputs/tsp${size}/${protocol}.npz"
    test ! -e "$input"
    mkdir -p "$(dirname "$input")"
    python -B methods/glop/tsp/prepare_instances.py \
      --dataset "$dataset" --problem-size "$size" --protocol "$protocol" \
      --offset 0 --count "$count" --output "$input"
  done
done
```

Define the command wrapper, then invoke the twelve cells individually. Every
invocation is one real native-batch run and writes its own four-file artifact.

```bash
run_glop_parallel_cell () {
  size="$1"; protocol="$2"; batch_size="$3"; bs1_summary="$4"
  python -B methods/glop/tsp/parallel_eval.py \
    --input "$GLOP_PARALLEL_ROOT/inputs/tsp${size}/${protocol}.npz" \
    --problem-size "$size" --protocol "$protocol" --batch-size "$batch_size" \
    --upstream "$GLOP_UPSTREAM" --asset-root "$GLOP_ASSET_ROOT" \
    --bs1-summary "$bs1_summary" \
    --output-dir "$GLOP_PARALLEL_ROOT/results/tsp${size}/${protocol}/bs${batch_size}" \
    --device cuda:0
}

run_glop_parallel_cell 100 official_standard 16 "$GLOP_BS1_TSP100_STANDARD"
run_glop_parallel_cell 100 official_standard 128 "$GLOP_BS1_TSP100_STANDARD"
run_glop_parallel_cell 100 official_more 16 "$GLOP_BS1_TSP100_MORE"
run_glop_parallel_cell 100 official_more 128 "$GLOP_BS1_TSP100_MORE"
run_glop_parallel_cell 500 official_standard 16 "$GLOP_BS1_TSP500_STANDARD"
run_glop_parallel_cell 500 official_standard 128 "$GLOP_BS1_TSP500_STANDARD"
run_glop_parallel_cell 500 official_more 16 "$GLOP_BS1_TSP500_MORE"
run_glop_parallel_cell 500 official_more 128 "$GLOP_BS1_TSP500_MORE"
run_glop_parallel_cell 1000 official_standard 16 "$GLOP_BS1_TSP1000_STANDARD"
run_glop_parallel_cell 1000 official_standard 128 "$GLOP_BS1_TSP1000_STANDARD"
run_glop_parallel_cell 1000 official_more 16 "$GLOP_BS1_TSP1000_MORE"
run_glop_parallel_cell 1000 official_more 128 "$GLOP_BS1_TSP1000_MORE"
```

Each successful cell must contain `metadata.json`, `validated_records.jsonl`,
`batch_timings.jsonl`, and `summary.json`, with `summary.status=PAPER_READY`.
An RTX4090 OOM records `metadata.state=CUDA_OOM` and terminates without retrying
at a smaller batch size.

## 14. Return evidence

Return these files without editing their values:

- `artifacts/server/environment/server_environment.json`
- `artifacts/server/audit/server_datasets.json`
- `artifacts/server/audit/server_upstreams.json`
- `artifacts/server/audit/server_checkpoints.json`
- each requested `official_search_aug8_first5_validated.json`
- each requested NeuOpt `official_first5_validated.json`
- each requested GLOP TSP formal_first5_validated.json
- artifacts/server/audit/glop_dependency_environment.txt
- the regression-test terminal output

Also return SHA256 for each validated result artifact. No password, private key or SSH access is required.
