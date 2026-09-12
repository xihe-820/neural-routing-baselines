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

## 8. Regression tests

```bash
ML4CO_REFERENCE_TESTS=1 python -B -m unittest discover -s tests -v
```

## 9. Return evidence

Return these files without editing their values:

- `artifacts/server/environment/server_environment.json`
- `artifacts/server/audit/server_datasets.json`
- `artifacts/server/audit/server_upstreams.json`
- `artifacts/server/audit/server_checkpoints.json`
- each requested `official_search_aug8_first5_validated.json`
- the regression-test terminal output

Also return SHA256 for each validated result artifact. No password, private key or SSH access is required.
