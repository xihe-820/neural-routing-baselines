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

## 9. Regression tests

```bash
ML4CO_REFERENCE_TESTS=1 python -B -m unittest discover -s tests -v
```

## 10. Return evidence

Return these files without editing their values:

- `artifacts/server/environment/server_environment.json`
- `artifacts/server/audit/server_datasets.json`
- `artifacts/server/audit/server_upstreams.json`
- `artifacts/server/audit/server_checkpoints.json`
- each requested `official_search_aug8_first5_validated.json`
- each requested NeuOpt `official_first5_validated.json`
- the regression-test terminal output

Also return SHA256 for each validated result artifact. No password, private key or SSH access is required.
