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

Supply explicit dataset paths. Filename discovery is not used by formal
preparation.

```bash
export GLOP_UPSTREAM="$BASELINE_UPSTREAM_ROOT/GLOP"
read -r -p 'Official GLOP pretrained root: ' GLOP_ASSET_ROOT
read -r -p 'TSP100 dataset: ' GLOP_TSP100_DATASET
read -r -p 'TSP500 dataset: ' GLOP_TSP500_DATASET
read -r -p 'TSP1K dataset: ' GLOP_TSP1K_DATASET
read -r -p 'TSP2K dataset: ' GLOP_TSP2K_DATASET
read -r -p 'TSP5K dataset: ' GLOP_TSP5K_DATASET
read -r -p 'TSP10K dataset: ' GLOP_TSP10K_DATASET
read -r -p 'CVRP500 dataset: ' GLOP_CVRP500_DATASET
read -r -p 'CVRP1K dataset: ' GLOP_CVRP1K_DATASET
read -r -p 'CVRP2K dataset: ' GLOP_CVRP2K_DATASET
export GLOP_ASSET_ROOT GLOP_PAPER_ROOT="$BASELINE_ARTIFACT_ROOT/paper/glop"

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
  --output "$BASELINE_ARTIFACT_ROOT/audit/glop_paper_datasets.json"
```

For each of TSP500, TSP1K and TSP10K, run both official protocol names:

```bash
for size in 500 1000 10000; do
  case "$size" in
    500) dataset="$GLOP_TSP500_DATASET" ;;
    1000) dataset="$GLOP_TSP1K_DATASET" ;;
    10000) dataset="$GLOP_TSP10K_DATASET" ;;
  esac
  for protocol in official_standard official_more; do
    python -B methods/glop/tsp/prepare_instances.py \
      --dataset "$dataset" --problem-size "$size" --protocol "$protocol" \
      --offset 0 --count 2 \
      --output "$GLOP_PAPER_ROOT/tsp$size/$protocol/input.npz"
    python -B methods/glop/tsp/paper_eval.py \
      --input "$GLOP_PAPER_ROOT/tsp$size/$protocol/input.npz" \
      --problem-size "$size" --protocol "$protocol" \
      --upstream "$GLOP_UPSTREAM" --asset-root "$GLOP_ASSET_ROOT" \
      --output-dir "$GLOP_PAPER_ROOT/tsp$size/$protocol/preflight" \
      --device cuda:0
  done
done
```

```bash
python -B methods/glop/cvrp/prepare_instances.py \
  --dataset "$GLOP_CVRP1K_DATASET" --problem-size 1000 \
  --protocol official_single --offset 0 --count 2 \
  --output "$GLOP_PAPER_ROOT/cvrp1000/official_single/input.npz"
python -B methods/glop/cvrp/paper_eval.py \
  --input "$GLOP_PAPER_ROOT/cvrp1000/official_single/input.npz" \
  --problem-size 1000 --protocol official_single --upstream "$GLOP_UPSTREAM" \
  --asset-root "$GLOP_ASSET_ROOT" \
  --output-dir "$GLOP_PAPER_ROOT/cvrp1000/official_single/preflight" \
  --device cuda:0

python -B methods/glop/cvrp/prepare_instances.py \
  --dataset "$GLOP_CVRP2K_DATASET" --problem-size 2000 \
  --protocol official_single --offset 0 --count 2 \
  --output "$GLOP_PAPER_ROOT/cvrp2000/official_single/input.npz"
python -B methods/glop/cvrp/paper_eval.py \
  --input "$GLOP_PAPER_ROOT/cvrp2000/official_single/input.npz" \
  --problem-size 2000 --protocol official_single --upstream "$GLOP_UPSTREAM" \
  --asset-root "$GLOP_ASSET_ROOT" \
  --output-dir "$GLOP_PAPER_ROOT/cvrp2000/official_single/preflight" \
  --device cuda:0
```

Do not prepare or run formal preflights for TSP100, TSP2K, TSP5K, or
CVRP500. After inference, apply the Kit gate:

```bash
for protocol in official_standard official_more; do
  python -B methods/glop/validate_with_kit.py \
    --dataset "$GLOP_TSP500_DATASET" \
    --chunk-dirs "$GLOP_PAPER_ROOT/tsp500/$protocol/preflight"
  python -B methods/glop/validate_with_kit.py \
    --dataset "$GLOP_TSP1K_DATASET" \
    --chunk-dirs "$GLOP_PAPER_ROOT/tsp1000/$protocol/preflight"
  python -B methods/glop/validate_with_kit.py \
    --dataset "$GLOP_TSP10K_DATASET" \
    --chunk-dirs "$GLOP_PAPER_ROOT/tsp10000/$protocol/preflight"
done
python -B methods/glop/validate_with_kit.py \
  --dataset "$GLOP_CVRP1K_DATASET" \
  --chunk-dirs "$GLOP_PAPER_ROOT/cvrp1000/official_single/preflight"
python -B methods/glop/validate_with_kit.py \
  --dataset "$GLOP_CVRP2K_DATASET" \
  --chunk-dirs "$GLOP_PAPER_ROOT/cvrp2000/official_single/preflight"
```

## 11. Regression tests

```bash
ML4CO_REFERENCE_TESTS=1 python -B -m unittest discover -s tests -v
```

## 12. Return evidence

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
