# SERVER_RUNBOOK：用户手动部署和复核

仅包含目前已实现的命令。Codex不连接服务器；默认不安装、更换环境，不修改official source。当前server事实全为NOT_RUN。本地模型/数据identity已建立的部分只需复核服务器实际副本与执行环境。

## Step 1 — repository sync / setup

主仓库为 `https://github.com/xihe-820/neural-routing-baselines`；Phase 0/1 snapshot 是 `38f794fc19c3aaa988bfa78db0ae51143aab93ef`。服务器必须 clone/fetch 后 checkout 用户明确批准的项目 commit。当前 MVMoE/CVRP50 integration 尚未自动commit/push，因此实际服务器运行时请将下面变量设置为审查后包含本轮文件的commit；不能用旧snapshot冒充新runner。

在同一终端粘贴并输入当前实际路径（不依赖shell history）：

```bash
read -r -p 'New/existing baseline repository absolute path: ' BASELINE_PROJECT_ROOT
read -r -p 'Approved project commit containing MVMoE/CVRP50 integration: ' BASELINE_PROJECT_COMMIT
read -r -p 'Existing official checkout root: ' BASELINE_UPSTREAM_ROOT
read -r -p 'ML4CO-Bench dataset root: ' ML4CO_DATA_ROOT
export BASELINE_PROJECT_ROOT BASELINE_PROJECT_COMMIT BASELINE_UPSTREAM_ROOT ML4CO_DATA_ROOT
export PYTHONDONTWRITEBYTECODE=1
if [ -d "$BASELINE_PROJECT_ROOT/.git" ]; then
  git -C "$BASELINE_PROJECT_ROOT" checkout master
  git -C "$BASELINE_PROJECT_ROOT" pull --ff-only origin master
else
  git clone https://github.com/xihe-820/neural-routing-baselines.git "$BASELINE_PROJECT_ROOT"
fi
git -C "$BASELINE_PROJECT_ROOT" checkout --detach "$BASELINE_PROJECT_COMMIT"
test "$(git -C "$BASELINE_PROJECT_ROOT" rev-parse HEAD)" = "$BASELINE_PROJECT_COMMIT"
cd "$BASELINE_PROJECT_ROOT"
export BASELINE_ARTIFACT_ROOT="$BASELINE_PROJECT_ROOT/artifacts/server"
mkdir -p "$BASELINE_ARTIFACT_ROOT"
conda activate cp311_base
```

若activate失败，回传错误，不自动创建或重装。实际解释器由下一步记录。

## Step 2 — environment audit

```bash
python -B scripts/audit_environment.py --label server-user-executed \
  --output "$BASELINE_ARTIFACT_ROOT/environment/server_environment.json" \
  --markdown "$BASELINE_ARTIFACT_ROOT/environment/server_environment.md"
```

检查Python/Torch/CUDA/GPU、包import、pip check逐项结果；CUDA discovery不是model forward。

## Step 3 — dataset audit and SHA comparison

```bash
python -B scripts/audit_datasets.py --dataset-root "$ML4CO_DATA_ROOT" \
  --expected-manifest manifests/public_datasets.json --label server-user-executed \
  --output "$BASELINE_ARTIFACT_ROOT/audit/server_datasets.json"
```

每个候选输出`manifest_identity_match`；核对六格coverage、SHA、task class/shape、首例独立validator与Kit。hash匹配但loader失败只证明副本身份，不能证明服务器解析成功。非标准命名可用`--dataset PROBLEM SIZE PATH`重复指定。此脚本会读取可信官方pickle；不要把无关数据目录作为root。

## Step 4 — upstream checkout verification

```bash
python -B scripts/audit_assets.py --upstream-root "$BASELINE_UPSTREAM_ROOT" \
  --output "$BASELINE_ARTIFACT_ROOT/audit/server_upstreams.json"
```

核对实际URL/HEAD/dirty/tag与manifests/upstreams.yaml。默认checkout子目录为GLOP/NCO_code/CaDA/Routing-MVMoE/routefinder/moses_vrp/NeuOpt；未找到会记录exists=false。不会自动fetch/checkout/清理。名字不同先回传实际布局，不移动或覆盖原目录。

## Step 5 — checkpoint SHA comparison

```bash
python -B scripts/audit_assets.py --upstream-root "$BASELINE_UPSTREAM_ROOT" \
  --compare-manifest manifests/checkpoints.yaml \
  --output "$BASELINE_ARTIFACT_ROOT/audit/server_checkpoints.json"
```

`manifest_identity_match=true`只表示与该method已知本地SHA一致；false是不同副本，null表示尚无本地预期SHA。还应核对target_size/asset_role。完整binary、非LFS、SHA与architecture是分开的检查。需要首次payload检查时对同一命令添加`--load-checkpoints`，在隔离子进程CPU反序列化；不重复本地已冻结strict-load作为身份推断。

若资产在checkout外，显式补充：

```bash
read -r -p 'Method (GLOP/UDC/CaDA/MVMoE/RF-TE/MoSES(CaDA)/NeuOpt): ' CHECKPOINT_METHOD
read -r -p 'Official checkpoint root: ' BASELINE_CHECKPOINT_ROOT
read -r -p 'Checkpoint relative path: ' CHECKPOINT_FILE
export BASELINE_CHECKPOINT_ROOT
python -B scripts/audit_assets.py --upstream-root "$BASELINE_UPSTREAM_ROOT" \
  --checkpoint-root "$BASELINE_CHECKPOINT_ROOT" \
  --checkpoint "$CHECKPOINT_METHOD" "$CHECKPOINT_FILE" \
  --compare-manifest manifests/checkpoints.yaml --load-checkpoints \
  --output "$BASELINE_ARTIFACT_ROOT/audit/server_checkpoint_explicit.json"
```

多组件重复`--checkpoint METHOD PATH`；GLOP需要reviser配套args.json。外部路径payload检查支持，MVMoE/NeuOpt下面strict构造脚本目前仍使用官方repo相对权重布局；若资产布局不同先回传，不修改源码硬编码。

## Step 6 — per-method import/load smoke（仅导入与 CPU 构造）

以下是独立进程，不执行原 test/run entrypoint，因此不会意外进入训练、创建官方日志或复制源码。

```bash
python -B scripts/audit_model_load.py --repo "$BASELINE_UPSTREAM_ROOT/GLOP" \
  --module nets.attention_local --module nets.partition_net --module utils.insertion \
  --output "$BASELINE_ARTIFACT_ROOT/audit/glop_import.json"

python -B scripts/audit_model_load.py --repo "$BASELINE_UPSTREAM_ROOT/NCO_code" \
  --module-path single_objective/UDC-Large-scale-CO-master/UDC/TSP-AGNN-ICAM \
  --module TSPModel --module PartitionModel \
  --output "$BASELINE_ARTIFACT_ROOT/audit/udc_tsp_import.json"

python -B scripts/audit_model_load.py --repo "$BASELINE_UPSTREAM_ROOT/NCO_code" \
  --module-path single_objective/UDC-Large-scale-CO-master/UDC/CVRP-AGNN-ICAM \
  --module CVRPModel --module PartitionModel \
  --output "$BASELINE_ARTIFACT_ROOT/audit/udc_cvrp_import.json"

for PROBLEM_SIZE in 50 100; do
  python -B scripts/audit_model_load.py --repo "$BASELINE_UPSTREAM_ROOT/CaDA" \
    --module-path "$PROBLEM_SIZE" --module model --module envs.env \
    --output "$BASELINE_ARTIFACT_ROOT/audit/cada_${PROBLEM_SIZE}_import.json"
done

python -B scripts/audit_model_load.py --repo "$BASELINE_UPSTREAM_ROOT/Routing-MVMoE" \
  --method MVMoE --module models.MOEModel --module Tester \
  --output "$BASELINE_ARTIFACT_ROOT/audit/mvmoe_load.json"

python -B scripts/audit_model_load.py --repo "$BASELINE_UPSTREAM_ROOT/routefinder" \
  --module routefinder.models.policy --module routefinder.envs.mtvrp.env \
  --output "$BASELINE_ARTIFACT_ROOT/audit/rfte_import.json"

python -B scripts/audit_model_load.py --repo "$BASELINE_UPSTREAM_ROOT/moses_vrp" \
  --module models.policy --module envs.mtvrp.env \
  --output "$BASELINE_ARTIFACT_ROOT/audit/moses_cada_import.json"

python -B scripts/audit_model_load.py --repo "$BASELINE_UPSTREAM_ROOT/NeuOpt" \
  --method NeuOpt --module nets.actor_network --module agent.ppo \
  --output "$BASELINE_ARTIFACT_ROOT/audit/neuopt_load.json"
```

MVMoE/NeuOpt 包含官方模型 CPU strict load；其余方法目前只有 module import + Step 5 payload load，architecture construction/forward 要等依赖与资产信息回来再补。这些区别都在 JSON 中记录。NeuOpt 若复现历史 protobuf generated-code 错误，可额外设置 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 后写到另一份输出，以保留原错误证据；不要降级 protobuf。


## Step 7 — MVMoE/4E CVRP50 official-config smoke

以下命令只运行首5例、aug8/POMO50/argmax/seed2024，不训练。它直接使用官方MOEModel/CVRPEnv，不导入Tester或fine-tune路径。设置三个实际文件路径；checkpoint与dataset SHA不匹配时runner会拒绝。

```bash
read -r -p 'Exact CVRP50 benchmark pickle: ' CVRP50_DATASET
read -r -p 'Exact Routing-MVMoE checkout: ' MVMOE_UPSTREAM
read -r -p 'Exact MVMoE 4E n50 checkpoint: ' MVMOE_CHECKPOINT
export CVRP50_DATASET MVMOE_UPSTREAM MVMOE_CHECKPOINT

python -B methods/mvmoe/cvrp/prepare_instances.py \
  --dataset "$CVRP50_DATASET" --offset 0 --count 5 \
  --output "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrp50/input_first5.npz"

python -B methods/mvmoe/cvrp/run.py \
  --input "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrp50/input_first5.npz" \
  --upstream "$MVMOE_UPSTREAM" --checkpoint "$MVMOE_CHECKPOINT" \
  --output "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrp50/official_aug8_first5.json" \
  --aug-factor 8 --seed 2024 --device auto
```

`prepare_instances.py`需要当前解释器已有ML4CO-Kit；若cp311_base没有Kit，使用服务器已有Kit环境只执行第一条，再切回cp311_base运行模型。不要为此更换模型环境核心依赖。`--device auto`有CUDA则用当前GPU，否则合法回退CPU并在JSON明确记录。

## Step 8 — independent validation

`run.py`已对每个actual decoded solution调用共享 `problems.cvrp.validate.validate`，并保存官方reward对应成本、独立成本、误差、raw capacity检查及actual route。若任一`independent_feasible`不是true或成本误差明显偏离约1e-6量级，停止并回传原始JSON，不放宽容差。

## Step 9 — secondary ML4CO-Kit validation

在已有Kit的解释器中执行：

```bash
python -B methods/mvmoe/cvrp/validate_with_kit.py \
  --input "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrp50/official_aug8_first5.json" \
  --dataset "$CVRP50_DATASET" \
  --output "$BASELINE_ARTIFACT_ROOT/mvmoe_cvrp50/official_aug8_first5_validated.json"
```

必须检查5个row均为`evidence_status=LOCAL_VERIFIED`、`independent_feasible=true`、`kit_feasible=true`，并保留每个best augmentation/POMO和canonical solution。服务器验证通过后，证据等级才可升级SERVER_VERIFIED。

## Step 10 — regression tests

```bash
ML4CO_REFERENCE_TESTS=1 python -B -m unittest discover -s tests -v
```

如果服务器环境不含全部六组reference数据，可先不设置`ML4CO_REFERENCE_TESTS`运行其余unit tests；这不会替代上面的MVMoE/CVRP50实际解与Kit验证。

回传 `artifacts/server/environment/server_environment.json`、`artifacts/server/audit/` 和 `artifacts/server/mvmoe_cvrp50/official_aug8_first5_validated.json` 及测试终端输出。无需密码、私钥、SSH权限。
