# SERVER_RUNBOOK：用户手动部署和复核

仅包含目前已实现的命令。Codex不连接服务器；默认不安装、更换环境，不修改official source。当前server事实全为NOT_RUN。本地模型/数据identity已建立的部分只需复核服务器实际副本与执行环境。

## Step 1 — repository sync / setup

主仓库尚无commit/remote；手动复制自有源码、docs、manifests、tests，保持目录层级。external/checkpoints/datasets均gitignored，可沿用服务器已有副本；缺失六组数据时也可复制本地对应文件。不要把本地绝对环境路径搬入源码。

在同一终端粘贴并输入当前实际路径（不依赖shell history）：

```bash
read -r -p 'New baseline repository absolute path: ' BASELINE_PROJECT_ROOT
read -r -p 'Existing official checkout root: ' BASELINE_UPSTREAM_ROOT
read -r -p 'ML4CO-Bench dataset root: ' ML4CO_DATA_ROOT
export BASELINE_PROJECT_ROOT BASELINE_UPSTREAM_ROOT ML4CO_DATA_ROOT
export PYTHONDONTWRITEBYTECODE=1
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


## Step 7 — later GPU / native-format smoke

NOT_IMPLEMENTED：待一个method/problem/size wrapper完成后补具体命令。可用已有小official data或合法tiny native-format实例验证入口；不是绝对benchmark gate。最少合法batch先跑，NeuOpt避免batch1。

## Step 8 — later ML4CO small inference

NOT_IMPLEMENTED：保留固定source SHA/index和实际search配置，运行最少合法实例，随后约5例sanity。our_2/our_5仅命名约定。

## Step 9 — independent validation

Problem validators与reference tests已可运行：

```bash
python -B -m unittest discover -s tests -v
```

模型产生的solution validation等decoder实现再补runner命令；reference成功不能替代actual solution。

## Step 10 — secondary ML4CO-Kit reference validation

当前解释器已有Kit且数据为本地manifest对应六文件时：

```bash
ML4CO_REFERENCE_TESTS=1 python -B -m unittest discover -s tests -v
```

该测试要求ML4CO_DATA_ROOT下六个manifest文件直接可访问（若原目录有层级，请将变量设为包含这六文件的目录，或先回传布局）；不自动下载/安装。正式method输出Kit检查尚未实现。Kit是secondary，缺包/encoding错误单独记录。

回传 `artifacts/server/environment/server_environment.json` 与 `artifacts/server/audit/` JSON及测试终端输出。exit0可能只是报告写成功，须读逐项error/strict_load/identity/coverage。无需密码、私钥、SSH权限。服务器运行完成后才将相应字段升级SERVER_VERIFIED。
