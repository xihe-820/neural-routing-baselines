# 工程设计

主仓库拥有路径管理、provenance、method/problem adapters、实际解提取、独立验证、统一结果。外部官方仓库拥有 architecture、checkpoint loader、搜索与 decode。当前已实现审计层与problem validators，method-specific adapters仍为明确接口设计。禁止common_cvrp_tensordict_for_all_methods这类共享baseline输入转换。

目录按 `methods/<method>/<problem>/` 隔离；共享数学验证按 `problems/tsp|cvrp|cvrptw/` 隔离。未实现的模块使用 README 明示，不放返回成功的占位函数。

## 路径与进程

| Input | CLI | Environment | Default |
|---|---|---|---|
| 官方 checkout 根目录 | `--upstream-root`（兼容 `--external-root`） | `BASELINE_UPSTREAM_ROOT` | `external` |
| Benchmark 根目录 | `--dataset-root` | `ML4CO_DATA_ROOT` | 无；或逐个 `--dataset` |
| 额外 checkpoint 根目录 | `--checkpoint-root` | `BASELINE_CHECKPOINT_ROOT` | 当前目录 |
| 输出根目录 | `--artifact-root` | `BASELINE_ARTIFACT_ROOT` | `artifacts` |
| 特定 JSON | `--output` | — | 相对 artifact 根目录 |

`audit_model_load.py` 用显式 `--repo` 和 `--output`；`--module-path` 只允许该 repo 内部目录。相对路径均相对于命令 cwd。服务器路径不进入源码。单个 baseline 的所有 imports 在独立解释器内完成；环境检查也将每个 package 放入独立、有 timeout 的子进程。`-B` / `PYTHONDONTWRITEBYTECODE` 避免生成 upstream pycache。审计脚本不安装软件、不 fetch、不 checkout、不运行训练/测试入口。

## Provenance 与结果设计

后续结果每个 instance 独立一行：method、problem、size、instance_id；官方 URL/commit/dirty；主仓库 commit/dirty（尚无 commit 时明确 null）；checkpoint path/source/hash；dataset realpath/hash/index；adapter/source hash；环境 timestamp/host/Python/Torch/CUDA/GPU；seed、完整 decode 参数、batch size、augmentation、multistart、search iterations、width/alpha 的实际含义；实际 solution；reported/independent/Kit objective；independent/Kit feasibility 与各约束细节；wall-clock runtime。

Checkpoint 部件用列表记录（例如 GLOP partitioner + 多 reviser，UDC partition + conquer），不能只记一个总路径。主文件和 `args.json` 等构造配置均需 hash。完整 `state_dict` key/shape 保存在忽略的 audit JSON；manifest 只放来源、固定版本、预期布局与紧凑事实。manifest 使用 JSON 语法（YAML 1.2 的子集），基础审计无需 PyYAML。

未验证值用 null 或明确 pending 状态；不能将空数组/默认 True 解释为通过。成本比较记录绝对误差、相对误差与容差。失败保存原因，不伪造解或 fallback score。最终报告必须同时保留 baseline 原始 cost 与两次重算结果，不能用独立重算覆盖 reported cost。

## Fidelity 与输出提取

优先通过官方已有 actions/selected_node_list 获取最终 best tour，按同一 best start/augmentation 索引选择。只记录调用返回的 output，不再次运行搜索、不重新采样、不消耗额外 RNG。必要的临时 wrapper 必须调用原函数恰好一次、原样返回、finally 恢复，且记录 shim source hash。

GLOP 坐标解不能靠最近邻猜 ID：保留映射，处理重合坐标与 depot padding 歧义，完整验证 permutation/route。CaDA 原始 forward 未向外返回 actions，后续需要经过审查的 output capture。NeuOpt 迁移旧 successor decoder，保留 batch>=2 与 protobuf workaround 的历史约束；当前阶段不重新实现。

## 本轮交付的范围

`audit_environment.py`：版本、import、CUDA discovery、pip check；不证明 CUDA compiled ops。
`audit_assets.py`：checkout 状态、checkpoint 文件/哈希/LFS/CPU payload；不构建完整 algorithm。
`audit_datasets.py`：候选文件发现、实际 task class/size、首实例 schema、Euclidean reference distance、Kit 验证；调用独立problem validators验证首个reference（包含TW），另做Kit二次核验；不是method inference。
`audit_model_load.py`：单 repo 模块 import，及 MVMoE/NeuOpt 两组已审计构造的 strict CPU load；不执行 forward/decode。

独立validator契约见 [VALIDATION_DESIGN](VALIDATION_DESIGN.md)。每个method/problem README固定自己的native边界；共享的只有problem truth与provenance，不共享baseline TensorDict构造。Kit检查可以记录不可用原因，不覆盖独立验证结果。
