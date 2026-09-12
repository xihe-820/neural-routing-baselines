# MVMoE/4E / CVRP50 — LOCAL_VERIFIED integration

本目录只实现 CVRP50；CVRP100 与 CVRPTW 未复用或扩展。使用官方 upstream `af29e5af0595f94f3ecc3bc46d72df1089a62682`、`pretrained/mvmoe_4e_n50/epoch-5000.pt`（SHA256 `3417f302fbddf232fd19a2a886cd1c7f44290b6d8c7280fcb0ae3777eeed3192`）和正式 ML4CO CVRP50 数据。

- `prepare_instances.py`：在已有 Kit 环境中验证dataset SHA/task/size/capacity，把指定实例导出为pickle-free NPZ；保持raw demand/capacity。
- `adapter.py`：直接适配官方 `CVRPEnv.load_problems`，只做一次 `raw_demand/raw_capacity`；输出 `[B,1,2]`、`[B,50,2]`、`[B,50]`。
- `run.py`：直接构造官方 `MOEModel` 与 `CVRPEnv`，strict load后按 `load_problems → reset → pre_forward → pre_step → model/step` rollout；不导入Tester，不创建optimizer，不训练/backward/fine-tune。
- `decode.py`：按官方 reward 的同一 augmentation/POMO index gather `selected_node_list`。官方CVRPEnv首步为depot；finished POMO会继续选择depot等待，因此仅把最后连续depot规范为一个，保留内部route separators，不修复重复/缺失customer。
- `validate_with_kit.py`：在已有 Kit 环境读取canonical JSON和hash-pinned source task，补入secondary feasibility/objective。

正式模型：MOE（非MOE_LIGHT），embedding128、encoder6、decoder1、qkv16、heads8、ff512、4 experts、topk2、node/input_choice、argmax、instance norm/norm_last、experts at Enc0..5+Dec、checkpoint problem Train_ALL。`problem_size=pomo_size=50`，seed2024。Reduced smoke使用aug1并明确标为debug；completion smoke使用官方aug8。

8-fold augmentation来自官方 `CVRPEnv.augment_xy_data_by_8_fold`，只变换坐标、不改变节点顺序；因此native IDs仍是benchmark depot0/customers1..50。独立objective始终在原始benchmark coordinates上重算。每个result保存完整Git/checkpoint/dataset/source SHA、配置、best augmentation/POMO、actual canonical solution、官方/独立/Kit成本、约束和环境。

本地RTX4060 official-config first-5：5/5 independent feasible，5/5 Kit feasible；最大official reward与独立成本绝对差约`1.17e-6`。逐实例结果见 [integration manifest](../../../manifests/mvmoe_cvrp50.json)。本地成功不等于服务器成功；server字段仍NOT_RUN。

```bash
# Kit environment: prepare exact first five tasks
python -B methods/mvmoe/cvrp/prepare_instances.py --dataset "$CVRP50_DATASET" \
  --offset 0 --count 5 --output artifacts/mvmoe_cvrp50/input_first5.npz

# Model environment: official configuration
python -B methods/mvmoe/cvrp/run.py --input artifacts/mvmoe_cvrp50/input_first5.npz \
  --upstream "$MVMOE_UPSTREAM" --checkpoint "$MVMOE_CHECKPOINT" \
  --output artifacts/mvmoe_cvrp50/official_aug8_first5.json \
  --aug-factor 8 --seed 2024 --device auto

# Kit environment: secondary validation
python -B methods/mvmoe/cvrp/validate_with_kit.py \
  --input artifacts/mvmoe_cvrp50/official_aug8_first5.json \
  --dataset "$CVRP50_DATASET" \
  --output artifacts/mvmoe_cvrp50/official_aug8_first5_validated.json
```
