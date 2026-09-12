# Native 输入输出与依赖审计

证据基于 [upstreams manifest](../manifests/upstreams.yaml) 的实际固定 checkout。下面描述作者接口，不代表 benchmark adapter 已实现。服务器 cp311_base 的 import、load、forward 均待用户执行。

## GLOP

- 入口：`main.py:eval_dataset/_eval_dataset`；加载 `pretrained/Reviser-stage2/reviser_{L}/epoch-299.pt`，`utils/functions.py:load_model` 读取同目录 `args.json` 构造 attention model。TSP 数据是 `(B,N,2)` 坐标序列，random insertion → 多轮 SHPP revision → reconnect；宽度、augmentation、pruning 都影响搜索，不应擅自取消。
- `main.py:63` 已有 `problem_size<=100` 的 width/4 与 tsp augmentation 分支；README 有 TSP100 cross-distribution 示例，故“小规模从未被考虑”不成立。
- `utils/functions.py:308` 明确 `revision_len<=seed.size(1)`。TSP50 不能用 Reviser100；50/20 是源码可表达的候选（50 对 20 的余数由 decomposition 单独保留），但本地没有 reviser binary/args.json，未确认官方资产完整性或最终合法配置。TSP100 可通过 100/50/20 的尺寸前置条件；不能把这当 forward PASS。
- CVRP native pickle 为 `(depot, loc, raw_demand, capacity)`；`problems/cvrp.py:init` 保留 partitioner+sampler+sub-TSP reviser。`heatmap/cvrp/inst.py` 在图特征中做 demand/capacity，sampler 使用原始容量单位。
- `heatmap/cvrp/infer.py:K_SPARSE` 只有 1000/2000/5000/7000，默认 50/100 会 KeyError。仅传 `--ckpt_path` 不解决该键缺失；cvrplib 分支 k=300 也超过 51/101 个节点。`Net/EmbNet` Linear 参数依赖 feature/hidden/depth，静态没有全局 N 固定权重维；k_sparse 影响边图与 reshape，不仅是文件名。显式k参数的code-supported候选见SIZE_COMPATIBILITY.md；不得bypass partitioner或把候选宣称为已验证配置。
- 输出：TSP reconnect 返回重排坐标；CVRP 是 padded sub-TSP 坐标与路线分组，`main.py` 后续 flatten。需在 flatten 前保留结构，并无歧义恢复 node ID。
- 依赖：PyTorch、NumPy、SciPy、PyG/compiled ops、tqdm、random-insertion。`main.py` 顶层还导入 CVRP partitioner，所以单跑 TSP 也可能碰到 PyG imports。历史版本不是必须整体安装的 lockfile。
- `utils/functions.py:load_model` 实际构造 `nets.attention_local.AttentionModel`，并将 checkpoint 的 `model` 合并到新模型 state 后加载。后续需核对原始 checkpoint 的键完整性，不能让随机初始化的缺失键被合并操作隐藏。
- 结论：`SIZE_COMPATIBILITY_RISK`。TSP100 是首选后续兼容性 gate；CVRP50/100 默认路径不可直接运行；未做 checkpoint forward。

## UDC（仅两个 AGNN-ICAM 子目录）

- TSP：`test-rrc.py`，conquer/partition 默认 epoch 460；`TSPTesterrrc.py:61,68` 路径为 `checkpoint-tsp-460.pt`、`checkpoint-partition-460.pt`，dict key `model_state_dict`。CVRP：`test_rrc.py`，同模式 epoch 230；CVRP conquering 文件也叫 `checkpoint-tsp-230.pt`，不能凭文件名判断为 TSP checkpoint。
- TSP native loader 读取官方 txt 坐标/参考 tour，构造 `raw_problems`。AGNN 创建 initial permutation；ICAM 在 rolling 后的固定 subproblem 上修订，保留端点并选择 best POMO。最终 `solution_gnn`/`solution_out` 为全局 node-ID 解；输出原始 cost 不是可行性证明。
- `TSPTesterrrc.py:239` 实际写死 `torch.topk(..., k=100)`，即使 `gen_pyg_data` 参数名叫 k_sparse，也不能仅改参数解决 N50。`_test_one_batch:171` 用 `.view(batch, -1, sub_size)`；N50/sub_size100 没有合法每条 route 分块，augmentation 轴即使偶然使元素数可整除也会造成语义混合。N100 则满足这两个局部条件。
- CVRP native data 为 depot+customers 的坐标和 normalized demand；需要从 benchmark raw demand 显式除以 capacity 一次。官方 text loader 的大规模解析不能直接接收 ML4CO task pickle。`CVRPTester.py:_load_init_sol` 生成 customer permutation + route boundary `solution_flag`，后续 ICAM 做 subproblem 改进，不是简单 depot-separated action tensor。
- CVRP graph 默认 k=100：N50 加 depot 只有 51 个候选，topk 失败；同时初始化循环 `range(data.size(1)//sub_size)` 对 N50 是零轮（包括 depot 仍为 51//100=0）。输出预置零序列不能算有效解。N100 对 101 个节点的 topk 和单个 100-customer block 尺寸局部可行，仍需 checkpoint+实际 forward。
- 两个 PartitionModel/Conquering model 静态 Linear feature/embedding 维度不直接绑定全局 N，不能仅因为 `problem_size_low=500` 判定 checkpoint 不兼容。但 50 的 hardcoded neighborhood/block 路径不是已证实的合法参数调整。本地无此 UDC 资产，checkpoint shape/forward 均未验证。
- 依赖：torch、NumPy、networkx、PyG 与其编译算子、matplotlib/tqdm。test entrypoint 设置 CUDA/default tensor type、创建日志/复制源码，审计不执行这些 entrypoint。alpha 对应 tester 的 initial solution/augmentation 候选数；不能粗略称为普通坐标 augmentation。batch size 与跨 test set RNG 影响见官方 Readme。
- 结论：`SIZE_COMPATIBILITY_RISK`。100 候选兼容，50 有明确默认路径错误；需先获得官方双 checkpoint、验证 100，再就 50 的方法保真方案报告决策。

## CaDA

- 50/100 各有 `run.py --resume --epoch 300 --path_id ... --n_size ... --test --test_only`；目录分别为 `50/result/2024-1111-1139`、`100/result/2024-1121-1355`。`trainer.py:63` 使用 `checkpoint-300.pt`，key `model_state_dict`，strict=True。同尺寸共用 multi-task 模型用于 CVRP/CVRPTW；当前依据为官方 README/test/task conditioning，binary 尚不在本地。
- Native TensorDict：`locs[B,N+1,2]`、depot-inclusive `demand_linehaul/demand_backhaul[B,N+1]`、`vehicle_capacity[B,1]`、`capacity_original`、`speed`、`open_route`、`distance_limit`、`service_time[B,N+1]`、`time_windows[B,N+1,2]`。generator 在 `scale_demand` 时除 demand/capacity，并把 capacity 变为 1；`env._reset` 直接保留传入需求，不再自动补 depot。
- Task prompt `p_s_tag` 由约束推导，不能拿不匹配 TW/depot defaults 来“关闭”约束。官方测试 loader 是命名约定下的 NPZ；adapter 将由 benchmark task 字段显式构建 TensorDict。
- `model.py:forward` 在多起点 decode 中维护 `actions_list` 并 stack 成 `actions`，outdict 只返回 reward/log_likelihood；必须新增外层捕获设计，保持 greedy/multistart 与 best candidate 索引一致。此轮未修改源码或实现 capture。
- 依赖风险：README torch 2.0.1、torchrl 0.1.1、tensordict 0.1.2、rl4co 0.2.0；model imports einops/TensorDict，env 使用旧 `BoundedTensorSpec/CompositeSpec` API。requirements 含作者 `file://` 路径、flash-attn 本地 wheel、旧 GPU wheels，不可直接安装。需先实测当前服务器 APIs。
- 结论：`VERSION_RISK`；尺寸 intended support 明确，materialized checkpoint/构造待验证。

## MVMoE/4E

- 正式模型 `models.MOEModel.MOEModel`，`model_type=MOE, num_experts=4, routing_level=node, routing_method=input_choice`；默认 `test.py` 的 MOE_LIGHT 不适用。`pretrained/mvmoe_4e_n50|n100/epoch-5000.pt` 已本地实际验证，payload `problem=Train_ALL`，epoch=5000。
- 两个 checkpoint 各有 193 个 state tensor，depot embedding `[128,2]`，customer embedding `[128,5]`；本地 strict load 到官方 MOEModel 均无 missing/unexpected keys，参数数 3,682,176。仅 CPU construction/load，无 forward。
- CVRP pickle loader 接 `(depot,loc,raw_demand,capacity)` 并 normalize；`load_problems` 接已 normalized 三元组，边界不同。VRPTW pickle 增加 `(service_time,tw_start,tw_end)`，loader 除 demand/capacity 后给六元组。不能把 normalized demand 再送 pickle loader。
- `VRPTWEnv` speed=1、depot TW 固定 `[0,3]`；这是官方环境默认值；真实benchmark depot上界已确认约4.6，speed=1。adapter 必须比较 source depot TW/service 单位，无法表达时先报告，不静默换数据。
- 官方 POMO/augmentation 搜索保存 `env.selected_node_list[B,P,T]`。需按相同 best POMO/augmentation 索引获取路线，不能只保存 rewards。`fine_tune_epochs` 必须维持 0。
- README 最小版本 Python>=3.8/PyTorch>=1.12；实际 Tester→utils 还需要 SciPy、tqdm 等。本地模型可 strict load，但 Tester import 因缺 SciPy 失败。服务器不能据此判缺包。
- 结论：本地 `MISSING_LEAF_DEPENDENCY`；模型与尺寸证据较强，仍不标记 READY。

## RF-TE

- 固定 `v0.4.0`；实际配置路径为 `configs/experiment/main/rf/rf-transformer-50.yaml` 与 `...-100.yaml`，不是省略 `main/` 的历史路径。构造 `RouteFinderBase` + `RouteFinderPolicy`，RMS norm、pre-norm、post-layers norm、SiLU gated MLP，继承 rfbase 的相应 `num_loc`。
- 官方 Hugging Face model repo `ai4co/routefinder` 的文件是 `checkpoints/50/rf-transformer.ckpt`、`checkpoints/100/rf-transformer.ckpt`。model repo revision 已记录；本地没有二进制，不能声称 hash/shape/load 通过。GitHub v0.4.0 release 同期引入 HF 下载说明，但逐权重训练 commit 与该 release 的一一对应仍需 checkpoint metadata 验证。
- Native NPZ/TensorDict `locs[B,N+1,2]`，reset 前 `demand_linehaul[B,N]`，reset 自动补 depot 0；time_windows/service_time 覆盖 depot。generator normalized demand/capacity=1；adapter 要显式传容量与所有约束，不依赖 defaults。
- `test.py` 使用 Lightning `load_from_checkpoint(... strict=False)`；必须记录 missing/unexpected keys 后判断是否真匹配 transformer，不能仅以未抛异常判定。policy `return_actions=True`，有 best_multistart_actions / best_aug_actions。objective 为环境 reward 的负距离，还需 independent recompute。
- Python>=3.10；pyproject 依赖 HF hub、rl4co，uv source 竟指 rl4co main，不是固定 API。需记录服务器实际 rl4co commit/version；pyvrp/ortools 是 optional solver extra，不应无依据当成 neural inference 必需依赖。
- 结论：`VERSION_RISK`；release/config/source checkpoint 信息明确，实际反序列化、架构匹配待验证。

## MoSES(CaDA)

- 仅 `pretrained_moses_model/cada/{50,100}/multilora_denseroute_sigmoid.ckpt`。`scripts/test_script.sh` 与 `test.py` 指向 `CadaMultiLoRAPolicy`，`model_name=cada_multilora`，sigmoid，trainable_layer=1、dynamic_topK=0、basis_variants=0、rank=[32]*5。`test.py` 还指定 rms、无 pre/post norm、sparse ratio=0.5。
- 官方 loader 取 `torch.load(...weights_only=False)['state_dict']` 的 `policy.*` keys 再 `load_state_dict`；当前调用 strict=False，后续必须审计遗漏键。已经训练好的 LoRA/experts 可以加载推理，不能因 LoRA 名称误启动 fine-tuning。
- 本地两文件是完整 zip binary，有 SHA256。pickle 静态 GLOBAL 包含 CadaMultiLoRAPolicy、CadaEncoder、GateLayer、LoRALayer，以及 MTVRPEnv/TorchRL spec/OmegaConf 对象；不是仅 tensor dict。加唯一官方 repo 路径后本地 CPU torch.load 因缺 rl4co 失败。静态 opcode inspection 不等于反序列化或 shape 验证。
- 输入与当前 RF 分支相近：`locs` depot-inclusive，但 reset 前 demands customer-only，reset 自动补 depot；不能复用 CaDA 已含 depot 的需求 tensor。CVRPTW 必须保留 benchmark TW/service/speed。输出支持 return_actions 与 best start/augmentation 对应 actions。
- README 建议 Python3.10/Torch2.6，requirements 仅 rl4co/vrplib，不能代表完整 runtime lock。checkpoint 序列化字符串显示 Lightning 2.5.0.post0，并引用 TorchRL spec 类；这提示 API/反序列化版本风险，不足以要求升级服务器 Torch。
- 结论：`VERSION_RISK`，并有本地 missing rl4co。无需下载更多 MoSES 权重。

## NeuOpt CVRP

- 当前官方 `pre-trained/cvrp50.pt` 与 `cvrp100.pt` 均存在；两者 SHA256、CPU deserialize、官方 Actor strict load 已本地验证。actor 685,140 parameters，CVRP customer feature embedding `[64,8]`。
- `CVRPDataset.make_instance` 输入 raw `(depot,loc,demand,capacity)`，内部 demand/capacity。README dummy_rate=0.4（50→70 nodes）、0.2（100→120 nodes）；不能把 100 继续按 70-node successor 解码。`options.py` 的 `wo_MDP` 默认 **True**，Actor 构造必须依照此值。
- 搜索为官方 PPO rollout + flexible k-opt/GIRE/D2A。solution 是 successor graph，要提取 best feasible incumbent，检查单一 successor cycle 并将 dummy depot 映射为真实 depot，非最终一步任意 infeasible state。
- 历史 batch=1 `.squeeze()` 问题、protobuf Python backend workaround 保留在迁移设计；此轮未跑旧 smoke。旧本地 `our_5.json` 是 adapter metadata，不能拿来冒充本次 server model-result JSON。
- 依赖：torch、NumPy、tqdm、tensorboard_logger/protobuf。本地 Actor 可加载，`agent.ppo` import 因缺 tensorboard_logger 失败；cp311_base 未实测。
- 结论：`LIKELY_COMPATIBLE`（基于官方尺寸与本地 strict load），不是 READY；CVRP50 后续迁移旧已验证实现，CVRP100 新增独立验证。

针对rrc与lib tester的进一步区别和GLOP显式API参数配置，以 [SIZE_COMPATIBILITY.md](SIZE_COMPATIBILITY.md) 为准。各method/problem README已固定独立adapter边界；实现均为NOT_IMPLEMENTED。
