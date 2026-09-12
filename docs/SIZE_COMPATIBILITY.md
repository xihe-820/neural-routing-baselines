# GLOP / UDC：50/100 合法配置审计

本文件区分：作者 README 的 documented setting、未在 README 报告但官方代码参数/API 可表达的 configuration、以及仍缺权重执行证据的推断。没有训练、改权重、改官方源码或将目标改为500/1000。

## GLOP

| Target | 已确认事实 | 合法配置判断 | 未验证 |
|---|---|---|---|
| TSP50 | `utils/functions.py:reconnect` 行308拒绝 Reviser100；decomposition 行176–188支持余数保留 | 官方 CLI 可用 `revision_lens=[50,20]`；相应 `revision_iters=[25,5]` 可保留原50/20阶段预算。属于 code-supported configuration，不称作 paper setting | 官方 Reviser50/20 + args.json 尚未获取，checkpoint/load/forward 未运行 |
| TSP100 | main.py 行63有 <=100 分支；README 已有 TSP100 实验；100/50/20均通过长度条件 | 官方 CLI 可表达 `revision_lens=[100,50,20]`, `revision_iters=[20,25,5]`。README 的 TSP100 cross-distribution 示例另带 Reviser10 与不同预算，二者应明确区分 | 完整 reviser 配置加载/forward 未运行 |
| CVRP50 | 默认 K_SPARSE[50] 缺失；51个节点不能照搬k100/300；官方保持 partition+sampler+revision | `load_partitioner` 和 `infer` 都支持显式 `k_sparse`，可在 wrapper 中调用官方函数，使用如 k50、原官方 CVRP partitioner + Reviser20。代码允许传参且权重矩阵形状不绑定k/N；不是已验证 benchmark 配置 | 官方 partitioner strict load、k50的实际forward、quality/fidelity检查 |
| CVRP100 | 默认 K_SPARSE[100] 缺失；101个节点可容纳k100 | 显式 `ckpt_path` + `k_sparse=100` 调用官方函数，保留 partitioner 与 Reviser20，是 code-supported candidate；不能只给 main.py 一个ckpt_path | 同上，且checkpoint身份待确认 |

CVRP 候选并不建议“去掉 partitioner”：`problems/cvrp.py:init` 行33–53通过 partition heatmap/sampler形成 routes。没有找到该 CVRP method 的官方 no-partition 小规模配置；直接 bypass 会改变方法。原始 raw demand/capacity 可直接传给官方 infer/sampler，图输入内部做一次 normalization。

`nets/partition_net.py:EmbNet/ParNet/Net` 的 learned layers按 feature、hidden、depth构造。k_sparse保存在模块属性并用于 reshape；改变k不会需要截断/插值权重，但会改变邻域图，必须作为 inference configuration 明确记录，随后以原checkpoint执行验证。现在只确认 API 支持和静态形状推断，不能标成 checkpoint compatibility LOCAL_VERIFIED。

TSP 任意合法revision组合不是“随便换算法”：它使用官方暴露的多级 reviser 参数，并保留每个启用阶段的实现。最终 choice需在获得配套资产后锁定；不以未运行结果声称正式完成。Reviser100→50的跨长度复用没有必要，直接使用作者Reviser50组件。

本地已实际 import `nets.attention_local`；partitioner import 缺PyG、random insertion缺random_insertion。官方公开Drive页面可读，标题为 `pretrained.zip`，但并未选择性提取其中组件；本轮不镜像整个多任务资产包。资产存在性源证据与本地未落盘事实分别记录。

## UDC

只使用 `single_objective/UDC-Large-scale-CO-master/UDC/TSP-AGNN-ICAM` 与 `CVRP-AGNN-ICAM`。

| Target | 主要官方 rrc 路径 | 结论 |
|---|---|---|
| TSP50 / sub_size100 | graph helper `topk(k=100)` 对50点越界；conquer `.view(...,100)`不能形成正确单个50点子问题 | 默认路径 BLOCKED；不能声称 N<=sub_size 自然退化 |
| TSP100 / sub_size100 | topk100合法，100-node block尺寸合法 | code-supported single-block candidate；缺双checkpoint的strict-load/forward证据 |
| CVRP50 / sub_size100 | graph默认k100对51点越界；rrc初始化 `range(data.size(1)//100)` 为0，随后customer解不能正确分块 | 默认路径 BLOCKED；不是模型权重结构已证明不支持50 |
| CVRP100 / sub_size100 | topk100对101点可行，100个customer可形成单block | code-supported candidate，保留partition与conquer；缺实际权重验证 |

另外检查了同目录官方 lib tester，避免仅依据默认rrc下结论：

- `TSPTesterrrclib.py:gen_pyg_data` 行249仍写死k100；`_test_one_batch`行186附近使用 `solving_length=(N//sub_size)*sub_size`，N50/sub100得到0长度，不是自动改成50。仅换lib入口不能解决TSP50。
- `CVRPTesterlib.py:_load_init_sol` 行365附近有ceil风格分块、全客户访问后break；后续行414/437/463有padding逻辑。**因此“所有UDC代码都无小于100的处理”是错误结论**。但默认图仍用k100，且conquer先构造0个完整block，后续padding路径尚未实测；不能用这些局部padding代码宣称CVRP50已可用。
- CVRP的 `gen_pyg_data(...k_sparse=...)` 可接参数；TSP helper的k100写死，单改k_sparse形参无效。改变sub_size是配置能力，但并未消除所有50-node graph路径限制。当前未找到完整、无需语义修改的TSP50官方配置。
- 两个官方模型的 Linear 参数没有明显global-N维度；本地 TSPModel/CVRPModel import成功，PartitionModel因缺PyG失败。随机构造模型不能代替官方权重forward，所以不做随机权重“兼容成功”证明。

官方Drive页面标题为 `UDC-pretrained model & test sets.rar`；本地两个子树没有目标checkpoint。TSP要求partition/conquer epoch460、CVRP要求epoch230；这些是loader配置，不应强制寻找独立n50/n100命名文件。需要该bundle中的对应组件后继续actual shape/load audit。

## 建议的下一步

保持第一优先 **GLOP / TSP / 100**：代码已有明确小规模分支、尺寸条件最清晰，先取得三份reviser及args.json，再做最小forward和实际解捕获。随后TSP50验证code-supported 50/20配置。CVRP与UDC50需把参数调整/代码限制单独记录。

若短期拿不到GLOP组件，已有模型与dataset均LOCAL_VERIFIED的 **MVMoE/4E / CVRP / 50** 是可提出的替代起点；本轮未因此擅自开始完整adapter/inference。MVMoE CVRPTW默认depotTW3.0与benchmark4.6不一致，不能作为第一个无障碍接入任务。
