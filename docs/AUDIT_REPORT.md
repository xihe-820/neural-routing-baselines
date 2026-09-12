# Phase 0 / Phase 1 checkpoint report

本轮按修正后的 local development + user-executed server validation 收口；已完成本地基础层与明确接口，停在审查点，不开始全部26 rows实现。以下“事实”均指本轮实际证据；“推断”仅指代码支持的候选；服务器没有返回执行证据。

## A. git diff / 新增文件

独立仓库 `/home/xihe/projects/neural-routing-baselines` 保留已有工作。当前为unborn分支、文件尚未tracked，因此`git diff --stat`为空；`git status --short`显示自有文件为untracked，不是工作丢失。本轮没有git add/commit/push，七个official checkout最后审计dirty=false。

新增/完善：五个audit/download脚本；SHA与数值输入检查；TSP/CVRP/CVRPTW各自objective/validator；三个测试模块；六组dataset清单；组件式checkpoint/完整strict-load记录；26行细粒度status；13个method/problem接口README；源码/环境/数据/尺寸/runbook文档。实际文件树见B。下载数据与外部源码/权重/原始artifact保持gitignored。

## B. repository tree

见 [REPOSITORY_TREE.md](REPOSITORY_TREE.md)，只展示主仓库自有文件与ignored目录摘要，不展开official monorepo。共享数学真值在problems/，method输入、decoder和runner的接口在methods/；后三者尚未实现，不放成功stub。

## C. 26-row status

完整19列×26行见 [STATUS.md](STATUS.md)，机器可读为 [status.json](../manifests/status.json)。全部row：source audit LOCAL_VERIFIED；asset官方来源SOURCE_CONFIRMED；本地dataset与problem validator LOCAL_VERIFIED；adapter/decoder NOT_IMPLEMENTED；model forward、small inference、method-output Kit、所有server项NOT_RUN。以下列出逐size的资产/加载差异，不能将reference结果继承为method成功。

| Method | Problem | N | Local binary/SHA | Official strict load | Configuration / current limit |
|---|---|---:|---|---|---|
| GLOP | TSP | 50 | NOT_RUN | NOT_RUN | Code-supported candidate; components/args.json absent |
| GLOP | TSP | 100 | NOT_RUN | NOT_RUN | Code-supported candidate; components/args.json absent |
| GLOP | CVRP | 50 | NOT_RUN | NOT_RUN | Code-supported candidate; components/args.json absent |
| GLOP | CVRP | 100 | NOT_RUN | NOT_RUN | Code-supported candidate; components/args.json absent |
| UDC | TSP | 50 | NOT_RUN | NOT_RUN | Default path blocked at 50 |
| UDC | TSP | 100 | NOT_RUN | NOT_RUN | Single-block static candidate; components absent |
| UDC | CVRP | 50 | NOT_RUN | NOT_RUN | Default path blocked at 50 |
| UDC | CVRP | 100 | NOT_RUN | NOT_RUN | Single-block static candidate; components absent |
| CaDA | CVRP | 50 | NOT_RUN | NOT_RUN | Checkpoint absent; legacy API risk |
| CaDA | CVRP | 100 | NOT_RUN | NOT_RUN | Checkpoint absent; legacy API risk |
| CaDA | CVRPTW | 50 | NOT_RUN | NOT_RUN | Checkpoint absent; legacy API risk |
| CaDA | CVRPTW | 100 | NOT_RUN | NOT_RUN | Checkpoint absent; legacy API risk |
| MVMoE | CVRP | 50 | LOCAL_VERIFIED | LOCAL_VERIFIED | Tester missing scipy in gan; no forward |
| MVMoE | CVRP | 100 | LOCAL_VERIFIED | LOCAL_VERIFIED | Tester missing scipy in gan; no forward |
| MVMoE | CVRPTW | 50 | LOCAL_VERIFIED | LOCAL_VERIFIED | Tester missing scipy in gan; no forward; depot TW mapping required |
| MVMoE | CVRPTW | 100 | LOCAL_VERIFIED | LOCAL_VERIFIED | Tester missing scipy in gan; no forward; depot TW mapping required |
| RF-TE | CVRP | 50 | NOT_RUN | NOT_RUN | Transformer asset source confirmed; binary absent |
| RF-TE | CVRP | 100 | NOT_RUN | NOT_RUN | Transformer asset source confirmed; binary absent |
| RF-TE | CVRPTW | 50 | NOT_RUN | NOT_RUN | Transformer asset source confirmed; binary absent |
| RF-TE | CVRPTW | 100 | NOT_RUN | NOT_RUN | Transformer asset source confirmed; binary absent |
| MoSES(CaDA) | CVRP | 50 | LOCAL_VERIFIED | BLOCKED | Deserialize missing rl4co; no shape/load claim |
| MoSES(CaDA) | CVRP | 100 | LOCAL_VERIFIED | BLOCKED | Deserialize missing rl4co; no shape/load claim |
| MoSES(CaDA) | CVRPTW | 50 | LOCAL_VERIFIED | BLOCKED | Deserialize missing rl4co; no shape/load claim |
| MoSES(CaDA) | CVRPTW | 100 | LOCAL_VERIFIED | BLOCKED | Deserialize missing rl4co; no shape/load claim |
| NeuOpt | CVRP | 50 | LOCAL_VERIFIED | LOCAL_VERIFIED | Actor matched; PPO missing tensorboard_logger; no forward |
| NeuOpt | CVRP | 100 | LOCAL_VERIFIED | LOCAL_VERIFIED | Actor matched; PPO missing tensorboard_logger; no forward |

## D. 六组 ML4CO-Bench data

**事实：六文件均从对应官方ML4CO-Bench-101-SL源下载或按官方SHA复用并实际解析，合计67,877,012 bytes。** [DATASET_AUDIT.md](DATASET_AUDIT.md)包含六个source URL、filename、完整SHA256、bytes、count、schema与实际reference objective；[datasets.yaml](../manifests/datasets.yaml)冻结dtype/shape/属性和Kit task源码SHA。

TSP50/100各1280例；CVRP50/100各10000例；CVRPTW50/100各1000例。均为list of Kit tasks。TSP points[N,2]/closed ref_sol；VRP单depot、raw demands[N]、capacity40/50；TW与service包含depot。六文件全部Euclidean不取整；CVRPTW从time0出发、speed1、等待至下界、service-start在窗内、departure加service、回depot也检查上界，depot TW约[0,4.6]。没有从baseline默认env臆造这些语义。

## E. Independent validators / tests

**LOCAL_VERIFIED：32个tests通过**，其中26个handcrafted validator cases、5个audit失败分支、1个包含六文件×前5例的真实reference测试（30实例）。范围覆盖TSP single cycle/exactly-once/invalid IDs；CVRP一条或多条路线、重复/缺失、raw容量；CVRPTW等待、边界、迟到、service导致后续违规、回depot及CVRP约束。NumPy独立实现，无baseline/Kit objective调用。

真实reference同时通过Kit可行性与distance复核（rtol1e-6/atol1e-6）。CVRPTW time tolerance遵循实际task threshold。不是全量benchmark参考解校验，也没有模型实际解。原始测试输出为`artifacts/audit/final_tests.txt`；命令、结果与artifact SHA持久化于 [local_verification.json](../manifests/local_verification.json)。API与边界见 [VALIDATION_DESIGN.md](VALIDATION_DESIGN.md)。

## F. Local environment facts

**LOCAL_VERIFIED：WSL可见RTX4060 Laptop GPU（8188MiB），gan Torch2.5.1 CUDA12.1可发现GPU。** 先前sandbox NVML拒绝不等于host无GPU；driver591.86，nvidia-smi CUDA13.1为driver支持上限。没有执行model GPU forward。

strict load使用已有gan：Python3.10.20 / Torch2.5.1 / NumPy1.24.3。数据与reference tests使用已有ml4co_venv：Python3.10.20 / Torch2.12.0+cpu / NumPy2.2.6 / Kit0.5.4。两套证据分开记录。gan pip check=0不等于七个baseline依赖齐全。无package安装、核心stack替换、新建环境；Kit既有Concorde setup.py修改未动。详见 [ENVIRONMENT_AUDIT.md](ENVIRONMENT_AUDIT.md)。

## G. Checkpoints already LOCAL_VERIFIED

MVMoE/4E n50/n100：binary非LFS、SHA、torch.load、官方MOEModel strict=True均通过；各3,682,176 parameters、无missing/unexpected。NeuOpt CVRP50/100：同级别证据，官方Actor各685,140 parameters、seq70/120、with_simpleMDP=True。均CPU执行，已成功的四次strict load未为整理文档重跑。

MoSES(CaDA)50/100：只达到完整binary/SHA LOCAL_VERIFIED；deserialize因缺rl4co BLOCKED，不能声称state shape或architecture匹配。六份完整SHA/bytes/source及四次exact command/model config见 [CHECKPOINT_AUDIT.md](CHECKPOINT_AUDIT.md) 和 [checkpoints.yaml](../manifests/checkpoints.yaml)。server副本尚未比较。

GLOP/UDC/CaDA/RF-TE官方资产入口与intended role为SOURCE_CONFIRMED，本地未落盘。GLOP/UDC公开Drive页面可读，不声称需要权限或链接失效；包内尚未定向获取组件。manifest按target_size + required_official_assets + asset_role + size_specific + compatibility_basis组织，不强求n50/n100命名权重。

## H. GLOP 50/100 compatibility

**事实**：Reviser100在TSP50违反官方长度assert；TSP100有显式小规模分支。**静态候选**：TSP50用50/20（iters25/5）；TSP100用100/50/20（iters20/25/5）。这是官方CLI可表达的配置，不能称为README paper setting或已执行成功。

CVRP50/100默认K_SPARSE键缺失。官方load_partitioner/infer允许显式k_sparse，可保留原partitioner+Reviser20，用k50/k100作为候选；learned layer shapes静态不绑定全局N/k，但图邻域随k改变须记录。未找到official no-partition小规模路径；不bypass partitioner。**待验证**：目标组件及args.json、strict keys、actual forward与解捕获。详见 [SIZE_COMPATIBILITY.md](SIZE_COMPATIBILITY.md)。

## I. UDC 50/100 compatibility

**事实**：主要rrc路径sub_size100时，TSP50 hardcoded topk100越界；CVRP50只有51点也无法topk100，初始整块循环为0。不能认定N<=sub_size会自然退化为单子问题。**静态候选**：N100满足topk与单个100-customer/node block局部条件，但不等于权重forward通过。

另查官方lib tester：TSP仍hardcode topk100；CVRP存在ceil/padding处理，因此不能声称所有UDC代码都不处理小规模。但CVRP50默认k仍越界、完整padding执行未验证。当前TSP50未找到完整不改语义的合法配置；这不是证明其权重architecture必然绑定N>=100。后续先100，再单独报告50保真方案。目标两个子树均缺partition/conquer checkpoint，PyG缺失另阻碍partition import。

## J. PENDING SERVER VERIFICATION（各字段当前NOT_RUN）

服务器当前Python/Torch/CUDA/包状态、pip check、RTX4090真实可见性；服务器使用的dataset/checkpoint SHA与本地manifest是否一致；所选upstream commit/dirty；server imports/load、CUDA forward、实际small inference/independent validation、最终benchmark。历史cp311_base版本与路径仅为背景，不是当前事实。

[SERVER_RUNBOOK.md](SERVER_RUNBOOK.md)提供现有可复制audit命令与SHA比较；GPU/inference待对应wrapper实现才补命令。用户手动迁移执行并回传原始JSON，本地任务不因没有SSH停止。

## K. 真正 blockers / 未实现项

- UDC50默认topk/block限制有代码证据；GLOP CVRP默认K_SPARSE缺键需显式配置候选验证。
- GLOP/UDC/CaDA/RF-TE所选components本地未落盘；公开来源已知，当前不是已证实的权限障碍。
- gan缺PyG/random_insertion等阻碍相应导入；MoSES deserialize缺rl4co；CaDA/RF/MoSES的rl4co/TensorDict/TorchRL/Lightning API兼容仍有风险，未擅自改核心stack。
- MVMoE Tester缺scipy、NeuOpt PPO缺tensorboard_logger只阻碍对应entrypoint，未阻塞source审计或已成功strict load。
- MVMoE CVRPTW默认depot3.0与benchmark4.6需明确数据配置方案；不能静默修改benchmark。
- Method adapter/decoder/runner尚未实现是本轮有意停止的位置，不能写成失败或integration complete。官方数据smoke/Kit不是绝对门槛。

## L. 下一步第一个 implementation

推荐 **GLOP / TSP / 100**，保持既定优先级：先定向取得Reviser100/50/20与args.json，核对原始keys，运行最小forward，设计无歧义ID捕获，再接真实ML4CO少量实例和独立验证。随后GLOP/TSP50验证50/20组合。此建议是工程排序推断，不是已经执行的计划结果。

若GLOP组件暂时拿不到，可考虑 **MVMoE/4E / CVRP / 50**：已有strict-load与dataset证据，CVRP没有TW映射差异。当前按要求停在此checkpoint，未自动启动两者的完整实现。
