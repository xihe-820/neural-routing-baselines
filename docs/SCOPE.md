# 范围与证据

正式size只有50/100；TSP为节点数，CVRP/CVRPTW为customer数（不含depot）。TSP仅GLOP/UDC；CVRP七方法；CVRPTW仅CaDA/MVMoE/4E/RF-TE/MoSES(CaDA)，共26 rows。RF-TE固定Transformer，MoSES固定CaDA backbone，MVMoE固定MOE/4 experts。NeuOpt只迁移CVRP，历史结果不继承为新integration成功。

本轮收口：已有source/asset审计、六组真实公开benchmark、problem-level independent validators与测试、method-specific接口设计、GLOP/UDC尺寸配置分析、可运行audit scripts与runbook。完成后停在checkpoint report；不一次实现26 rows。

No training/fine-tuning/EAL/LoRA training；只用official pretrained assets。不得修改checkpoint、截断/插值权重、用随机权重冒充正式模型、改成500/1000。official source只读；无commit/push。不得整体安装legacy requirements。核心Torch/CUDA/NumPy/PyG/rl4co/TensorDict/TorchRL/Lightning更换需先报告；明确必要且低风险的叶子依赖允许记录后安装，本轮实际没有安装任何包。

工作模式：local development + user-executed server validation。已有本地CPU/GPU均可用于合法小测试；不为此重装环境。服务器由用户手动执行并回传JSON；不需要SSH。所有运行路径由CLI/config/env传入。

| Evidence | Meaning |
|---|---|
| SOURCE_CONFIRMED | Official source/README/release/HF confirms intended asset or behavior; no execution claim |
| LOCAL_VERIFIED | Actual WSL inspection/execution with recorded environment and artifact |
| SERVER_VERIFIED | User-returned server execution evidence; currently none |
| BLOCKED | Specific technical failure, with cause; not merely unrun work |
| NOT_APPLICABLE | Check does not apply |
| NOT_RUN / NOT_IMPLEMENTED | Separate execution/implementation states, not evidence of failure |

未来completion的核心是actual decoded solution + independent feasibility + independent objective。Kit为secondary validation，能跑则双重核验；版本/encoding障碍单独记录。官方dataset smoke有价值但不是绝对gate，合法tiny native-format测试可替代入口sanity；不能用随机数据替代正式ML4CO benchmark。最少合法batch先验证，约5实例做稳定sanity；our_2/our_5只是命名约定，NeuOpt避免batch1。
