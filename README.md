# Neural Routing Baselines

独立、可复现、可审计的 neural routing baseline 接入仓库。使用作者官方代码与 pretrained checkpoint，在 ML4CO-Bench 的 **50 / 100** 实例上进行真实推理、提取解、独立验证约束与目标值，并尽量用 ML4CO-Kit 二次核验。

Phase 0 / Phase 1 snapshot 固定为 `38f794fc19c3aaa988bfa78db0ae51143aab93ef`；首个已提交的 **MVMoE/4E + CVRP50** integration 为 `54271cf4f97d7bd99784f6e6ee5ce52ee9f5994b`。当前工作树已把同一个adapter扩展到CVRP100；50/100均完成本地前5例 official search/decode configuration smoke、实际解提取、独立验证与Kit二次验证。CVRP100扩展尚未commit/push。

工作模式为 **local development + user-executed server validation**：Codex 在 WSL 开发；用户从 GitHub checkout 经批准的项目 commit 后手动运行服务器命令，再回传原始 JSON。服务器当前版本、路径、GPU、checkpoint 和数据一律等待实际证据。历史 NeuOpt CVRP50 成功记录保留为迁移背景，不直接继承为新仓库 PASS。

| Method | TSP | CVRP | CVRPTW |
|---|---|---|---|
| GLOP | 50 / 100 | 50 / 100 | — |
| UDC | 50 / 100 | 50 / 100 | — |
| CaDA | — | 50 / 100 | 50 / 100 |
| MVMoE/4E | — | 50 / 100 | 50 / 100 |
| RF-TE | — | 50 / 100 | 50 / 100 |
| MoSES(CaDA) | — | 50 / 100 | 50 / 100 |
| NeuOpt | — | 50 / 100 | — |

共 13 个 method/problem integration、26 个独立 size 状态。

- [本轮审计报告](docs/AUDIT_REPORT.md)：本地证据与服务器待验证项。
- [MVMoE/4E + CVRP50 integration report](docs/MVMOE_CVRP50_REPORT.md)：首个真实本地端到端结果。
- [MVMoE/4E + CVRP100 integration report](docs/MVMOE_CVRP100_REPORT.md)：同一adapter的第二个正式size。
- [正式范围](docs/SCOPE.md)、[长期状态矩阵](docs/STATUS.md)。
- [工程设计](docs/DESIGN.md)、[输入输出与依赖审计](docs/NATIVE_IO_AUDIT.md)、[validator 设计](docs/VALIDATION_DESIGN.md)。
- [服务器操作手册](docs/SERVER_RUNBOOK.md)：用户手动执行，默认不安装或更换依赖。
- [本地环境报告](docs/ENVIRONMENT_AUDIT.md)、[checkpoint 清单](docs/CHECKPOINT_AUDIT.md)、[数据报告](docs/DATASET_AUDIT.md)。

官方仓库放在被忽略的 `external/`，也可通过 CLI 或环境变量指向现有 checkout。主仓库只维护审计脚本、adapters、validators、provenance 与文档。不提交官方源码、模型、数据、日志或结果。

```bash
# 使用已有含NumPy的解释器；真实reference测试另需已有ML4CO-Kit
python -B -m unittest discover -s tests -v
python3 -B scripts/audit_assets.py --upstream-root external
```

CPU checkpoint 反序列化需要当前解释器已有 PyTorch；使用 `--load-checkpoints` 显式启用。脚本对受信任官方 pickle 使用 `weights_only=False`，不修改文件、不加载优化器进行训练、不执行推理。报告中的成功反序列化不等于模型匹配、CUDA forward 或 benchmark 验证成功。

No training / no fine-tuning；当前开发轮未经明确要求不自动commit/push，official source read-only。所有smoke runtime仅用于工程诊断，**NOT paper-comparable runtime**。
