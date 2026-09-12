# Upstream audit

本地审计时间：2026-09-11T09:09:44.148888+00:00。根目录：`/home/xihe/projects/neural-routing-baselines/external`。七个 checkout 均已实际检查 HEAD、branch、origin、status、tags；原始证据 `artifacts/audit/assets.json` 与 `remote_refs.json`。

| Method | URL | Actual commit / recommended pin | Branch | Dirty | Local/latest remote tag |
|---|---|---|---|---|---|
| GLOP | https://github.com/henry-yeh/GLOP | `e540bc0153a0598e923e35116deeaecaf9c1cfff` | master | no | 无 advertised tags |
| UDC | https://github.com/CIAM-Group/NCO_code | `274df3c4975384592b60fe7f79fbb2441ce11c15` | main | no | 无 advertised tags |
| CaDA | https://github.com/CIAM-Group/CaDA | `b9868e1e09b3a1a3754960d830e801e51c7cb38d` | main | no | 无 advertised tags |
| MVMoE | https://github.com/RoyalSkye/Routing-MVMoE | `af29e5af0595f94f3ecc3bc46d72df1089a62682` | main | no | 无 advertised tags |
| RF-TE | https://github.com/ai4co/routefinder | `fe0e45b6df118af03c5f42db8b93a351f7629131` | DETACHED | no | v0.4.0 |
| MoSES(CaDA) | https://github.com/panyxy/moses_vrp | `e301478b7a5df6d7b0b10a0543f4dee5e3c027a8` | main | no | 无 advertised tags |
| NeuOpt | https://github.com/yining043/NeuOpt | `ccf6b5f0f6a8fda2792b4be11d4ec35390a8139b` | main | no | 无 advertised tags |

`git ls-remote` 成功核实：除 RF-TE 外，本地 commit 与远端 HEAD 一致；RouteFinder 远端 main 是 `cc3ab078650e8db62c0562a60f421defe38d5ae3`，本地为 detached v0.4.0。未 fetch 或 checkout。GitHub releases API 返回 403 rate limit，六个无 tag 仓库不能据此声称“无任何 release”；latest release 未核实。RF v0.4.0 页面已独立核实。

Pin rationale：

- **GLOP**：保留官方 random-insertion 修复版本；与此前审阅 commit 及远端 HEAD 一致。
- **UDC**：保留当前 monorepo commit，仅审计指定两个 AGNN-ICAM 子树；与此前审阅及远端 HEAD 一致。
- **CaDA**：保留已审阅官方 test-only 版本；HEAD learning-rate default 修改不要求迁移 test checkpoint。
- **MVMoE**：同一 commit 随附官方 4E n50/n100 binary，当前实际 strict load 匹配。
- **RF-TE**：保留 v0.4.0 release；README 提供对应 HF 下载入口。暂不跟随远端 main；单个 checkpoint 的训练 commit/架构仍待加载确认。
- **MoSES(CaDA)**：同一 commit 随附目标 CaDA sigmoid 权重及测试脚本；保持代码和资产共同 provenance。
- **NeuOpt**：保留已验证历史版本，同一 commit 内含 CVRP50/100 官方权重。

[RouteFinder v0.4.0 official release](https://github.com/ai4co/routefinder/releases/tag/v0.4.0)；[HF official model repo](https://huggingface.co/ai4co/routefinder)。HF repository revision 与 Git commit 是不同 provenance，不可互相替代。

服务器 checkout 路径、commit、dirty 均为 `NOT_RUN`。上述 pins 是本地审阅建议，未更改任何服务器代码。
