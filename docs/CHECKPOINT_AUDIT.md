# Checkpoint audit

以下严格区分 `SOURCE_CONFIRMED`、`LOCAL_VERIFIED` 与 `SERVER_VERIFIED`。MVMoE n50/n100服务器副本已在CVRP运行前匹配固定SHA；其它方法仍为`NOT_RUN`。完整本地state tensor key/shape存在忽略的`artifacts/audit/assets.json`。

| Method | Size | Official relative file / component | WSL actual evidence | Server |
|---|---|---|---|---|
| GLOP | 50 | `TSP candidate revisers 50+20; CVRP: large-scale partitioner + Reviser20，未确认50/100专用权重` | NOT FOUND in inspected local checkout; remote/server existence not negated | NOT_RUN |
| GLOP | 100 | `TSP candidate revisers 100+50+20; CVRP: large-scale partitioner + Reviser20，未确认50/100专用权重` | NOT FOUND in inspected local checkout; remote/server existence not negated | NOT_RUN |
| UDC | 50 | `single_objective/UDC-Large-scale-CO-master/UDC/TSP-AGNN-ICAM/checkpoint-tsp-460.pt / single_objective/UDC-Large-scale-CO-master/UDC/TSP-AGNN-ICAM/checkpoint-partition-460.pt` | NOT FOUND in inspected local checkout; remote/server existence not negated | NOT_RUN |
| UDC | 100 | `single_objective/UDC-Large-scale-CO-master/UDC/TSP-AGNN-ICAM/checkpoint-tsp-460.pt / single_objective/UDC-Large-scale-CO-master/UDC/TSP-AGNN-ICAM/checkpoint-partition-460.pt` | NOT FOUND in inspected local checkout; remote/server existence not negated | NOT_RUN |
| CaDA | 50 | `50/result/2024-1111-1139/checkpoint-300.pt` | NOT FOUND in inspected local checkout; remote/server existence not negated | NOT_RUN |
| CaDA | 100 | `100/result/2024-1121-1355/checkpoint-300.pt` | NOT FOUND in inspected local checkout; remote/server existence not negated | NOT_RUN |
| MVMoE | 50 | `pretrained/mvmoe_4e_n50/epoch-5000.pt` | binary/hash + CPU deserialize + official strict model load | SERVER_VERIFIED |
| MVMoE | 100 | `pretrained/mvmoe_4e_n100/epoch-5000.pt` | binary/hash + CPU deserialize + official strict model load | SERVER_VERIFIED |
| RF-TE | 50 | `checkpoints/50/rf-transformer.ckpt` | NOT FOUND in inspected local checkout; remote/server existence not negated | NOT_RUN |
| RF-TE | 100 | `checkpoints/100/rf-transformer.ckpt` | NOT FOUND in inspected local checkout; remote/server existence not negated | NOT_RUN |
| MoSES(CaDA) | 50 | `pretrained_moses_model/cada/50/multilora_denseroute_sigmoid.ckpt` | binary/hash; CPU deserialize fails: missing rl4co | NOT_RUN |
| MoSES(CaDA) | 100 | `pretrained_moses_model/cada/100/multilora_denseroute_sigmoid.ckpt` | binary/hash; CPU deserialize fails: missing rl4co | NOT_RUN |
| NeuOpt | 50 | `pre-trained/cvrp50.pt` | binary/hash + CPU deserialize + official strict model load | NOT_RUN |
| NeuOpt | 100 | `pre-trained/cvrp100.pt` | binary/hash + CPU deserialize + official strict model load | NOT_RUN |

## 实际 materialized binary

以下六个目标文件均不是 LFS pointer，format 为 PyTorch zip archive。SHA 是 WSL 文件 SHA，不能替代服务器副本的 hash。

| Method / size | Bytes | SHA256 |
|---|---:|---|
| MVMoE: `pretrained/mvmoe_4e_n100/epoch-5000.pt` | 44536097 | `554d6daea825e17d62c1b9db40d56869312923848504bdf4970663673c971bdc` |
| MVMoE: `pretrained/mvmoe_4e_n50/epoch-5000.pt` | 44533389 | `3417f302fbddf232fd19a2a886cd1c7f44290b6d8c7280fcb0ae3777eeed3192` |
| MoSES(CaDA): `pretrained_moses_model/cada/100/multilora_denseroute_sigmoid.ckpt` | 54585022 | `2eac9b038ae4655581aa73e4dbe8ad529aefd1963368c9a92d254b6269f8aabf` |
| MoSES(CaDA): `pretrained_moses_model/cada/50/multilora_denseroute_sigmoid.ckpt` | 54581822 | `1aa499f3fce5d3412c2544c9632bbb9709309a7299b4298b535fa7e9014ef803` |
| NeuOpt: `pre-trained/cvrp100.pt` | 10633669 | `502a5904182306c1a3f65f7b1a8503a609ff2a044db054abcf93690af36594fb` |
| NeuOpt: `pre-trained/cvrp50.pt` | 10630323 | `1cd201ca47888e51068a157389460641c81d71054f064d9c8ea1743312289e3a` |

## Architecture 与配置证据

- MVMoE 两份均 epoch=5000、problem=Train_ALL；193 state tensors；官方 MOEModel strict=True、3,682,176 parameters、无 missing/unexpected keys。Test config 明确 MOE/4E，而非 MOE_LIGHT。
- NeuOpt 两份含 actor/critic/optimizer/RNG；分别使用 seq_length 70/120 构造官方 Actor，strict=True、685,140 parameters、无缺键。未执行 PPO/inference。
- MoSES 两份 pickle 静态可见 CadaMultiLoRAPolicy/CadaEncoder/LoRALayer 与 environment、TorchRL spec、OmegaConf 对象。`static_pickle` 仅 opcode inspection；实际反序列化在唯一官方 repo namespace 下因 rl4co 缺失失败，不能推导 state tensor shape 已验证。
- CaDA 官方 README + trainer.py 定义 50/100 两份 epoch300 multi-task checkpoint。本地两文件均未找到；官方 HF checkpoint.zip 未下载。
- RF-TE HF model API 在 revision `4edf73d68a5915787dd2f81ba177d77e13317190` 列出 50/100 rf-transformer.ckpt；本地未 materialize。路径与 v0.4.0 README 一致，但 source release 不足以证明 checkpoint 的训练 commit。
- GLOP reviser args.json 与权重必须配套；不能声明 TSP50 可用 Reviser100。CVRP使用partitioner组件而非强制每个target size一份文件；默认K_SPARSE无50/100，显式参数候选详见SIZE_COMPATIBILITY.md。
- UDC 两个目标子树没有 checkpoint。官方 TSP 默认文件 epoch460、CVRP epoch230；只确认官方 loader 命名及下载入口，不确认下载包内部布局或独立50/100训练权重。

## 官方来源

完整 path/config/用途见 [manifest](../manifests/checkpoints.yaml) 与 [native audit](NATIVE_IO_AUDIT.md)。

- [GLOP](https://drive.google.com/file/d/1u9-GVTMRux3rWGcbipSqyTyBx_V8pm9G/view)
- [UDC](https://drive.google.com/file/d/1lVWBPvxhHDd-ZLJNCorGJugi_Smrx8uh/view)
- [CaDA](https://huggingface.co/datasets/Goodyee/CaDA/tree/main)
- [MVMoE](https://github.com/RoyalSkye/Routing-MVMoE/tree/af29e5af0595f94f3ecc3bc46d72df1089a62682/pretrained)
- [RF-TE](https://huggingface.co/ai4co/routefinder/tree/4edf73d68a5915787dd2f81ba177d77e13317190/checkpoints)
- [MoSES(CaDA)](https://github.com/panyxy/moses_vrp/tree/e301478b7a5df6d7b0b10a0543f4dee5e3c027a8/pretrained_moses_model/cada)
- [NeuOpt](https://github.com/yining043/NeuOpt/tree/ccf6b5f0f6a8fda2792b4be11d4ec35390a8139b/pre-trained)


## Frozen strict-load evidence

四次已成功加载没有为整理文档重复执行。[checkpoint manifest](../manifests/checkpoints.yaml)的每个对应row包含upstream commit、source、bytes/SHA、exact command、timestamp、CPU/解释器/Torch、完整model config、load method和missing/unexpected keys。原始输出为 `artifacts/audit/mvmoe_load.json`、`neuopt_load.json`。

MVMoE：MOEModel，embedding128、encoder6、decoder1、heads8、qkv16、ff512、experts4/topk2、Train_ALL、argmax、instance/norm_last、Enc0..5+Dec、node/input_choice；50/100分别读取其n50/n100权重。

NeuOpt：Actor，embedding/hidden128、heads4、layers3、layer norm、v_range6、k4、RNN/feature1/feature3=True、with_simpleMDP=True；seq_length分别70/120。torch.load(map_location=cpu, weights_only=False)之后官方load_state_dict(strict=True)。构造配置遵循options.py与README dummy_rate；没有训练、optimizer step或forward。

完整binary/hash是LOCAL_VERIFIED；MoSES deserialize仍BLOCKED（missing rl4co）。任何strict-load成功都只说明checkpoint与所构造模型匹配。MVMoE n50/n100另有server asset match和CVRP actual inference证据，见[server manifest](../manifests/server_mvmoe_cvrp.json)；其它server副本仍NOT_RUN。
