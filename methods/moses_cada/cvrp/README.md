# MoSES(CaDA) / CVRP — adapter contract

Target sizes: 50 / 100. `adapter=NOT_IMPLEMENTED`, `solution_decoder=NOT_IMPLEMENTED`, `runner=NOT_IMPLEMENTED`。本轮固定接口，不提供返回成功的stub。

输入由经审计的benchmark task字段与source hash/index构成。转换器归本method/problem目录所有；不调用共享baseline input adapter。共同validator可复用。

- Official source boundary: `envs/mtvrp/env.py:MTVRPEnv._reset (210–225); models/policy.py:CadaMultiLoRAPolicy; test.py; scripts/test_script.sh`。版本见 [upstreams](../../../manifests/upstreams.yaml)。
- Native input: MoSES自己的TensorDict：locs[B,N+1,2]；reset前demand_linehaul/backhaul[B,N]为customers，reset补depot；raw/capacity一次。TW/service包含depot，保留capacity_original。
- Native output: return_actions及best start/augmentation对应actions；使用CadaMultiLoRAPolicy已训练sigmoid checkpoint。
- Solution decoder obligation: 与原cost使用相同candidate index，按MoSES自身padding语义恢复routes；验证actual decoded solution。
- Configuration constraints: CaDA backbone多LoRA已训练权重只加载推理；不启动training；不同于CaDA原repo的需求边界，独立实现converter。

接口设计：`adapt_instance(task, *, device, config)`只做明确字段/单位/索引转换并返回本方法native对象及provenance；`decode_solution(native_output, selection, mapping)`返回canonical IDs/routes和选择证据。这两个签名是设计契约，尚无实现。不得在adapter内训练、运行额外search或静默补不支持字段。具体张量batch与关闭约束值将在该方法实现测试中固定。

Benchmark truth见 [DATASET_AUDIT](../../../docs/DATASET_AUDIT.md)，源码/依赖与保真限制见 [NATIVE_IO_AUDIT](../../../docs/NATIVE_IO_AUDIT.md)，尺寸候选见 [SIZE_COMPATIBILITY](../../../docs/SIZE_COMPATIBILITY.md)。最终真实solution必须通过 [independent validators](../../../docs/VALIDATION_DESIGN.md)，Kit为secondary。
