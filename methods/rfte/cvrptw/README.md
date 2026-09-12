# RF-TE / CVRPTW — adapter contract

Target sizes: 50 / 100. `adapter=NOT_IMPLEMENTED`, `solution_decoder=NOT_IMPLEMENTED`, `runner=NOT_IMPLEMENTED`。本轮固定接口，不提供返回成功的stub。

输入由经审计的benchmark task字段与source hash/index构成。转换器归本method/problem目录所有；不调用共享baseline input adapter。共同validator可复用。

- Official source boundary: `routefinder/envs/mtvrp/env.py:MTVRPEnv._reset (193–207); test.py; routefinder/models/policy.py; configs/experiment/main/rf/rf-transformer-{50,100}.yaml`。版本见 [upstreams](../../../manifests/upstreams.yaml)。
- Native input: 本方法TensorDict：locs[B,N+1,2]；reset前demand_linehaul/backhaul[B,N] **只有customers**，reset自动补depot0。需求raw/capacity一次、vehicle_capacity1、capacity_original保留。TW/service包含depot。
- Native output: policy(return_actions=True)，best_multistart_actions/best_aug_actions。
- Solution decoder obligation: 提取同一reward所选actions，映射为depot0/customer1..N routes；记录padding规则与best index；保持Transformer配置。
- Configuration constraints: CVRP显式关闭其他约束，CVRPTW使用真实tw/service/speed/depot值；不把CaDA depot-inclusive需求传入reset。

接口设计：`adapt_instance(task, *, device, config)`只做明确字段/单位/索引转换并返回本方法native对象及provenance；`decode_solution(native_output, selection, mapping)`返回canonical IDs/routes和选择证据。这两个签名是设计契约，尚无实现。不得在adapter内训练、运行额外search或静默补不支持字段。具体张量batch与关闭约束值将在该方法实现测试中固定。

Benchmark truth见 [DATASET_AUDIT](../../../docs/DATASET_AUDIT.md)，源码/依赖与保真限制见 [NATIVE_IO_AUDIT](../../../docs/NATIVE_IO_AUDIT.md)，尺寸候选见 [SIZE_COMPATIBILITY](../../../docs/SIZE_COMPATIBILITY.md)。最终真实solution必须通过 [independent validators](../../../docs/VALIDATION_DESIGN.md)，Kit为secondary。
