# CaDA / CVRP — adapter contract

Target sizes: 50 / 100. `adapter=NOT_IMPLEMENTED`, `solution_decoder=NOT_IMPLEMENTED`, `runner=NOT_IMPLEMENTED`。本轮固定接口，不提供返回成功的stub。

输入由经审计的benchmark task字段与source hash/index构成。转换器归本method/problem目录所有；不调用共享baseline input adapter。共同validator可复用。

- Official source boundary: `50/envs/env.py:MTVRPEnv._reset (151); 50/envs/generator.py (218,229–230); corresponding 100/ files; model.py:forward`。版本见 [upstreams](../../../manifests/upstreams.yaml)。
- Native input: 单独CaDA TensorDict：locs[B,N+1,2]；demand_linehaul/demand_backhaul[B,N+1] **包含depot0**；normalize raw/capacity一次，vehicle_capacity=1并保留capacity_original。time_windows/service_time包含depot。
- Native output: model.forward内部actions_list/actions，但outdict仅reward/log_likelihood。
- Solution decoder obligation: 未来capture原forward真实actions，按同一个best start/augmentation选择；不能rerun搜索补解。转换0-depot routes并验证。
- Configuration constraints: CVRP显式关闭TW/backhaul/open-route/distance-limit约束；CVRPTW传入真实tw/service、speed1和depot4.6。具体关闭值遵循本方法env，禁止共享RF converter。

接口设计：`adapt_instance(task, *, device, config)`只做明确字段/单位/索引转换并返回本方法native对象及provenance；`decode_solution(native_output, selection, mapping)`返回canonical IDs/routes和选择证据。这两个签名是设计契约，尚无实现。不得在adapter内训练、运行额外search或静默补不支持字段。具体张量batch与关闭约束值将在该方法实现测试中固定。

Benchmark truth见 [DATASET_AUDIT](../../../docs/DATASET_AUDIT.md)，源码/依赖与保真限制见 [NATIVE_IO_AUDIT](../../../docs/NATIVE_IO_AUDIT.md)，尺寸候选见 [SIZE_COMPATIBILITY](../../../docs/SIZE_COMPATIBILITY.md)。最终真实solution必须通过 [independent validators](../../../docs/VALIDATION_DESIGN.md)，Kit为secondary。
