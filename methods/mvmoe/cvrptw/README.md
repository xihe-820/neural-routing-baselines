# MVMoE/4E / CVRPTW — adapter contract

Target sizes: 50 / 100. `adapter=NOT_IMPLEMENTED`, `solution_decoder=NOT_IMPLEMENTED`, `runner=NOT_IMPLEMENTED`。本轮固定接口，不提供返回成功的stub。

输入由经审计的benchmark task字段与source hash/index构成。转换器归本method/problem目录所有；不调用共享baseline input adapter。共同validator可复用。

- Official source boundary: `envs/CVRPEnv.py:load_dataset (295–303), load_problems (111); envs/VRPTWEnv.py:load_dataset (330–338), load_problems (119)`。版本见 [upstreams](../../../manifests/upstreams.yaml)。
- Native input: pickle边界CVRP=(depot,loc,raw_demand,capacity)，CVRPTW另附customer service_time,tw_start,tw_end；官方loader内部demand/capacity。若直接load_problems则提供已normalized demand，只选一个边界，禁止二次normalize。
- Native output: env.selected_node_list[B,P,T]与best POMO/augmentation的reward。
- Solution decoder obligation: 按同一best POMO/augmentation index选真实route，显式移除结束padding、加闭合depot边界并验证，不从reward反推解。
- Configuration constraints: MOE/4E、fine_tune_epochs=0；复用作者 `Tester._solve_cvrptwlib` 已有的 per-instance `env.depot_end = data[0,5] / scaler` 路径，把真实benchmark depot上界传入。该机制为官方evaluation行为，不是算法patch；actual CVRPTW validation仍未实现。

接口设计：`adapt_instance(task, *, device, config)`只做明确字段/单位/索引转换并返回本方法native对象及provenance；`decode_solution(native_output, selection, mapping)`返回canonical IDs/routes和选择证据。这两个签名是设计契约，尚无实现。不得在adapter内训练、运行额外search或静默补不支持字段。具体张量batch与关闭约束值将在该方法实现测试中固定。

Benchmark truth见 [DATASET_AUDIT](../../../docs/DATASET_AUDIT.md)，源码/依赖与保真限制见 [NATIVE_IO_AUDIT](../../../docs/NATIVE_IO_AUDIT.md)，尺寸候选见 [SIZE_COMPATIBILITY](../../../docs/SIZE_COMPATIBILITY.md)。最终真实solution必须通过 [independent validators](../../../docs/VALIDATION_DESIGN.md)，Kit为secondary。
