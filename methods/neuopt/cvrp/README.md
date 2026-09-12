# NeuOpt / CVRP — adapter contract

Target sizes: 50 / 100. `adapter=NOT_IMPLEMENTED`, `solution_decoder=NOT_IMPLEMENTED`, `runner=NOT_IMPLEMENTED`。本轮固定接口，不提供返回成功的stub。

输入由经审计的benchmark task字段与source hash/index构成。转换器归本method/problem目录所有；不调用共享baseline input adapter。共同validator可复用。

- Official source boundary: `problems/problem_cvrp.py:CVRPDataset.make_instance (383–391); agent/ppo.py:rollout; options.py; nets/actor_network.py:Actor`。版本见 [upstreams](../../../manifests/upstreams.yaml)。
- Native input: raw (depot,loc,demand,capacity)进入官方make_instance，内部normalize并插入dummy depots。50→70节点，100→120节点。
- Native output: best feasible incumbent的successor graph，非最后一步任意search state。
- Solution decoder obligation: 迁移历史decoder；先验证单一successor cycle，再将全部dummy depot映射0、customer映射1..N，明确处理相邻dummy边，独立检查routes。
- Configuration constraints: 只CVRP；保持batch>=2避免历史squeeze问题，with_simpleMDP=True；不重做旧项目smoke或继承其完成状态。

接口设计：`adapt_instance(task, *, device, config)`只做明确字段/单位/索引转换并返回本方法native对象及provenance；`decode_solution(native_output, selection, mapping)`返回canonical IDs/routes和选择证据。这两个签名是设计契约，尚无实现。不得在adapter内训练、运行额外search或静默补不支持字段。具体张量batch与关闭约束值将在该方法实现测试中固定。

Benchmark truth见 [DATASET_AUDIT](../../../docs/DATASET_AUDIT.md)，源码/依赖与保真限制见 [NATIVE_IO_AUDIT](../../../docs/NATIVE_IO_AUDIT.md)，尺寸候选见 [SIZE_COMPATIBILITY](../../../docs/SIZE_COMPATIBILITY.md)。最终真实solution必须通过 [independent validators](../../../docs/VALIDATION_DESIGN.md)，Kit为secondary。
