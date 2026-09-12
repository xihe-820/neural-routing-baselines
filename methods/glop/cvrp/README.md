# GLOP / CVRP — adapter contract

Target sizes: 50 / 100. `adapter=NOT_IMPLEMENTED`, `solution_decoder=NOT_IMPLEMENTED`, `runner=NOT_IMPLEMENTED`。本轮固定接口，不提供返回成功的stub。

输入由经审计的benchmark task字段与source hash/index构成。转换器归本method/problem目录所有；不调用共享baseline input adapter。共同validator可复用。

- Official source boundary: `main.py:eval_dataset/_eval_dataset; utils/functions.py:reconnect (308); problems/cvrp.py:init; heatmap/cvrp/inst.py:gen_pyg_data`。版本见 [upstreams](../../../manifests/upstreams.yaml)。
- Native input: TSP: points[B,N,2]。CVRP: 每例(depot,loc,raw_demand,raw_capacity)，保留raw单位；graph内部normalize一次。
- Native output: TSP重排坐标，CVRP padded sub-TSP坐标与route grouping；保留insertion/revision的原始ID映射及padding掩码，禁止按最近邻猜ID。
- Solution decoder obligation: TSP按原始node index输出单闭环；CVRP在flatten前保留route边界，映射为depot0/customers1..N。
- Configuration constraints: GLOP使用多份reviser及配套args.json；CVRP保留partitioner。50/100配置见SIZE_COMPATIBILITY，不照搬Reviser100到TSP50。

接口设计：`adapt_instance(task, *, device, config)`只做明确字段/单位/索引转换并返回本方法native对象及provenance；`decode_solution(native_output, selection, mapping)`返回canonical IDs/routes和选择证据。这两个签名是设计契约，尚无实现。不得在adapter内训练、运行额外search或静默补不支持字段。具体张量batch与关闭约束值将在该方法实现测试中固定。

Benchmark truth见 [DATASET_AUDIT](../../../docs/DATASET_AUDIT.md)，源码/依赖与保真限制见 [NATIVE_IO_AUDIT](../../../docs/NATIVE_IO_AUDIT.md)，尺寸候选见 [SIZE_COMPATIBILITY](../../../docs/SIZE_COMPATIBILITY.md)。最终真实solution必须通过 [independent validators](../../../docs/VALIDATION_DESIGN.md)，Kit为secondary。
