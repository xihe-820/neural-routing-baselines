# UDC / TSP — adapter contract

Target sizes: 50 / 100. `adapter=NOT_IMPLEMENTED`, `solution_decoder=NOT_IMPLEMENTED`, `runner=NOT_IMPLEMENTED`。本轮固定接口，不提供返回成功的stub。

输入由经审计的benchmark task字段与source hash/index构成。转换器归本method/problem目录所有；不调用共享baseline input adapter。共同validator可复用。

- Official source boundary: `single_objective/UDC-Large-scale-CO-master/UDC/{TSP,CVRP}-AGNN-ICAM/{TSPTesterrrc,CVRPTester}.py:_test_one_batch/_load_init_sol; TSPTesterrrc.py:gen_pyg_data (239)`。版本见 [upstreams](../../../manifests/upstreams.yaml)。
- Native input: TSP显式坐标[B,N,2]；CVRP depot+customer coordinates、customer raw demands/capacity只normalize一次。官方txt loader不直接接收Kit task pickle。
- Native output: TSP solution_gnn/solution_out全局node IDs；CVRP customer permutation + solution_flag route boundaries。
- Solution decoder obligation: 保留同一best candidate的solution与flag，转换成完整TSP cycle或depot-separated CVRP routes；验证客户coverage，不能只返回cost。
- Configuration constraints: 仅两个AGNN-ICAM目录；100为静态single-block candidate，50默认rrc失败；lib的padding差异详见SIZE_COMPATIBILITY。

接口设计：`adapt_instance(task, *, device, config)`只做明确字段/单位/索引转换并返回本方法native对象及provenance；`decode_solution(native_output, selection, mapping)`返回canonical IDs/routes和选择证据。这两个签名是设计契约，尚无实现。不得在adapter内训练、运行额外search或静默补不支持字段。具体张量batch与关闭约束值将在该方法实现测试中固定。

Benchmark truth见 [DATASET_AUDIT](../../../docs/DATASET_AUDIT.md)，源码/依赖与保真限制见 [NATIVE_IO_AUDIT](../../../docs/NATIVE_IO_AUDIT.md)，尺寸候选见 [SIZE_COMPATIBILITY](../../../docs/SIZE_COMPATIBILITY.md)。最终真实solution必须通过 [independent validators](../../../docs/VALIDATION_DESIGN.md)，Kit为secondary。
