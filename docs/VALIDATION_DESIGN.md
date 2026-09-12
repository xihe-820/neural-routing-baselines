# 独立 objective / validator：已实现

实现仅依赖NumPy，不调用任何baseline或Kit成本/约束函数。当前支持六组实际benchmark的unrounded Euclidean、closed TSP/single-depot CVRP/CVRPTW。实例规模可用于小型handcrafted tests，正式运行scope仍50/100。服务器还未执行。

| Module / API | Input | Output / constraints |
|---|---|---|
| `problems.tsp.validate.validate(points, solution, representation="tour")` | points[N,2], integer permutation; optional repeated start; or explicit `representation="successors"` | range, exactly once, one complete cycle, independently closed distance |
| `problems.cvrp.validate.validate(depot, points, demands, capacity, solution, capacity_tolerance=0)` | single depot[2]/[1,2], customers[N,2], raw demands[N], raw capacity, depot0-separated integer solution | starts/ends depot, no empty routes, every customer once, raw route loads, Euclidean distance |
| `problems.cvrptw.validate.validate(..., time_windows, service_times, solution, *, speed, start_time=0, time_tolerance=0, capacity_tolerance=0)` | same raw CVRP units; TW[N+1,2], service[N+1], explicit speed | all CVRP checks plus per-route arrival/wait/service start/service/departure/return timeline |

Common result has `feasible`, `independent_objective`, `constraint_details`; well-formed routing solutions include `routes`, TSP includes canonical `solution`. Invalid shape/type/IDs are rejected; malformed input returns failure with null objective. Structurally valid but capacity/TW-infeasible routes retain independently computed distance for diagnosis. Cost alone never implies feasibility. Floats and booleans are rejected as node IDs, not silently cast. TSP successor decoder rejects multiple disjoint cycles.

CVRPTW模拟：`arrival = prior_departure + distance/speed`; `service_start = max(arrival, tw_start)`; `departure = service_start + service_duration`。检查service-start上界而非强迫service-finish在窗内；最后travel回depot检查上界，每条route独立从start_time出发。initial depot service不另加（实际数据depot service=0，Kit同样从time0出发）；departure必须处于depot窗内。六文件speed=1、depotTW约4.6，详见 [真实数据证据](DATASET_AUDIT.md)。接口不推断单位，不把baseline默认值当truth。

连续depot padding必须由对应method decoder显式剥离并记录映射；validator不静默修复错误解。未实现open route、多depot、rounded distance、车辆数目标；不宣称支持这些未审计variant。

Tests：`tests/test_validators.py` 26个handcrafted cases覆盖valid/duplicate/missing/invalid ID、one/multi-route、capacity、waiting、exact boundary、late start、service引发后续迟到、depot return与非法参数；`tests/test_audits.py`检查缺文件/LFS/路径隔离等失败分支。`tests/test_benchmark_references.py`对六文件各前5例校验SHA、独立可行性/距离及Kit（30 reference instances）。参考解不是模型输出。

```bash
# Current environment must already provide NumPy
python -B -m unittest discover -s tests -v
# Optional real-reference suite; current interpreter must already provide Kit
ML4CO_REFERENCE_TESTS=1 python -B -m unittest discover -s tests -v
```

成本比较rtol=1e-6/atol=1e-6。CVRP独立capacity默认严格0容差；CVRPTW reference time_tolerance取实际task.threshold=1e-4，未随意放宽约束。单位测试不需要GPU/checkpoint/服务器。
