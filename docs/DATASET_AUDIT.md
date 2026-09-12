# 六组正式 benchmark 数据：LOCAL_VERIFIED

来源：[官方 ML4CO/ML4CO-Bench-101-SL](https://huggingface.co/datasets/ML4CO/ML4CO-Bench-101-SL/tree/main/test_dataset)。只获取当前六文件，共67,877,012 bytes；其中已有TSP50/CVRP50按官方SHA复用，其余四文件定向下载。每文件与官方LFS SHA256一致。main URL可变化，内容身份由SHA256固定；未取得dataset repository revision，不能声称commit pin。

| Problem / N | Filename / official source | Bytes | SHA256 | Count |
|---|---|---:|---|---:|
| TSP / 50 | [tsp50_concorde_5.688.pkl](https://huggingface.co/datasets/ML4CO/ML4CO-Bench-101-SL/resolve/main/test_dataset/tsp/tsp50_concorde_5.688.pkl) | 1264010 | `1ede2b289d2e6fbfe614219a86d1ba6dfca2a726925a8fe8cef9c8196ee0b213` | 1280 |
| TSP / 100 | [tsp100_concorde_7.756.pkl](https://huggingface.co/datasets/ML4CO/ML4CO-Bench-101-SL/resolve/main/test_dataset/tsp/tsp100_concorde_7.756.pkl) | 2288145 | `a2bfe99857b8072bdba051f6ae402b7e241f01b0462c5f379ed0aa03786406a0` | 1280 |
| CVRP / 50 | [cvrp50_hgs-1s_10.366.pkl](https://huggingface.co/datasets/ML4CO/ML4CO-Bench-101-SL/resolve/main/test_dataset/cvrp/cvrp50_hgs-1s_10.366.pkl) | 20125346 | `eea12fbefe9c1bcc008d56ecfc1c50dadd64ac774f3547774c9fade8a7baa6c2` | 10000 |
| CVRP / 100 | [cvrp100_hgs-20s_15.563.pkl](https://huggingface.co/datasets/ML4CO/ML4CO-Bench-101-SL/resolve/main/test_dataset/cvrp/cvrp100_hgs-20s_15.563.pkl) | 36487115 | `bb47d5a113848e5a404edefc562d5d2ef6b0ade1aafc287828bdf60364e23532` | 10000 |
| CVRPTW / 50 | [cvrptw50_pyvrp-10s_16.038.pkl](https://huggingface.co/datasets/ML4CO/ML4CO-Bench-101-SL/resolve/main/test_dataset/cvrptw/cvrptw50_pyvrp-10s_16.038.pkl) | 2727373 | `a16975d9dd242739973191256e1cdbd8166c4405759ef195b985a8c3ec49df40` | 1000 |
| CVRPTW / 100 | [cvrptw100_pyvrp-20s_25.431.pkl](https://huggingface.co/datasets/ML4CO/ML4CO-Bench-101-SL/resolve/main/test_dataset/cvrptw/cvrptw100_pyvrp-20s_25.431.pkl) | 4985023 | `3b74fa520f42f7aa607a5bd00a7d4aaa118e0715ca1672ee854fc850bac67867` | 1000 |

所有容器均为 Python list，实例为实际 ML4CO-Kit task 对象；全量 task class 与 points shape 均检查。详细首实例属性、dtype、shape及source hash保存在 [datasets.yaml](../manifests/datasets.yaml) 与 `artifacts/audit/datasets.json`，没有把文件名当schema证据。

| Problem | Actual schema | Reference solution |
|---|---|---|
| TSP | TSPTask; points[N,2] float32 | ref_sol[N+1] int64, 0..N-1, repeated first node closes cycle |
| CVRP | CVRPTask; depots[2], points[N,2], coords[N+1,2] float32; demands[N] raw float32; norm_demands=raw/capacity | int64, depot0 separators, customers1..N, explicit depot return |
| CVRPTW | CVRPTWTask; above plus tw[N+1,2] and service[N+1] float32, both depot-inclusive | Same route encoding; service-start TW and return-depot constraints |

坐标为真实二维Euclidean坐标；这六组均使用 `DISTANCE_TYPE.EUC_2D`、`dist_eval.round_type=ROUND_TYPE.NO`。独立目标用float64逐边距离求和，包含回depot/闭环边；不采用文件名平均cost。CVRP/CVRPTW原始demand为1..9；capacity在50为40、100为50。normalized需求仅作为派生值，不覆盖raw数据。

CVRPTW两组所有depot TW均为 `[0, 4.599999904632568]`，depot service=0。`tw[:,0]`/`tw[:,1]` 是service-start上下界，`service`是服务时长。真实数据与Kit `CVRPTWTask._check_route_tw`共同确认：每条路线time=0出发；travel time=Euclidean distance（speed=1）；到达后等待至下界；service start <= upper；departure=start+service；最后到depot也必须不晚于depot upper。目标仍是总距离，无车辆数项或等待/服务惩罚。Kit tolerance为CVRP 1e-5、CVRPTW 1e-4，独立接口将tolerance显式参数化；CVRPTW reference测试使用task time threshold，CVRP capacity tolerance保持0。

这些结论只覆盖已审计的六文件和Kit代码版本，不从任何baseline env推断benchmark。尤其MVMoE默认depot TW上界3.0与实际4.6不一致，后续必须显式传入并验证，不能截断benchmark TW。

| Reference instance 0 | Independent distance | Kit distance | Feasible / objective agreement |
|---|---:|---:|---|
| TSP50 | 5.826049794217 | 5.826050281525 | Yes / Yes |
| TSP100 | 7.797036008478 | 7.797035694122 | Yes / Yes |
| CVRP50 | 10.973379950203 | 10.973381996155 | Yes / Yes |
| CVRP100 | 14.605232219007 | 14.605233192444 | Yes / Yes |
| CVRPTW50 | 13.199350986061 | 13.199351310730 | Yes / Yes |
| CVRPTW100 | 23.487735013781 | 23.487731933594 | Yes / Yes |

此外六文件各前5个reference通过独立约束与Kit二次核验（30实例），成本比较rtol=1e-6、atol=1e-6。这不是全量参考解验证，更不是baseline推理结果。服务器只需确认实际使用的副本SHA及自身loader环境。
