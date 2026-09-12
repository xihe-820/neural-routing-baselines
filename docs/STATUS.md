# 26-row STATUS

Source of truth: [status.json](../manifests/status.json). LV=LOCAL_VERIFIED; SC=SOURCE_CONFIRMED; SV=SERVER_VERIFIED; NR=NOT_RUN; NI=NOT_IMPLEMENTED; B=BLOCKED; NA=NOT_APPLICABLE.

`independent_validator=LV` is shared problem-level evidence from handcrafted tests and 30 reference solutions. `kit_validation` concerns **method-generated solutions**: MVMoE/CVRP50, CVRP100 and CVRPTW50 are LV. MVMoE/CVRP50+100 alone have user-executed server evidence at project commit `f6db50e694cbab8870f4f1a1544856c49e9e0106`; other server columns remain NR.

| method | problem | size | source_audit | official_asset_identified | local_asset_verified | local_checkpoint_load | local_model_smoke | dataset_available_local | adapter | solution_decoder | independent_validator | local_small_smoke | kit_validation | server_asset_match | server_env_verified | server_gpu_smoke | server_small_smoke | final_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GLOP | TSP | 50 | LV | SC | NR | NR | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Foundation / NI |
| GLOP | TSP | 100 | LV | SC | NR | NR | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Foundation / NI |
| GLOP | CVRP | 50 | LV | SC | NR | NR | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Foundation / NI |
| GLOP | CVRP | 100 | LV | SC | NR | NR | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Foundation / NI |
| UDC | TSP | 50 | LV | SC | NR | NR | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Default config blocked |
| UDC | TSP | 100 | LV | SC | NR | NR | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Foundation / NI |
| UDC | CVRP | 50 | LV | SC | NR | NR | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Default config blocked |
| UDC | CVRP | 100 | LV | SC | NR | NR | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Foundation / NI |
| CaDA | CVRP | 50 | LV | SC | NR | NR | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Foundation / NI |
| CaDA | CVRP | 100 | LV | SC | NR | NR | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Foundation / NI |
| CaDA | CVRPTW | 50 | LV | SC | NR | NR | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Foundation / NI |
| CaDA | CVRPTW | 100 | LV | SC | NR | NR | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Foundation / NI |
| MVMoE | CVRP | 50 | LV | SC | LV | LV | LV | LV | LV | LV | LV | LV | LV | SV | SV | SV | SV | SERVER_VERIFIED_INTEGRATION |
| MVMoE | CVRP | 100 | LV | SC | LV | LV | LV | LV | LV | LV | LV | LV | LV | SV | SV | SV | SV | SERVER_VERIFIED_INTEGRATION |
| MVMoE | CVRPTW | 50 | LV | SC | LV | LV | LV | LV | LV | LV | LV | LV | LV | NR | NR | NR | NR | LOCAL_VERIFIED_INTEGRATION_SERVER_NOT_RUN |
| MVMoE | CVRPTW | 100 | LV | SC | LV | LV | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Foundation / NI |
| RF-TE | CVRP | 50 | LV | SC | NR | NR | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Foundation / NI |
| RF-TE | CVRP | 100 | LV | SC | NR | NR | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Foundation / NI |
| RF-TE | CVRPTW | 50 | LV | SC | NR | NR | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Foundation / NI |
| RF-TE | CVRPTW | 100 | LV | SC | NR | NR | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Foundation / NI |
| MoSES(CaDA) | CVRP | 50 | LV | SC | LV | B | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Deserialize blocked |
| MoSES(CaDA) | CVRP | 100 | LV | SC | LV | B | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Deserialize blocked |
| MoSES(CaDA) | CVRPTW | 50 | LV | SC | LV | B | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Deserialize blocked |
| MoSES(CaDA) | CVRPTW | 100 | LV | SC | LV | B | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Deserialize blocked |
| NeuOpt | CVRP | 50 | LV | SC | LV | LV | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Foundation / NI |
| NeuOpt | CVRP | 100 | LV | SC | LV | LV | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Foundation / NI |

Compatibility risks and per-row notes are retained in JSON. See [native IO](NATIVE_IO_AUDIT.md), [size compatibility](SIZE_COMPATIBILITY.md), and [checkpoint evidence](CHECKPOINT_AUDIT.md).
