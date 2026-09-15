# 26-row STATUS

GLOP's manuscript-scale protocol audit is recorded separately in
[`GLOP_PAPER_PROTOCOL_AUDIT.md`](GLOP_PAPER_PROTOCOL_AUDIT.md). Official
algorithm protocols are frozen for TSP100/500/1K/2K/5K/10K and
CVRP500/1K/2K; their server preflights are prepared. TSP and CVRP paper-row
mappings are project-frozen, and the legacy 50/100 integration rows do not
authorize a formal paper run.
Formal external-baseline hardware is RTX4090; GLOP hardware policy is resolved,
while formal GLOP inference remains `NOT_RUN` pending server count-two preflight.

Source of truth: [status.json](../manifests/status.json). LV=LOCAL_VERIFIED; SV=SERVER_VERIFIED; SC=SOURCE_CONFIRMED; NR=NOT_RUN; NI=NOT_IMPLEMENTED; B=BLOCKED; NA=NOT_APPLICABLE.

| method | problem | size | source_audit | official_asset_identified | local_asset_verified | local_checkpoint_load | local_model_smoke | dataset_available_local | adapter | solution_decoder | independent_validator | local_small_smoke | kit_validation | server_asset_match | server_env_verified | server_gpu_smoke | server_small_smoke | final_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GLOP | TSP | 50 | LV | SC | LV | LV | LV | LV | LV | LV | LV | LV | LV | NR | NR | NR | NR | LOCAL_VERIFIED_INTEGRATION_SERVER_NOT_RUN |
| GLOP | TSP | 100 | LV | SC | LV | LV | LV | LV | LV | LV | LV | LV | LV | NR | NR | NR | NR | LOCAL_VERIFIED_INTEGRATION_SERVER_NOT_RUN |
| GLOP | CVRP | 50 | LV | SC | B | B | B | LV | B | B | LV | B | NR | NR | NR | NR | NR | Official small-size config undefined |
| GLOP | CVRP | 100 | LV | SC | B | B | B | LV | B | B | LV | B | NR | NR | NR | NR | NR | Official small-size config undefined |
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
| MVMoE | CVRPTW | 50 | LV | SC | LV | LV | LV | LV | LV | LV | LV | LV | LV | SV | SV | SV | SV | SERVER_VERIFIED_INTEGRATION |
| MVMoE | CVRPTW | 100 | LV | SC | LV | LV | LV | LV | LV | LV | LV | LV | LV | SV | SV | SV | SV | SERVER_VERIFIED_INTEGRATION |
| RF-TE | CVRP | 50 | LV | SC | NR | NR | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Foundation / NI |
| RF-TE | CVRP | 100 | LV | SC | NR | NR | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Foundation / NI |
| RF-TE | CVRPTW | 50 | LV | SC | NR | NR | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Foundation / NI |
| RF-TE | CVRPTW | 100 | LV | SC | NR | NR | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Foundation / NI |
| MoSES(CaDA) | CVRP | 50 | LV | SC | LV | B | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Deserialize blocked |
| MoSES(CaDA) | CVRP | 100 | LV | SC | LV | B | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Deserialize blocked |
| MoSES(CaDA) | CVRPTW | 50 | LV | SC | LV | B | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Deserialize blocked |
| MoSES(CaDA) | CVRPTW | 100 | LV | SC | LV | B | NR | LV | NI | NI | LV | NR | NR | NR | NR | NR | NR | Deserialize blocked |
| NeuOpt | CVRP | 50 | LV | SC | LV | LV | LV | LV | LV | LV | LV | LV | LV | SV | SV | SV | SV | SERVER_VERIFIED_INTEGRATION |
| NeuOpt | CVRP | 100 | LV | SC | LV | LV | LV | LV | LV | LV | LV | LV | LV | SV | SV | SV | SV | SERVER_VERIFIED_INTEGRATION |
