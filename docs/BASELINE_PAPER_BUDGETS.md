# Baseline paper budgets

## UDC

UDC paper rows use project-selected budgets over the pinned official UDC
primitives. They are not released configurations named "official fewer" or
"official more" by the UDC authors. Alpha is the number of sampled initial
solutions; `x` is the number of iterative conquering/refinement stages.

| Problem | Paper label | Alpha | x | Sub-size | Configured POMO | Effective POMO |
|---|---:|---:|---:|---:|---:|---:|
| TSP | fewer | 50 | 2 | 100 | 2 | 2 |
| TSP | more | 50 | 50 | 100 | 2 | 2 |
| CVRP | fewer | 50 | 50 | 100 | 10 | 1 |
| CVRP | more | 50 | 250 | 100 | 10 | 1 |

The immutable machine-readable registry is
`manifests/paper_budgets.json`. Production uses seed 1234, batch size one, one
RTX 4090, and no geometric augmentation. The fixed pilot uses dataset indices
0, 1, and 2 of TSP500 and CVRP500. It confirms the preselected budgets without
sweeping or selecting a new value.

The latest manuscript sources audited on 2026-09-17 are
`main/tsp_main.tex`, `main/cvrp_main.tex`, `appendix/tsp_full.tex`, and
`appendix/cvrp_full.tex` in `ICLR27_Review_RS4CO`. The main tables contain the
required TSP 100/500/1K/2K/5K/10K and CVRP 200/500/1K/2K UDC cells with the
two frozen budgets. The appendix contains the same size columns but currently
uses an unfilled single `UDC` placeholder in each relevant block. Before paper
submission those placeholders must be expanded editorially to `UDC (fewer)`
and `UDC (more)` and populated from the same production artifacts as the main
tables. This is a table-layout edit and does not create another solver run.
