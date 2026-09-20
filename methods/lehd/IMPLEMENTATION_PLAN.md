# LEHD implementation plan

1. Pin the clean official NCO_code checkout and audit its two repository-owned
   size-100 checkpoints, native datasets, strict loading, architecture, RNG,
   and RRC loop.
2. Freeze the nine-size, three-protocol matrix in `config.py`, including exact
   checkpoint, dataset, scale-origin, BS1, seed, and hardware provenance.
3. Convert ML4CO tasks at a project-owned boundary, capture the official final
   incumbent without extra solver/objective calls, and validate independently
   and with ML4CO-Kit.
4. Use an isolated untimed warm-up Tester, exact RNG restoration, a fresh
   formal Tester, CUDA-synchronized BS1 timing, atomic records, and exact-prefix
   resume.
5. Require same-size/project/source/dataset/checkpoint/protocol evidence before
   every fullset: Greedy and RRC50 use our-data smoke; RRC500 uses preflight.
6. Run official TSP1K/CVRP1K native smoke, the 18 our-data smoke cells, nine
   RRC500 preflights, then the 27 formal cells on the user-operated server.

Formal LEHD evaluation runs on a single NVIDIA RTX 4090 with original-instance
BS=1. A complete validated fullset is eligible for `PAPER_READY`.
