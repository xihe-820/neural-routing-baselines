# LEHD implementation plan

1. Pin the clean official NCO_code checkout and audit its two repository-owned
   size-100 checkpoints, native datasets, strict loading, architecture, RNG,
   and RRC loop.
2. Freeze the nine-size, three-protocol matrix in `config.py`, including exact
   checkpoint, dataset, scale-origin, BS1, seed, and hardware provenance.
3. Convert ML4CO tasks at a project-owned boundary, capture the official final
   incumbent without extra solver/objective calls, and validate independently
   and with ML4CO-Kit.
4. Run official-style real batches for result reproduction only, preserving
   per-instance capture, independent validation, and ML4CO-Kit validation.
5. Use an isolated untimed warm-up Tester and a separate CUDA-synchronized BS1
   small sample for timing; its counts are Greedy=5, RRC50=3, RRC500=1.
6. Keep the prior exact-prefix-resume BS1 fullset and same-size evidence gates
   as a strict legacy audit, without making it the baseline-result requirement.

```
LEHD_RESULT_PROTOCOL = OFFICIAL_STYLE_BATCHED
LEHD_TIMING_PROTOCOL = BS1_SMALL_SAMPLE
```

Both paths require a single NVIDIA RTX 4090. Author-batch total wall time is a
diagnostic and is never presented as a BS1 timing result.
