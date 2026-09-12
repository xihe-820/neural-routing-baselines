# Common utilities

`hashing.py`: streaming SHA256. `validation.py`: finite numeric shapes/scalars and strict integer solution parsing. `provenance.py` records read-only Git/source/environment identity. `result_schema.py` writes strict per-instance JSON and rejects missing identity, false validation claims, and NaN/Inf. Problem-specific constraints live under problems/. No shared baseline TensorDict adapter.
