# Common utilities

`hashing.py`: streaming SHA256. `validation.py`: finite numeric shapes/scalars and strict integer solution parsing. `objective_agreement.py` is the single NumPy `isclose` policy (`rtol=atol=1e-6`). `provenance.py` records read-only Git/source/environment identity and normalizes common GitHub HTTPS/SSH remote forms. `result_schema.py` records explicit timing semantics and requires all four objective/feasibility gates for `LOCAL_VERIFIED`. Problem-specific constraints live under problems/. No shared baseline TensorDict adapter.
