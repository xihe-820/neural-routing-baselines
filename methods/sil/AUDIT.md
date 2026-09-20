# Pinned SIL source and protocol audit

Audited source: `https://github.com/CIAM-Group/SIL` at
`9ec783e90a1631f7b95f84eb20f8f9751cb45c10`. The checkout must be detached,
clean, and read-only. The paper evidence is the ICLR 2025 proceedings version of
*Boosting Neural Combinatorial Optimization for Large-Scale Vehicle Routing
Problems*.

## Official behavior

The greedy scripts construct a complete solution directly with model argmax and
set `budget=0`, `random_insertion=False`. Their main functions disable k-nearest
only for this purely greedy configuration. Greedy is retained here as a native
compatibility diagnostic.

The PRC scripts set `PRC=True`, `random_insertion=True`, maximum reconstructed
sub-solution length 1,000, and iterate the `budget` loop exactly that many times.
They first generate a complete solution by random insertion. Before the loop they
draw one sub-length and one first index per iteration. Every iteration may invert
or rearrange the current solution, destroys the selected sub-solution, invokes
the same neural model to reconstruct it, and accepts it only under the official
comparison. `budget` therefore means the exact number of PRC iterations after
random-insertion initialization.

Both official test configurations use POMO size 1, beam width 16, greedy decode,
embedding dimension 128, qkv dimension 16, eight heads, hidden dimension 512,
logit clipping 10, and `k_nearest_num=1000`. TSP declares six encoder layers;
CVRP declares six decoder layers. In the model the kNN branch is conditional on
the current data length being strictly greater than 1,000. The initial path is
therefore full for n=500 and n=1,000, and kNN for n=2,000 (then falls back once
the remaining set is at most 1,000).

TSP represents a solution as an integer permutation `[B,N]`. CVRP represents one
visit per customer as `[B,N,2]`: column zero is the 1-based customer ID and
column one marks that the customer is reached via the depot, i.e. starts a route.
TSP objective is the Euclidean closed-cycle length. CVRP `cal_length` turns the
route-start markers into depot edges and sums the complete depot-separated route
length.

The tester constructor uses seed 123 for CUDA, NumPy, and Python. CVRP additionally
calls `torch.manual_seed(123)` at the beginning of every `_test_one_batch`.
Official PRC also uses Torch RNG for inversion, reconstruction lengths/indices,
and TSP insertion orders; batch composition can therefore change RNG consumption.
The project fixes original-instance BS=1 and saves all RNG states on resume.

The official `run()` and `_test_one_batch()` return objective aggregates and
problem size, not the final route. The final solution remains a local
`best_select_node_list`. Objective calls receive that local value, which makes a
non-mutating observation hook possible.

The CVRP loader constructs `[x,y,raw_demand,capacity]`. `CVRP_Encoder.forward`
then divides demand by `capacity`. Passing ML4CO `norm_demands` would normalize
twice. The integration passes raw `task.demands` and `task.capacity`. The
official synthetic capacity table is not used for benchmark tasks. Formal BS=1
also avoids the official model's `raw_data_capacity.ravel()[0]` assumption
mixing different per-instance capacities. Because the official CVRP random
insertion casts both demands and capacity to integer types, the adapter rejects
fractional raw units rather than silently changing them.

Official native checkpoint mapping is 1K, 5K, and 10K for TSP and 1K for the
CVRP scope here. The repository's source scripts additionally list larger
checkpoints outside this project's scope. The four required release filenames
and Drive IDs are in `manifests/sil_checkpoints.json`; their bytes and SHA256 are
pending server verification.

## Project adaptation

TSP500 and TSP2000 use the TSP1K checkpoint/settings; CVRP500 and CVRP2000 use
CVRP1K. These are `senior_approved_adaptation`, not official size-specific SIL
models. TSP1K, TSP5K, TSP10K, and CVRP1K are `official_native`. Actual input size
is never padded, duplicated, or truncated.

The project wrapper injects exact model tensors into the official environment,
then calls the pinned tester method. A temporary hook around the environment's
existing objective method retains its last full-size solution argument. It does
not call the solver, reconstruction, or objective again; it returns the official
value unchanged. The retained tensor is cloned after CUDA-synchronized timing.

## Project evaluation policy

The paper's synthetic Table 1 reports greedy and PRC10, PRC50, PRC100, PRC500,
and PRC1000 for both TSP and CVRP. It states that PRC uses random insertion for
initial solutions. Both PRC50 and PRC500 are therefore author-reported settings
within the same algorithmic pipeline. This project maps `fewer=PRC50` and
`more=PRC500`. This mapping is a project evaluation policy; the authors did not
name these settings “fewer” and “more”. `PROTOCOL_PENDING` is false. Greedy is a
diagnostic only and cannot produce a formal fullset artifact.

Formal timing is CUDA-synchronized wall time around one official BS1
`_test_one_batch`, including initialization/search and its internal official
objective/logging calls. It excludes data/checkpoint/model loading, adapter work,
post-timing tensor clone, independent/Kit validation, and artifact I/O. The
formal GPU requirement is RTX 4090. Server smoke must establish Python 3.11 /
Torch 2.5 compatibility; the official README only documents Python 3.8.6 and
Torch 1.12.1.
