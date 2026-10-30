# Kernel design

## Contract and addressing

Inputs: `q[B,Hq,D]`, `k/v[P,T,Hkv,D]`, a CPU-validated page table
`pages[B,logical_pages]`, and `lengths[B]`. Output: FP32 `[B,Hq,D]`.
The query is the last decode token and sees every token in `[0,length)`.
There is no extra causal triangle for this one-token query.

Head `h` uses KV head `floor(h/(Hq/Hkv))`. Token `t` uses physical page
`pages[b,t//T]` and offset `t%T`. Unused table entries may be `-1`; live entries
must be in range. Sequences may share prefix pages; allocation and copy-on-write
are the caller's responsibility. K must already be position-encoded if needed.

## Thread mapping

The partial grid is `[Hq,B,S]`, where `S=max(1,ceil(max(lengths)/split_tokens))`.
Each CTA owns a sequence, query head, and contiguous context partition.
Its four warps visit disjoint token sets with stride four. Lane `l` owns dimensions
`l+32*i`, giving coalesced KV loads within a token despite fragmented pages.

Query components and weighted-value accumulators stay in registers. Warp shuffles
reduce the dot product and broadcast the score. Each lane updates its dimensions.
No cross-warp shared memory is used. Each warp and split redundantly loads Q;
query heads sharing a KV head still issue independent reads. These are deliberate
first-version trade-offs, not claims of optimal reuse.

Every warp writes state, even when it owns no tokens. The replay path does not
need workspace zero-initialization.

## Online softmax

For `x=dot(q,k)/sqrt(D)`, retain maximum `m`, denominator `l`, and numerator `n`.
Initialize `m=-inf,l=0,n=0`. For each score/value pair:

```math
m' = max(m,x)
alpha = exp(m-m')
beta = exp(x-m')
l' = alpha*l + beta
n' = alpha*n + beta*v
```

State and output are FP32 for both input dtypes. No logits/probabilities vector
is stored. Fast-math is not enabled. `expf`, `rsqrtf`, FMA, and reduction ordering
still differ from FP64. Extreme finite inputs may overflow dot products.

## Corrected merge

For all nonempty partials `(m_j,l_j,n_j)`:

```math
M = max_j(m_j)
L = sum_j(exp(m_j-M)*l_j)
N = sum_j(exp(m_j-M)*n_j)
output = N/L
```

Testing `l_j>0` before exponentiation avoids `exp(-inf-(-inf))`. An entirely empty
row returns zeros. There are no floating-point atomics. Reduction ordering is
fixed per launch shape; cross-architecture bitwise identity is not promised.

The merge has one CTA per `(sequence,head)` and one thread per output dimension.
Each dimension scans all partials, redundantly calculating max and denominator.
A hierarchical merge may be useful when S is large; this must be profiled.

## Workspace and complexity

There are `4*S` states per query head:

```math
workspace_bytes = B * Hq * 4 * S * (D+2) * 4
```

The last four is bytes per FP32 element. Output and metadata are extra.
For B=4,Hq=32,D=128,Lmax=4096,split_tokens=256 this is **4,259,840 bytes**.
Compute is O(sum(lengths)*Hq*D), plus O(B*Hq*S*D) merge work. The custom path has
no O(B*Hq*L) logits tensor, but intermediate memory still grows with S.

More splits expose parallelism but add merge and memory cost. A rectangular grid
based on the longest sequence launches empty CTAs for ragged batches. Compaction
is future work.

## Runtime and lifetime

The wrapper validates CPU metadata once and uploads copies. `DecodePlan.run`
performs no tensor allocation or host/device synchronization. A device guard and
the current PyTorch CUDA stream select the launch context. Workspace and output
are reused; output is borrowed and a plan is not reentrant. Callers must establish
cross-stream dependencies and keep inputs alive. Use separate plans for concurrency.

Compilation and warmup must finish before graph capture. The nondefault-stream and graph-replay test passed on the documented
Tesla T4 / PyTorch 2.11 / CUDA 13 stack. Other stacks require their own acceptance run.
The extension entry is internal and must not receive untrusted/mutated device
metadata. Its page-id guard is defense in depth, not a supported masking policy.

Head dimensions 64/128 keep two/four register components per lane. Page sizes
16/32 bound the initial contract. BF16, larger dimensions, quantized KV, and native
engine layouts require their own accuracy and performance work.
