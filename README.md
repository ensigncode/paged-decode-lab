# Ensigncode Paged Decode Lab

CUDA C++ paged grouped-query attention for the single-token LLM decode step,
with independent correctness references and reproducible benchmarks.

**Validated on a real NVIDIA Tesla T4:** 57 tests passed; NVIDIA memory and
synchronization checks reported zero errors, and race checking reported zero
errors or warnings. Forty-two benchmark configurations are published with raw
samples, environment details, and source hashes.

## Explore the complete project

The implementation and current documentation are on the dedicated public branch:

**[Open the CUDA portfolio project](https://github.com/ensigncode/paged-decode-lab/tree/portfolio/paged-gqa-decode)**

- [Kernel source](https://github.com/ensigncode/paged-decode-lab/blob/portfolio/paged-gqa-decode/src/paged_decode/csrc/decode.cu)
- [GPU validation record](https://github.com/ensigncode/paged-decode-lab/blob/portfolio/paged-gqa-decode/docs/VALIDATION.md)
- [Complete measured matrix](https://github.com/ensigncode/paged-decode-lab/blob/portfolio/paged-gqa-decode/results/2026-10-31-t4/SUMMARY.md)
- [Customer walkthrough](https://github.com/ensigncode/paged-decode-lab/blob/portfolio/paged-gqa-decode/docs/PORTFOLIO.md)

This is an experimental attention-operator study. It is not a complete LLM server,
and its dense PyTorch comparisons do not establish a speedup over a tuned paged
production serving engine. See the full results for measurement scope and regressions.
