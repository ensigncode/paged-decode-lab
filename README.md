# Ensigncode Paged Decode Lab

CUDA C++ paged grouped-query attention for the single-token LLM decode step,
with independent correctness references and reproducible benchmarks.

**Validated on NVIDIA H100 PCIe with a pinned CUDA 13.0/PyTorch 2.11 stack:**
125 tests passed with no skips. NVIDIA memcheck, racecheck, synccheck and
initcheck each passed all 83 custom CUDA tests with zero reported errors or
race warnings. A further 1,000 randomized oracle cases exercised both merge
implementations. Three fresh H100 reports compare eager and CUDA graph timings.

The same final kernel also passed a fresh free-T4 run: 125 tests, four clean
sanitizer checks and 1,000 randomized cases. Nine CUDA activity traces, real
Nsight Compute counters and split-size experiments are now available. H100
stage/counter profiling remains outstanding; the T4 measurements are scoped to T4.

## Explore the complete project

The implementation and current documentation are on the dedicated public branch:

**[Open the CUDA portfolio project](https://github.com/ensigncode/paged-decode-lab/tree/portfolio/paged-gqa-decode)**

- [Kernel source](https://github.com/ensigncode/paged-decode-lab/blob/portfolio/paged-gqa-decode/src/paged_decode/csrc/decode.cu)
- [H100 acceptance evidence and retained failures](https://github.com/ensigncode/paged-decode-lab/blob/portfolio/paged-gqa-decode/results/2026-10-31-h100/SUMMARY.md)
- [H100 reproduction instructions](https://github.com/ensigncode/paged-decode-lab/blob/portfolio/paged-gqa-decode/docs/H100_VALIDATION.md)
- [Current T4 validation and profiling](https://github.com/ensigncode/paged-decode-lab/blob/portfolio/paged-gqa-decode/results/2026-10-31-final-t4/SUMMARY.md)
- [Historical T4 hardening evidence](https://github.com/ensigncode/paged-decode-lab/blob/portfolio/paged-gqa-decode/results/2026-10-31-hardening/SUMMARY.md)
- [Customer walkthrough](https://github.com/ensigncode/paged-decode-lab/blob/portfolio/paged-gqa-decode/docs/PORTFOLIO.md)

FlashInfer CUDA-core was 1.61–4.53× faster than the custom serial implementation
in the three fresh H100 eager measurements. CUDA 12.8 memcheck failures remain
visible; that older H100 configuration is not labeled sanitizer-clean.

This is an experimental attention-operator study. It does not provide a complete
LLM server, performance leadership, or exhaustive correctness coverage. Historical
T4 evidence retains its original source revisions.
