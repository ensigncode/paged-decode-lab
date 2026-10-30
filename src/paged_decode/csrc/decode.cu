#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {
constexpr int kWarp = 32;
constexpr int kWarps = 4;
constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        value += __shfl_down_sync(kFullMask, value, delta);
    return __shfl_sync(kFullMask, value, 0);
}

// A CTA covers one (sequence, query head, context partition). Each warp keeps
// its own online softmax state. Lanes own adjacent dimensions: KV loads are
// coalesced even when the page table is arbitrarily fragmented.
template <typename scalar_t, int D>
__global__ void partial_decode(
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v, const int32_t* __restrict__ table,
    const int32_t* __restrict__ lengths, float* __restrict__ numerator,
    float* __restrict__ stats, int heads, int kv_heads, int page_size,
    int table_width, int physical_pages, int splits, int split_tokens) {
    const int head = blockIdx.x;
    const int batch = blockIdx.y;
    const int split = blockIdx.z;
    const int lane = threadIdx.x % kWarp;
    const int warp = threadIdx.x / kWarp;
    const int kv_head = head / (heads / kv_heads);
    const int parts = splits * kWarps;
    const int64_t part = (int64_t(batch) * heads + head) * parts
                        + split * kWarps + warp;
    const int begin = split * split_tokens;
    const int end = min(begin + split_tokens,
                        min(lengths[batch], table_width * page_size));
    float query[D / kWarp], accumulator[D / kWarp];
    #pragma unroll
    for (int i = 0; i < D / kWarp; ++i) {
        query[i] = float(q[(int64_t(batch) * heads + head) * D + i * kWarp + lane]);
        accumulator[i] = 0.0f;
    }
    float maximum = -INFINITY;
    float denominator = 0.0f;
    const float scale = rsqrtf(float(D));
    for (int token = begin + warp; token < end; token += kWarps) {
        const int page = table[int64_t(batch) * table_width + token / page_size];
        // The public Plan validates metadata before device upload. Guard the
        // low-level entry as well so an invalid id cannot address arbitrary VRAM.
        if (page < 0 || page >= physical_pages) continue;
        const int64_t base = ((int64_t(page) * page_size + token % page_size)
                              * kv_heads + kv_head) * D;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < D / kWarp; ++i)
            dot = fmaf(query[i], float(k[base + i * kWarp + lane]), dot);
        const float score = warp_sum(dot) * scale;
        const float next_maximum = fmaxf(maximum, score);
        const float alpha = expf(maximum - next_maximum);
        const float beta = expf(score - next_maximum);
        #pragma unroll
        for (int i = 0; i < D / kWarp; ++i)
            accumulator[i] = fmaf(beta, float(v[base + i * kWarp + lane]),
                                  accumulator[i] * alpha);
        denominator = fmaf(denominator, alpha, beta);
        maximum = next_maximum;
    }
    #pragma unroll
    for (int i = 0; i < D / kWarp; ++i)
        numerator[part * D + i * kWarp + lane] = accumulator[i];
    if (lane == 0) {
        stats[part * 2] = maximum;
        stats[part * 2 + 1] = denominator;
    }
}

// Merge unnormalized partials with max correction. Empty warps have l=0;
// testing l before exp avoids the undefined -inf - -inf case for empty rows.
template <int D>
__global__ void merge_decode(const float* __restrict__ numerator,
                             const float* __restrict__ stats,
                             float* __restrict__ out, int parts) {
    const int64_t row = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= D) return;
    float maximum = -INFINITY;
    for (int p = 0; p < parts; ++p) {
        const int64_t index = (row * parts + p) * 2;
        if (stats[index + 1] > 0.0f) maximum = fmaxf(maximum, stats[index]);
    }
    float denominator = 0.0f, value = 0.0f;
    for (int p = 0; p < parts; ++p) {
        const int64_t index = (row * parts + p) * 2;
        if (stats[index + 1] == 0.0f) continue;
        const float correction = expf(stats[index] - maximum);
        denominator = fmaf(correction, stats[index + 1], denominator);
        value = fmaf(correction, numerator[(row * parts + p) * D + d], value);
    }
    out[row * D + d] = denominator > 0.0f ? value / denominator : 0.0f;
}

template <typename scalar_t, int D>
void launch(torch::Tensor q, torch::Tensor k, torch::Tensor v,
            torch::Tensor table, torch::Tensor lengths, torch::Tensor numerator,
            torch::Tensor stats, torch::Tensor out, int split_tokens,
            cudaStream_t stream) {
    const int splits = numerator.size(2) / kWarps;
    dim3 grid(q.size(1), q.size(0), splits);
    partial_decode<scalar_t, D><<<grid, kWarp * kWarps, 0, stream>>>(
        q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
        table.data_ptr<int32_t>(), lengths.data_ptr<int32_t>(),
        numerator.data_ptr<float>(), stats.data_ptr<float>(), q.size(1),
        k.size(2), k.size(1), table.size(1), k.size(0), splits, split_tokens);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    merge_decode<D><<<q.size(0) * q.size(1), D, 0, stream>>>(
        numerator.data_ptr<float>(), stats.data_ptr<float>(),
        out.data_ptr<float>(), splits * kWarps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace

void decode_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                 torch::Tensor table, torch::Tensor lengths,
                 torch::Tensor numerator, torch::Tensor stats,
                 torch::Tensor out, int64_t split_tokens) {
    for (const auto& tensor : {q, k, v, table, lengths, numerator, stats, out}) {
        TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(),
                    "all tensors must be contiguous CUDA tensors");
        TORCH_CHECK(tensor.device() == q.device(), "device mismatch");
    }
    TORCH_CHECK(q.dim() == 3 && k.dim() == 4 && v.sizes() == k.sizes(),
                "expected q[B,H,D], k/v[P,T,KV,D]");
    const int64_t b = q.size(0), h = q.size(1), d = q.size(2);
    TORCH_CHECK(b > 0 && b <= 65535 && h > 0 && h <= 65535,
                "batch and head count must be in [1,65535]");
    TORCH_CHECK(d == 64 || d == 128, "supported head dimensions: 64,128");
    TORCH_CHECK(k.size(0) > 0 && k.size(1) > 0 && k.size(2) > 0 &&
                k.size(3) == d && h % k.size(2) == 0 && k.size(0) <= INT32_MAX &&
                (k.size(1) == 16 || k.size(1) == 32), "invalid cache shape");
    TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
                "q/k/v dtype mismatch");
    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kFloat,
                "supported input dtypes: float16,float32");
    TORCH_CHECK(table.dim() == 2 && table.size(0) == b && table.size(1) <= 65536 &&
                lengths.dim() == 1 && lengths.size(0) == b, "metadata shape mismatch");
    TORCH_CHECK(table.scalar_type() == at::kInt && lengths.scalar_type() == at::kInt,
                "metadata must be int32");
    TORCH_CHECK(numerator.dim() == 4 && numerator.size(0) == b &&
                numerator.size(1) == h && numerator.size(2) > 0 &&
                numerator.size(2) % kWarps == 0 && numerator.size(3) == d,
                "invalid numerator workspace");
    const int64_t parts = numerator.size(2);
    TORCH_CHECK(parts / kWarps <= 65535 && split_tokens > 0 &&
                split_tokens <= 1048576, "invalid split configuration");
    TORCH_CHECK(stats.dim() == 4 && stats.size(0) == b && stats.size(1) == h &&
                stats.size(2) == parts && stats.size(3) == 2, "invalid stats workspace");
    TORCH_CHECK(out.sizes() == q.sizes(), "invalid output shape");
    TORCH_CHECK(numerator.scalar_type() == at::kFloat && stats.scalar_type() == at::kFloat &&
                out.scalar_type() == at::kFloat, "workspaces/output must be float32");
    // Values in metadata are checked on CPU by DecodePlan, outside the hot path.
    // This internal entry is not a supported API for untrusted device metadata.
    c10::cuda::CUDAGuard guard(q.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "paged_decode", [&] {
        if (d == 64) launch<scalar_t, 64>(q,k,v,table,lengths,numerator,stats,out,split_tokens,stream);
        else launch<scalar_t, 128>(q,k,v,table,lengths,numerator,stats,out,split_tokens,stream);
    });
}
