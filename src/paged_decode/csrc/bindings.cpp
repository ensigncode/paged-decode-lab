#include <torch/extension.h>

void decode_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                 torch::Tensor table, torch::Tensor lengths,
                 torch::Tensor numerator, torch::Tensor stats,
                 torch::Tensor out, int64_t split_tokens, int64_t merge_mode);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode", &decode_cuda, "Paged split-K GQA decode (CUDA)");
}
