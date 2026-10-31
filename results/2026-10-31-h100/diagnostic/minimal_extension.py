import torch
from torch.utils.cpp_extension import load_inline
m=load_inline(name='independent_sanitizer_probe',cpp_sources='void launch(torch::Tensor x);',cuda_sources='''#include <torch/extension.h>
#include <cuda_runtime.h>
__global__ void basic(float*x){x[threadIdx.x]=threadIdx.x;}
void launch(torch::Tensor x){basic<<<1,32>>>(x.data_ptr<float>());auto e=cudaGetLastError();TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e));}
''',functions=['launch'],verbose=False)
x=torch.empty(32,device='cuda');m.launch(x);torch.cuda.synchronize();torch.testing.assert_close(x,torch.arange(32,device='cuda',dtype=torch.float));print('INDEPENDENT_EXTENSION_PASSED')
