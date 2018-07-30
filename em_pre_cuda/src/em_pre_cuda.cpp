#include "em_pre_cuda.h"

at::Tensor median_filter(at::Tensor input, at::Tensor filter)
{
    CHECK_INPUT(input)
    CHECK_INPUT(filter)
    int32_t halo = 0;
    cudaStream_t stream = 0;
    return cuda_median_3d(input, filter, halo, stream);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("median_filter", &median_filter, "3D median filter (CUDA)");
}
