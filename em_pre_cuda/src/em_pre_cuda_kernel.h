#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <tuple>
#include "/usr/local/cuda/samples/common/inc/helper_math.h"

at::Tensor cuda_median_3d(at::Tensor input, at::Tensor filter, int32_t halo, cudaStream_t stream);

template <typename scalar_t>
__global__ void __median_3d(scalar_t* __restrict__ deviceSrc, 
    scalar_t* __restrict__ deviceDst,
    scalar_t* __restrict__ filter, 
    int32_t dim_x, int32_t dim_y, int32_t dim_z, int32_t halo); 
