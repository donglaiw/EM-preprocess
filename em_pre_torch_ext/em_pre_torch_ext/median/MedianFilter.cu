/**********************************************************************************************************************
 * Name: TorchExtensionKernel.cpp
 * Author: Matin Raayai Ardakani
 * Email: raayai.matin@gmail.com
 * Contains the CUDA kernels written using Pytorch's Aten backend.
 * For function documentation, refer to the associated header file.
 **********************************************************************************************************************/
 #include "MedianFilter.h"


at::Tensor median_filter_cuda(const at::Tensor& imStack) {
    at::Tensor out = at::zeros_like(imStack[0]);
    const int32_t dimX = imStack.size(2), dimY = imStack.size(1), dimZ = imStack.size(0);
    const dim3 blockDim(BLOCK_DIM_LEN, BLOCK_DIM_LEN, 1);
    const dim3 gridDim((dimX / blockDim.x + ((dimX % blockDim.x) ? 1 : 0)),
            (dimY / blockDim.y + ((dimY % blockDim.y) ? 1 : 0)), 1);

    AT_DISPATCH_FLOATING_TYPES(imStack.type(), "median_filter_kernel", ([&] {
        median_filter_kernel<scalar_t><<<gridDim, blockDim>>>(
            imStack.data<scalar_t>(),
            out.data<scalar_t>(),
            dimX,
            dimY,
            dimZ);
      }));
    return out;
}

at::Tensor median_filter_cuda(const at::Tensor& imStack, const int radX, const int radY, const int radZ) {

    at::Tensor out = at::zeros_like(imStack);
    const int32_t dimX = imStack.size(2), dimY = imStack.size(1), dimZ = imStack.size(0);

    const dim3 blockDim(BLOCK_DIM_LEN, BLOCK_DIM_LEN, BLOCK_DIM_LEN);
    const dim3 gridDim(
            (dimX/blockDim.x + ((dimX%blockDim.x)?1:0)),
            (dimY/blockDim.y + ((dimY%blockDim.y)?1:0)),
            (dimZ/blockDim.z + ((dimZ%blockDim.z)?1:0)));

    AT_DISPATCH_FLOATING_TYPES(imStack.type(), "median_filter_kernel", ([&] {
        median_filter_kernel<scalar_t><<<gridDim, blockDim>>>(
                        imStack.data<scalar_t>(),
                        out.data<scalar_t>(),
                        dimX,
                        dimY,
                        dimZ,
                        radX,
                        radY,
                        radZ);
    }));
    return out;
}

template<typename scalar_t>
__global__
void median_filter_kernel(scalar_t* __restrict__ stackIn,
        scalar_t* __restrict__ imOut,
        int32_t dimX,
        int32_t dimY,
        int32_t dimZ) {
    auto get_1d_idx = [&] (int32_t x, int32_t y, int32_t z) {
        return clamp_mirror(z, 0, dimZ - 1) * dimY * dimX +
        clamp_mirror(y, 0, dimY - 1) * dimX + clamp_mirror(x, 0, dimX - 1);
    };

    const int32_t col_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t sht_idx = dimZ / 2;
	scalar_t windowVec[MAX_GPU_ARRAY_LEN] = {0.};
    int32_t vSize = 0;

    for (int32_t z = -dimZ; z <= dimZ; z++)
        windowVec[vSize++] = stackIn[get_1d_idx(col_idx, row_idx, sht_idx + z)];

    imOut[get_1d_idx(col_idx, row_idx, sht_idx)] = get_median_of_array(windowVec, vSize);
}


template<typename scalar_t>
__global__
void __median_3d(scalar_t* __restrict__ stackIn,
        scalar_t* __restrict__ stackOut,
        int32_t dimX,
        int32_t dimY,
        int32_t dimZ,
        int32_t radX,
        int32_t radY,
        int32_t radZ)
    {
    auto get_1d_idx = [&] (int32_t x, int32_t y, int32_t z) {
        return clamp_mirror(z, 0, dimZ - 1) * dimY * dimX + 
        clamp_mirror(y, 0, dimY - 1) * dimX + clamp_mirror(x, 0, dimX - 1);
    };

    const int32_t col_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t sht_idx = blockIdx.z * blockDim.z + threadIdx.z;

	scalar_t windowVec[MAX_GPU_ARRAY_LEN] = {0.};
    int32_t vSize = 0;

    for (int32_t z = -radZ; z <= radZ; z++)
    for (int32_t y = -radY; y <= radY; y++)
    for (int32_t x = -radX; x <= radX; x++)
        windowVec[vSize++] = stackIn[get_1d_idx(x + col_idx, y + row_idx, z + sht_idx)];

    stackOut[get_1d_idx(col_idx, row_idx, sht_idx)] = calculate_median(windowVec, vSize);
}
