/**********************************************************************************************************************
 * Name: TorchExtensionKernel.cpp
 * Author: Matin Raayai Ardakani
 * Email: raayai.matin@gmail.com
 * Where the CUDA magic happens for the em_pre_cuda Python package.
 * Based on the code from Pytorch's tutorials: https://github.com/pytorch/extension-cpp
 **********************************************************************************************************************/
 #include "TorchExtensionKernel.h"

/**
* A getter used by the threads in each kernel to access a 3d slice stack.
*/
inline __device__ __host__ int clamp_mirror(int idx, int minIdx, int maxIdx)
{
    if(idx < minIdx) return (minIdx + (minIdx - idx));
    else if(idx > maxIdx) return (maxIdx - (idx - maxIdx));
    else return idx;
}


template<typename scalar_t>
__device__ __host__ scalar_t get_median_of_array(scalar_t* vector, int size)
{
    for (int32_t i = 0; i < size; i++) {
        for (int32_t j = i + 1; j < size; j++) {
            if (vector[i] > vector[j]) {
                scalar_t tmp = vector[i];
                vector[i] = vector[j];
                vector[j] = tmp;
            }
        }
    }
    return vector[size / 2];
}


/**
 * Applies a median filter to the image stack with a window shape of [1, 1, 2 * radZ + 1] and returns the middle slice.
 * @param imStack input image stack as a ATen CUDA tensor with float data type.
 * @param radZ the z-radius of the median filter.
 * @return the middle slice of the output of the median filter as an ATen CUDA Tensor with float data type.
 */
at::Tensor cuda_median_3d(const at::Tensor& sliceStack) {
    at::Tensor out = at::zeros_like(sliceStack[0]);
    const int32_t dimX = sliceStack.size(2), dimY = sliceStack.size(1), dimZ = sliceStack.size(0);
    const dim3 blockDim(BLOCK_DIM_LEN, BLOCK_DIM_LEN, 0);
    const dim3 gridDim((dimX / blockDim.x + ((dimX % blockDim.x) ? 1 : 0)),
            (dimY / blockDim.y + ((dimY % blockDim.y) ? 1 : 0)), 0);

    AT_DISPATCH_FLOATING_TYPES(sliceStack.type(), "__median_3d", ([&] {
        __median_3d<scalar_t><<<gridDim, blockDim>>>(
            sliceStack.data<scalar_t>(),
            out.data<scalar_t>(),
            dimX,
            dimY,
            dimZ);
      }));
    return out;
}

at::Tensor cuda_median_3d(const at::Tensor& sliceStack, const int radX, const int radY, const int radZ) {

    at::Tensor out = at::zeros_like(sliceStack);
    const int32_t dimX = sliceStack.size(2), dimY = sliceStack.size(1), dimZ = sliceStack.size(0);

    const dim3 blockDim(BLOCK_DIM_LEN, BLOCK_DIM_LEN, BLOCK_DIM_LEN);
    const dim3 gridDim(
            (dimX/blockDim.x + ((dimX%blockDim.x)?1:0)),
            (dimY/blockDim.y + ((dimY%blockDim.y)?1:0)),
            (dimZ/blockDim.z + ((dimZ%blockDim.z)?1:0)));

    AT_DISPATCH_FLOATING_TYPES(sliceStack.type(), "__median_3d", ([&] {
        __median_3d<scalar_t><<<gridDim, blockDim>>>(
                        sliceStack.data<scalar_t>(),
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
void __median_3d(scalar_t* __restrict__ imStackIn, scalar_t* __restrict__ sliceOut, int32_t dimX, int32_t dimY,
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
        windowVec[vSize++] = imStackIn[get_1d_idx(col_idx, row_idx, sht_idx + z)];

    sliceOut[get_1d_idx(col_idx, row_idx, sht_idx)] = get_median_of_array(windowVec, vSize);
}


template<typename scalar_t>
__global__
void __median_3d(scalar_t* __restrict__ imStackIn, scalar_t* __restrict__ imStackOut, int32_t dimX, int32_t dimY,
    int32_t dimZ, int32_t radX, int32_t radY, int32_t radZ)
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
        windowVec[vSize++] = imStackIn[get_1d_idx(x + col_idx, y + row_idx, z + sht_idx)];

    imStackOut[get_1d_idx(col_idx, row_idx, sht_idx)] = get_median_of_array(windowVec, vSize);
}
