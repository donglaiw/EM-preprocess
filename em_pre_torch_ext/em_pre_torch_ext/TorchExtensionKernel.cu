/**********************************************************************************************************************
 * Name: TorchExtensionKernel.cpp
 * Author: Matin Raayai Ardakani, Tran Minh Quan
 * Email: raayai.matin@gmail.com
 * Where the CUDA magic happens for the em_pre_cuda Python package.
 * Based on the code from Pytorch's tutorials: https://github.com/pytorch/extension-cpp
 * and Tran Minh Quan's 3d_median filter: 
 * https://github.com/tmquan/hetero/blob/ad3c48d1b49b6f79cb06e69ae4199302efd2ffb3/research/ldav14/segment_threshold/median_3d.cu
 **********************************************************************************************************************/
 #include "TorchExtensionKernel.h"


// at::TensorAccessor getFiltRadsAccessor(const at::Tensor& filtRads) {
//     switch (tensor.type().scalarType()) {
//         case at::ScalarType::Char: return filtRads.accessor<int8_t, 1>();
//         case at::ScalarType::Int: return filtRads.accessor<int32_t, 1>();
//         case at::ScalarType::Long: return filtRads.accessor<int64_t, 1>();
//         case at::ScalarType::Short: return filtRads.accessor<int16_t, 1>();
//         case at::ScalarType::Byte: return filtRads.accessor<uint8_t, 1>();
//         case at::ScalarType::Double: return filtRads.accessor<double, 1>();
//         case at::ScalarType::Float: return filtRads.accessor<float, 1>();
//         default: thtx "The filter Radius Tensor data type not supported.\n";    
//  }

// size_t get_ten_scalar_t_size(const at::Tensor& tensor) {
//     switch (tensor.type().scalarType()) {
//         case at::ScalarType::Char: return sizeof(int8_t);
//         case at::ScalarType::Int: return sizeof(int32_t);
//         case at::ScalarType::Long: return sizeof(int64_t);
//         case at::ScalarType::Short: return sizeof(int16_t);
//         case at::ScalarType::Byte: return sizeof(uint8_t);
//         case at::ScalarType::Double: return sizeof(double);
//         case at::ScalarType::Float: return sizeof(float);
//         default: throw "The Image Stack Tensor scalar type not supported.\n";   
//     }
// }

at::Tensor cuda_median_3d(const at::Tensor& imStack, const at::Tensor& filtRads) {

    at::Tensor imStackOut = at::zeros_like(imStack);
    const int32_t dimX = imStack.size(2), dimY = imStack.size(1), dimZ = imStack.size(0);
    auto f_copy = filtRads;
    //TODO: Make this accept all types.
    auto fa = f_copy.accessor<float, 1>();
    const int32_t radX = static_cast<int32_t>(fa[2]);
    const int32_t radY = static_cast<int32_t>(fa[1]);
    const int32_t radZ = static_cast<int32_t>(fa[0]);

    const dim3 blockDim(BLOCK_DIM_LEN, BLOCK_DIM_LEN, BLOCK_DIM_LEN);
    const dim3 gridDim(
        (dimX/blockDim.x + ((dimX%blockDim.x)?1:0)),
        (dimY/blockDim.y + ((dimY%blockDim.y)?1:0)),
        (dimZ/blockDim.z + ((dimZ%blockDim.z)?1:0)));

    AT_DISPATCH_FLOATING_TYPES(imStack.type(), "__median_3d", ([&] {
        __median_3d<scalar_t><<<gridDim, blockDim>>>(
            imStack.data<scalar_t>(),
            imStackOut.data<scalar_t>(),
            dimX,
            dimY,
            dimZ,
            radX,
            radY,
            radZ);
      }));
    return imStackOut;
}

inline __device__ __host__ int clamp_mirror(int idx, int minIdx, int maxIdx)
{
    if(idx < minIdx) return (minIdx + (minIdx - idx));
    else if(idx > maxIdx) return (maxIdx - (idx - maxIdx));
    else return idx;
}

template<typename scalar_t>
__global__
void __median_3d(scalar_t* __restrict__ imStackIn, 
    scalar_t* __restrict__ imStackOut,
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
        windowVec[vSize++] = imStackIn[get_1d_idx(x + col_idx, y + row_idx, z + sht_idx)];
        
	for (int32_t i = 0; i < vSize; i++) {
		for (int32_t j = i + 1; j < vSize; j++) {
			if (windowVec[i] > windowVec[j]) { 
				scalar_t tmp = windowVec[i];
				windowVec[i] = windowVec[j];
				windowVec[j] = tmp;
			}
		}
    }

    imStackOut[get_1d_idx(col_idx, row_idx, sht_idx)] = windowVec[vSize/2];   //Set the output variables.
}
