/**********************************************************************************************************************
 * Name: em_pre_cuda_kernel.h
 * Author: Matin Raayai Ardakani
 * Email: raayai.matin@gmail.com
 * Header file for em_pre_cuda_kernel.h
 * Based on the code from Pytorch's tutorials: https://github.com/pytorch/extension-cpp
 **********************************************************************************************************************/
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define BLOCK_DIM_LEN 8
#define MAX_GPU_ARRAY_LEN 200

at::Tensor cuda_median_3d(const at::Tensor& imStack, const at::Tensor& filtRads);

template<typename scalar_t>
__global__
void __median_3d(scalar_t* __restrict__ stackIn, 
    scalar_t* __restrict__ stackOut,
    int32_t dimX,
    int32_t dimY,
    int32_t dimZ,
    int32_t radX,
    int32_t radY,
    int32_t radZ);
