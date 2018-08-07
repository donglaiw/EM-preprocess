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
#include "/usr/local/cuda-9.2/samples/common/inc/helper_math.h"

at::Tensor cuda_median_3d(const at::Tensor& imStack, const at::Tensor& filtRads, int32_t halo);

template<typename scalar_t>
__global__
void __median_3d(scalar_t* __restrict__ stackIn, 
    scalar_t* __restrict__ stackOut,
    int32_t dimX,
    int32_t dimY,
    int32_t dimZ,
    int32_t radX,
    int32_t radY,
    int32_t radZ,
    int32_t halo);
