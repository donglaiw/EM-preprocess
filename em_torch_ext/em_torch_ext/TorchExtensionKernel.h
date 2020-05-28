/**********************************************************************************************************************
 * Name: em_pre_cuda_kernel.h
 * Author: Matin Raayai Ardakani
 * Email: matinraayai@seas.harvard.edu
 * Header file for em_pre_cuda_kernel.h
 * Based on the code from Pytorch's tutorials: https://github.com/pytorch/extension-cpp
 **********************************************************************************************************************/
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define BLOCK_DIM_LEN 8
#define MAX_GPU_ARRAY_LEN 10

torch::Tensor cuda_median_3d(const torch::Tensor& imStack);

torch::Tensor cuda_median_3d(const torch::Tensor& imStack, const int radX, const int radY, const int radZ);
