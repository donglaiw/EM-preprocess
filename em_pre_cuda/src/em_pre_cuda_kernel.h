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
#include <vector>
#include <tuple>
#include "/usr/local/cuda-9.2/samples/common/inc/helper_math.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

at::Tensor cuda_median_3d(at::Tensor deviceSrc, at::Tensor deviceDst, int dimx, int dimy, int dimz, int radius, int halo, cudaStream_t stream);

template <typename scalar_t>
__global__
void __median_3d(scalar_t* deviceSrc, scalar_t* deviceDst, int dimx, int dimy, int dimz, int radius, int halo);
