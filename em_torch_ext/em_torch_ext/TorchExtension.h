/**
 * Name: TorchExtension.h
 * Author: Matin Raayai Ardakani
 * Email: matinraayai@seas.harvard.edu
 * Based on the code from Pytorch's tutorials: https://github.com/pytorch/extension-cpp
 **********************************************************************************************************************/
#pragma once
#include <torch/extension.h>
#include <vector>
#include "TorchExtensionKernel.h"

//Torch Tensor checks
//#define CHECK_TENSOR_IS_CUDA(x) TORCH_CHECK(x.scalar_type().is_cuda(), #x " must be a CUDA tensor")
//#define CHECK_TENSOR_IS_CPU(x) TORCH_CHECK(!x.scalar_type().is_cuda(), #x " must be a CPU tensor")
//#define CHECK_TENSOR_IS_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
//#define CHECK_CUDA_TENSOR(x) CHECK_TENSOR_IS_CUDA(x); CHECK_TENSOR_IS_CONTIGUOUS(x)
//#define CHECK_CPU_TENSOR(x) CHECK_TENSOR_IS_CPU(x); CHECK_TENSOR_IS_CONTIGUOUS(x)

/*3D Median filter needed by traditional deflicker in em_pre.=========================================================*/

torch::Tensor median_filter(const torch::Tensor& tensor, const torch::Tensor& filtRads);

torch::Tensor median_filter_v2(const torch::Tensor& tensor);

torch::Tensor median_filter_v3(const torch::Tensor& tensor, const std::vector<int>& filtRads);

/*Image deformation model distance needed in em_pre.==================================================================*/
//torch::Tensor idm_dist(const torch::Tensor& tensor1,
//                       const torch::Tensor& tensor2,
//                       int patch_size,
//                       int warp_size,
//                       int step,
//                       int metric);
