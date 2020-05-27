/**
 * Name: TorchExtension.h
 * Author: Matin Raayai Ardakani
 * Email: raayai.matin@gmail.com
 * Based on the code from Pytorch's tutorials: https://github.com/pytorch/extension-cpp
 **********************************************************************************************************************/
#pragma once
#include <torch/torch.h>
#include <vector>

at::Tensor cuda_median_3d(const at::Tensor& sliceStack);

at::Tensor cuda_median_3d(const at::Tensor& sliceStack, int radX, int radY, int radZ);


//#define CHECK_TENSOR_IS_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
//#define CHECK_TENSOR_IS_CPU(x) AT_CHECK(!x.type().is_cuda(), #x " must be a CPU tensor")
//#define CHECK_TENSOR_IS_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
//#define CHECK_CUDA_TENSOR(x) CHECK_TENSOR_IS_CUDA(x); CHECK_TENSOR_IS_CONTIGUOUS(x)
//#define CHECK_CPU_TENSOR(x) CHECK_TENSOR_IS_CPU(x); CHECK_TENSOR_IS_CONTIGUOUS(x)