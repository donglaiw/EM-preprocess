/**
 * Name: em_pre_cuda.h
 * Author: Matin Raayai Ardakani
 * Email: raayai.matin@gmail.com
 * Based on the code from Pytorch's tutorials: https://github.com/pytorch/extension-cpp
 **********************************************************************************************************************/
#pragma once
#include <torch/torch.h>
#include <vector>
at::Tensor cuda_median_3d(at::Tensor deviceSrc, at::Tensor deviceDst, int dimx, int dimy, int dimz, int radius, int halo, cudaStream_t stream);

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x) 
