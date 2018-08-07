/**********************************************************************************************************************
 * Name: em_pre_cuda.cpp
 * Author: Matin Raayai Ardakani
 * Email: raayai.matin@gmail.com
 * This file contains all the Python function interfaces for the package em_pre_cuda.
 * Based on the code from Pytorch's tutorials: https://github.com/pytorch/extension-cpp
 **********************************************************************************************************************/
#include "em_pre_cuda.h"
//TODO: Create a Hashmap that holds the documentation for each function.

at::Tensor median_filter(const at::Tensor& imStack, const at::Tensor& filtRads) {
    CHECK_INPUT_CUDA(imStack);
    CHECK_INPUT_CPU(filtRads);
    int32_t halo = 0;
    return cuda_median_3d(imStack, filtRads, halo);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("median_filter", &median_filter, "CUDA 3D median filter.");
}
