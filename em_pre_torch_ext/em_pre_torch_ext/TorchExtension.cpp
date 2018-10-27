/**********************************************************************************************************************
 * Name: TorchExtension.cpp
 * Author: Matin Raayai Ardakani
 * Email: raayai.matin@gmail.com
 * This file contains all the Python function interfaces for the package em_pre_cuda.
 * Based on the code from Pytorch's tutorials: https://github.com/pytorch/extension-cpp
 **********************************************************************************************************************/
#include "TorchExtension.h"

at::Tensor median_filter_tensor_rads(const at::Tensor& sliceStack, const at::Tensor& filtRads) {
    CHECK_TENSOR_IS_CUDA(sliceStack);
    CHECK_TENSOR_IS_CPU(filtRads);
    auto f_copy = filtRads;
    //TODO: Make this accept all types.
    auto fa = f_copy.accessor<long, 1>();
    int radX = static_cast<int>(fa[2]);
    int radY = static_cast<int>(fa[1]);
    int radZ = static_cast<int>(fa[0]);
    return median_filter_cuda(sliceStack, radX, radY, radZ);
}

at::Tensor median_filter_vector_rads(const at::Tensor& sliceStack, const std::vector<int> filtRads) {
    CHECK_TENSOR_IS_CUDA(sliceStack);
    return median_filter_cuda(sliceStack, filtRads[2], filtRads[1], filtRads[0]);
}


at::Tensor median_filter_middle_slice_only(const at::Tensor& sliceStack) {
    CHECK_TENSOR_IS_CUDA(sliceStack);
    //Check if imStack has a float ScalarType
    return median_filter_cuda(sliceStack);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("median_filter", &median_filter_tensor_rads, "CUDA 3D median filter.");
  m.def("median_filter", &median_filter_vector_rads, "CUDA 3D median filter. ");
  m.def("median_filter", &median_filter_middle_slice_only, "CUDA 3D median filter.");
}