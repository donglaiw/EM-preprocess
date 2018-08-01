/**********************************************************************************************************************
 * Name: em_pre_cuda.cpp
 * Author: Matin Raayai Ardakani
 * Email: raayai.matin@gmail.com
 * This file contains all the Python function interfaces for the package em_pre_cuda.
 * Based on the code from Pytorch's tutorials: https://github.com/pytorch/extension-cpp
 **********************************************************************************************************************/
#include "em_pre_cuda.h"
#include <string>
//TODO: Create a Hashmap that holds the documentation for each function added here.

at::Tensor median_filter(at::Tensor input, at::Tensor filter_rads)
{
    CHECK_INPUT(input)
    CHECK_INPUT(filter_rads)
    cudaStream_t stream = 0;
    int32_t halo = 0;
    //Uncomment when passed the correct header to the function.
    //at::cuda::getDefaultCUDAStream();
    return cuda_median_3d(input, filter_rads, halo, stream);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("median_filter", &median_filter, "Applies the 3D median filter to the passed image stack on the gpu.\n:param input The torch tensor (at::Tensor) containing the image stack in the shape [im_x, im_y, batch_idx]. The tensor's dtype must be torch.float32.\n:param filter_rads 3D torch tensor containing each radius of the filter: [rad_x, rad_y, rad_z]. Each radius is the filter's half-dimension. For example, a passed filter array of torch.tensor([3.0, 0.0, 0.0]) will result in torch.tensor([7.0, 1.0, 1.0]).\n:param halo Used in Peta-byte data processing.\n:return The output of the median filter as a torch tensor with the same dimensions as the input.");
}
