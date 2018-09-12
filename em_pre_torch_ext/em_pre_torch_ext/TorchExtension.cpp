/**********************************************************************************************************************
 * Name: TorchExtension.cpp
 * Author: Matin Raayai Ardakani
 * Email: raayai.matin@gmail.com
 * This file contains all the Python function interfaces for the package em_pre_cuda.
 * Based on the code from Pytorch's tutorials: https://github.com/pytorch/extension-cpp
 **********************************************************************************************************************/
#include "TorchExtension.h"
//TODO: Create a Hashmap that holds the documentation for each function.

std::tuple<at::Tensor, at::Tensor, at::Tensor> Coarse2FineFlow(
        const at::Tensor& Im1, const at::Tensor& Im2,
        double alpha, double ratio, int minWidth,
        int nOuterFPIterations, int nInnerFPIterations,
        int nSORIterations, int colType,
        double warp_step, int medfilt_hsz, double flow_scale) {
  return OpticalFlow::Coarse2FineFlow(Im1, Im2, alpha, ratio, minWidth,
                                nOuterFPIterations, nInnerFPIterations,
                                nSORIterations, warp_step, medfilt_hsz, 
                                flow_scale);

        }



std::tuple<at::Tensor, at::Tensor> Coarse2FineFlow_flows(const at::Tensor& Ims,
                            double alpha, double ratio, int minWidth,
                            int nOuterFPIterations, int nInnerFPIterations,
                            int nSORIterations, int colType,
                            double warp_step, int im_step, int medfilt_hsz, double flow_scale) {
  DImage ImFormatted1, ImFormatted2;
  DImage vxFormatted, vyFormatted, warpI2Formatted;

  // format input in the format needed by backend
  ImFormatted1.allocate(w, h, c);
  ImFormatted2.allocate(w, h, c);
  ImFormatted1.setColorType(colType);
  ImFormatted2.setColorType(colType);
  vxFormatted.allocate(w, h);
  vyFormatted.allocate(w, h);
  warpI2Formatted.allocate(w, h, c);

  size_t im_size = h * w * c ;
  
  for (int i=0;i<nIm-im_step;i++){
      cout<<"warp image: "<<(i+1)<<"/"<<nIm-1<<endl;
      memcpy(ImFormatted1.pData, Ims + i*im_size, im_size* sizeof(double));
      memcpy(ImFormatted2.pData, Ims + (i+im_step)*im_size, im_size* sizeof(double));
      // call optical flow backend
      OpticalFlow::Coarse2FineFlow(vxFormatted, vyFormatted, warpI2Formatted,
                                    ImFormatted1, ImFormatted2,
                                    alpha, ratio, minWidth,
                                    nOuterFPIterations, nInnerFPIterations,
                                    nSORIterations, warp_step, medfilt_hsz, flow_scale);

      // copy formatted output to a contiguous memory to be returned
      memcpy(warpI2 + i*im_size, warpI2Formatted.pData, im_size* sizeof(double));
  }

  // clear c memory
  ImFormatted1.clear();
  ImFormatted2.clear();
  vxFormatted.clear();
  vyFormatted.clear();
  warpI2Formatted.clear();

  return;
}











at::Tensor median_filter_v2(const at::Tensor& imStack) {
    CHECK_INPUT_CUDA(imStack);
    //Check if imStack has a float ScalarType
    return cuda_median_3d(imStack);
}


at::Tensor median_filter(const at::Tensor& imStack, const at::Tensor& filtRads) {
    CHECK_INPUT_CUDA(imStack);
    CHECK_INPUT_CPU(filtRads);
    return cuda_median_3d(imStack, filtRads);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("median_filter", &median_filter, "CUDA 3D median filter.");
  m.def("median_filter", &median_filter_v2, "CUDA 3D median filter v.2 returning only the middle slice.");

}
