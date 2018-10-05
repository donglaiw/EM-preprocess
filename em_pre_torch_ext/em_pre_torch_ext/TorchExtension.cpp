/**********************************************************************************************************************
 * Name: TorchExtension.cpp
 * Author: Matin Raayai Ardakani
 * Email: raayai.matin@gmail.com
 * This file contains all the Python function interfaces for the package em_pre_cuda.
 * Based on the code from Pytorch's tutorials: https://github.com/pytorch/extension-cpp
 **********************************************************************************************************************/
#include "TorchExtension.h"
//TODO: Create a Hashmap that holds the documentation for each function.

at::Tensor median_filter_v2(const at::Tensor& sliceStack) {
    CHECK_TENSOR_IS_CUDA(sliceStack);
    //Check if imStack has a float ScalarType
    return cuda_median_3d(sliceStack);
}

at::Tensor median_filter(const at::Tensor& sliceStack, const at::Tensor& filtRads) {
    CHECK_TENSOR_IS_CUDA(sliceStack);
    CHECK_TENSOR_IS_CPU(filtRads);
    auto f_copy = filtRads;
    //TODO: Make this accept all types.
    auto fa = f_copy.accessor<long, 1>();
    int radX = static_cast<int>(fa[2]);
    int radY = static_cast<int>(fa[1]);
    int radZ = static_cast<int>(fa[0]);
    return cuda_median_3d(sliceStack, radX, radY, radZ);
}

at::Tensor median_filter_v3(const at::Tensor& sliceStack, const std::vector<int> filtRads) {
    CHECK_TENSOR_IS_CUDA(sliceStack);
    return cuda_median_3d(sliceStack, filtRads[2], filtRads[1], filtRads[0]);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("median_filter", &median_filter, "CUDA 3D median filter.");
  m.def("median_filter", &median_filter_v2, "CUDA 3D median filter v.2 returning only the middle slice.");
  m.def("median_filter", &median_filter_v3, "CUDA 3D median filter v.3.");
}

//std::tuple<at::Tensor, at::Tensor, at::Tensor> Coarse2FineFlow(
//        const at::Tensor& Im1, const at::Tensor& Im2,
//        double alpha, double ratio, int minWidth,
//        int nOuterFPIterations, int nInnerFPIterations,
//        int nSORIterations, int colType,
//        double warp_step, int medfilt_hsz, double flow_scale) {
//  return OpticalFlow::Coarse2FineFlow(Im1, Im2, alpha, ratio, minWidth,
//                                nOuterFPIterations, nInnerFPIterations,
//                                nSORIterations, warp_step, medfilt_hsz,
//                                flow_scale);
//
//        }
//
//
//
//std::tuple<at::Tensor, at::Tensor> Coarse2FineFlow_flows(const at::Tensor& Ims,
//                            double alpha, double ratio, int minWidth,
//                            int nOuterFPIterations, int nInnerFPIterations,
//                            int nSORIterations, int colType,
//                            double warp_step, int im_step, int medfilt_hsz, double flow_scale) {
//  DImage ImFormatted1, ImFormatted2;
//  DImage vxFormatted, vyFormatted, warpI2Formatted;
//
//  // format input in the format needed by backend
//  ImFormatted1.allocate(w, h, c);
//  ImFormatted2.allocate(w, h, c);
//  ImFormatted1.setColorType(colType);
//  ImFormatted2.setColorType(colType);
//  vxFormatted.allocate(w, h);
//  vyFormatted.allocate(w, h);
//  warpI2Formatted.allocate(w, h, c);
//
//  size_t im_size = h * w * c ;
//
//  for (int i=0;i<nIm-im_step;i++){
//      cout<<"warp image: "<<(i+1)<<"/"<<nIm-1<<endl;
//      memcpy(ImFormatted1.pData, Ims + i*im_size, im_size* sizeof(double));
//      memcpy(ImFormatted2.pData, Ims + (i+im_step)*im_size, im_size* sizeof(double));
//      // call optical flow backend
//      OpticalFlow::Coarse2FineFlow(vxFormatted, vyFormatted, warpI2Formatted,
//                                    ImFormatted1, ImFormatted2,
//                                    alpha, ratio, minWidth,
//                                    nOuterFPIterations, nInnerFPIterations,
//                                    nSORIterations, warp_step, medfilt_hsz, flow_scale);
//
//      // copy formatted output to a contiguous memory to be returned
//      memcpy(warpI2 + i*im_size, warpI2Formatted.pData, im_size* sizeof(double));
//  }
//
//  // clear c memory
//  ImFormatted1.clear();
//  ImFormatted2.clear();
//  vxFormatted.clear();
//  vyFormatted.clear();
//  warpI2Formatted.clear();
//
//  return;
//}