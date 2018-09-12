// This is a wrapper for Ce Liu's Coarse2Fine optical flow implementation.
// It converts the contiguous image array to the format needed by the optical
// flow code. Handling conversion in the wrapper makes the cythonization
// simpler.
// Author: Deepak Pathak (c) 2016

// override-include-guard
#include "Image.h"
#include "OpticalFlow.h"
#include <torch.torch.h>
#include <tuple>
//vx, vy, wrapI2
std::tuple<at::Tensor, at::Tensor, at::Tensor> Coarse2FineFlowWrapper(
        const at::Tensor& Im1, const at::Tensor& Im2,
        double alpha, double ratio, int minWidth,
        int nOuterFPIterations, int nInnerFPIterations,
        int nSORIterations, int colType, 
        double warp_step, int medfilt_hsz, double flow_scale);
//
std::tuple<at::Tensor, at::Tensor> Coarse2FineFlowWrapper_flows(
                                  at::Tensor& Ims, int nIm,
                                  double alpha, double ratio, int minWidth,
                                  int nOuterFPIterations, int nInnerFPIterations,
                                  int nSORIterations, int colType, 
                                  double warp_step, int im_step, int medfilt_hsz,
                                  double flow_scale);

