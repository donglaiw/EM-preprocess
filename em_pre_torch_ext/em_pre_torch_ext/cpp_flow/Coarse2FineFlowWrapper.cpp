// This is a wrapper for Ce Liu's Coarse2Fine optical flow implementation.
// It converts the contiguous image array to the format needed by the optical
// flow code. Handling conversion in the wrapper makes the cythonization
// simpler.
// Author: Deepak Pathak (c) 2016

#include "Coarse2FineFlowWrapper.h"
using namespace std;

std::tuple<at::Tensor, at::Tensor, at::Tensor> Coarse2FineFlowWrapper(
        const at::Tensor& Im1, const at::Tensor& Im2,
        double alpha, double ratio, int minWidth,
        int nOuterFPIterations, int nInnerFPIterations,
        int nSORIterations, int colType,
        double warp_step, int medfilt_hsz, double flow_scale) {
    //cout<<"do 1"<<endl;
  //cout<<h<<","<<w<<","<<c<<endl;
  //cout<<alpha<<","<<ratio<<","<<minWidth<<","<<nOuterFPIterations<<","<<nInnerFPIterations<<","<<nSORIterations<<","<<warp_step<<","<<medfilt_hsz<<","<<flow_scale<<endl;
  return OpticalFlow::Coarse2FineFlow(Im1, Im2, alpha, ratio, minWidth,
                                nOuterFPIterations, nInnerFPIterations,
                                nSORIterations, warp_step, medfilt_hsz, 
                                flow_scale);

 }



std::tuple<at::Tensor, at::Tensor>
void Coarse2FineFlowWrapper_flows(double * warpI2,
                              const double * Ims, int nIm,
                              double alpha, double ratio, int minWidth,
                              int nOuterFPIterations, int nInnerFPIterations,
                              int nSORIterations, int colType,
                              int h, int w, int c, 
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
