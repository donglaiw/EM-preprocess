import numpy as np
cimport numpy as np

cdef extern from "cpp_flow/Coarse2FineFlowWrapper.h":
    void Coarse2FineFlowWrapper(double * vx, double * vy, double * warpI2,
                                const double * Im1, const double * Im2,
                                double alpha, double ratio, int minWidth,
                                int nOuterFPIterations, int nInnerFPIterations,
                                int nSORIterations, int colType,
                                int h, int w, int c, 
                                double warp_step, int medfilt_hsz);
    void Coarse2FineFlowWrapper_ims(double * warpI2,
                                    const double * Ims, int nIm,
                                    double alpha, double ratio, int minWidth,
                                    int nOuterFPIterations, int nInnerFPIterations,
                                    int nSORIterations, int colType,
                                    int h, int w, int c, 
                                    double warp_step,int im_step,int medfilt_hsz);

# for debug
cdef extern from "cpp_flow/MedianFilter.h":
    void padImage(double *im, double *im_p, 
            int hh, int ww, int cc, int win_hsize);
    void medianFilter(double *im, int mHeight, int imWidth, 
            int nChannels, int win_hsize);


def pad_image(np.ndarray[double, ndim=3] im not None,
                int win_hsize=5):
    cdef int h = im.shape[0]
    cdef int w = im.shape[1]
    cdef int c = im.shape[2]
    cdef np.ndarray[double, ndim=3] im_p = np.ascontiguousarray(np.zeros((h+2*win_hsize, w+2*win_hsize, c), dtype=np.float64))
    padImage(&im[0,0,0], &im_p[0,0,0], h, w, c, win_hsize)
    return im_p

def medfilt2d(np.ndarray[double, ndim=3] im not None,
                int win_hsize=5):
    cdef int h = im.shape[0]
    cdef int w = im.shape[1]
    cdef int c = im.shape[2]
    medianFilter(&im[0,0,0], h, w, c, win_hsize)

def coarse2fine_flow(np.ndarray[double, ndim=3] Im1 not None,
                     np.ndarray[double, ndim=3] Im2 not None,
                     double alpha=1, double ratio=0.5, int minWidth=40,
                     int nOuterFPIterations=3, int nInnerFPIterations=1,
                     int nSORIterations=20, int colType=0, 
                     double warp_step=1.0, int medfilt_hsz=0):
    """
    Input Format:
      double * vx, double * vy, double * warpI2,
      const double * Im1 (range [0,1]), const double * Im2 (range [0,1]),
      double alpha (1), double ratio (0.5), int minWidth (40),
      int nOuterFPIterations (3), int nInnerFPIterations (1),
      int nSORIterations (20),
      int colType (0 or default:RGB, 1:GRAY)
    Images Format: (h,w,c): float64: [0,1]
    """
    cdef int h = Im1.shape[0]
    cdef int w = Im1.shape[1]
    cdef int c = Im1.shape[2]
    cdef np.ndarray[double, ndim=2, mode="c"] vx = \
        np.ascontiguousarray(np.zeros((h, w), dtype=np.float64))
    cdef np.ndarray[double, ndim=2, mode="c"] vy = \
        np.ascontiguousarray(np.zeros((h, w), dtype=np.float64))
    cdef np.ndarray[double, ndim=3, mode="c"] warpI2 = \
        np.ascontiguousarray(np.zeros((h, w, c), dtype=np.float64))
    Im1 = np.ascontiguousarray(Im1)
    Im2 = np.ascontiguousarray(Im2)

    Coarse2FineFlowWrapper(&vx[0, 0], &vy[0, 0], &warpI2[0, 0, 0],
                            &Im1[0, 0, 0], &Im2[0, 0, 0],
                            alpha, ratio, minWidth, nOuterFPIterations,
                            nInnerFPIterations, nSORIterations, colType,
                            h, w, c, warp_step, medfilt_hsz)
    return vx, vy, warpI2

def coarse2fine_ims(np.ndarray[double, ndim=4] Ims not None,
                    double alpha=1, double ratio=0.5, int minWidth=40,
                    int nOuterFPIterations=3, int nInnerFPIterations=1,
                    int nSORIterations=20, int colType=0, 
                    double warp_step=1, int im_step=2, int medfilt_hsz=0):
    """
    Input Format:
      double * vx, double * vy, double * warpI2,
      const double * Im1 (range [0,1]), const double * Im2 (range [0,1]),
      double alpha (1), double ratio (0.5), int minWidth (40),
      int nOuterFPIterations (3), int nInnerFPIterations (1),
      int nSORIterations (20),
      int colType (0 or default:RGB, 1:GRAY)
    Images Format: (n,h,w,c): float64: [0,1]
    """
    cdef int n = Ims.shape[0]
    cdef int h = Ims.shape[1]
    cdef int w = Ims.shape[2]
    cdef int c = Ims.shape[3]
    cdef np.ndarray[double, ndim=4, mode="c"] warpI2 = \
        np.ascontiguousarray(np.zeros((n-im_step, h, w, c), dtype=np.float64))

    Ims = np.ascontiguousarray(Ims)
    Coarse2FineFlowWrapper_ims(&warpI2[0, 0, 0, 0],
                            &Ims[0, 0, 0, 0], n,
                            alpha, ratio, minWidth, nOuterFPIterations,
                            nInnerFPIterations, nSORIterations, colType,
                            h, w, c, warp_step, im_step, medfilt_hsz)
    return warpI2
