import numpy as np
cimport numpy as np

def imd_main(np.ndarray[np.float32_t, ndim=3] im1, 
             np.ndarray[np.float32_t, ndim=3] im2, 
             psz=11, wsz=5):
    cdef np.ndarray[np.float32_t, ndim=1] dist
    #
    # Guarantee or convert im1 and im2 to c-contiguous
    im1 = np.ascontiguousarray(im1)
    im2 = np.ascontiguousarray(im2)
    #
    # Make a C-contiguous array as the output
    #
    dist = np.zeros_like(im1, order="C")
    idm_dist(<float *>(im1.data),
             <float *>(im2.data),
             <float *>(dist.data),
             im1.shape[0], im1.shape[1], psz, psz,
             wsz, wsz)
    return dist

cdef extern from "src/idm.h":
    void idm_dist(float *img1, float *img2, float *out, 
            int im_row, int im_col, int patch_row, int patch_col,
            int warp_row, int warp_col);
