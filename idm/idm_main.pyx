def imd_main(np.ndarray[np.float32_t, ndim=3] im1, 
             np.ndarray[np.float32_t, ndim=3] im2, 
             psz=11, wsz=5):
    cdef np.ndarray[np.float32, ndim=1] dist
    return dist

cdef extern from "src/idm.h":
    void idm_dist(float *img1, float *img2, float *out, 
            int im_row, int im_col, int patch_row, int patch_col,
            int warp_row, int warp_col);
