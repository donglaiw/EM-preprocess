
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

at::Tensor idm_dist(const at::Tensor &img1, const at::Tensor &img2,
        int patch_sz, int warp_sz, int step, int metric);