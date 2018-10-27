#pragma once
#include "idm_main.h"

template<typename scalar_t>
scalar_t patch_distance (int A_x,int A_y, int B_x, int B_y, int im_row, int im_col, int im_chan,
        int patch_sz, scalar_t *img1, scalar_t *img2, bool metric) {
    scalar_t dist = 0, temp_h;
    int c, x, y, count = 0;
    /* only move around patchB */
    scalar_t patch_sum = 0;
    for(y = -patch_sz; y <= patch_sz; y++) {
    for(x = -patch_sz; x <= patch_sz; x++) {
    if(check_bounds(A_x + x, A_y + y, im_row, im_col) && check_bounds(B_x + x, B_y + y, im_row, im_col)) {
    for(int c = 0; c < im_chan; c++) {
        temp_h = img1[get_1d_idx(A_x + x, A_y + y, c, im_row, im_col, im_chan)] - 
        img2[get_1d_idx(A_x + x, A_y + y, c, im_row, im_col, im_chan)];
        dist += std::abs(temp_h);
        patch_sum += img1[get_1d_idx(A_x + x, A_y + y, c, im_row, im_col, im_chan)]
        count++;
    }}}}
    if (metric):
        dist = dist /= patch_sum;
    return dist / count;
}




at::Tensor idm_dist(const at::Tensor &img1, const at::Tensor &img2,
                    int patch_sz, int warp_sz, int step, bool metric) {
    const int32_t dimX = img1.size(2), dimY = img1.size(1), dimZ = img1.size(0);
    auto distDims = {(int) std::ceil(dimX / (float) patch_step), (int) std:ceil(dimY / (float) patch_step)}
    at::Tensor dist = at::CUDA(img1.type()).zeros(distDims);
    const dim3 blockDim(BLOCK_DIM_LEN, BLOCK_DIM_LEN, BLOCK_DIM_LEN);
    const dim3 gridDim(
            (dimX/blockDim.x + ((dimX%blockDim.x)?1:0)),
            (dimY/blockDim.y + ((dimY%blockDim.y)?1:0)),
            (dimZ/blockDim.z + ((dimZ%blockDim.z)?1:0)));

    AT_DISPATCH_FLOATING_TYPES(sliceStack.type(), "__idm_", ([&] {
        __idm_<scalar_t><<<gridDim, blockDim>>>(
            img1.data<scalar_t>(),
            img2.data<scalar_t>(),
            dist.data<scalar_t>(),
            dimX,
            dimY,
            dimZ,
            patch_sz,
            wrap_sz,
            step,
            metric);
    }));
    return dist;

}


template<typename scalar_t>
__global__
void __idm_(scalar_t* __restrict__ img1, scalar_t* __restrict__ img2, scalar_t* __restrict__ dist,
        int32_t dimX, int32_t dimY, int32_t dimZ, int32_t patch_sz, int32_t warp_sz, int32_t step, bool metric)
{
    scalar_t best_dis = std::numeric_limits<scalar_t>::max();
    const int32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t col_idx = blockIdx.y * blockDim.y + threadIdx.y;
    for(int32_t x = row_idx - warp_sz; x <= row_idx + warp_sz; x++) {
    for (int32_t y = col_idx - warp_sz; y <= col_idx + warp_sz; y++) {
    if (check_bounds(x, y, dimX, dimY)) {
        temp = patch_distance(row_idx, col_idx, x, y, dimX, dimY, dimZ,
                                    patch_sz, img1, img2, metric);
        if (temp < best_dis)
            best_dis = temp;
    }}}
    if (get_1d_idx(row_idx, col_idx) % step == 0)
        dist[get_1d_idx(row_idx, col_idx, dimX, dimY) / step] = best_dis;
}