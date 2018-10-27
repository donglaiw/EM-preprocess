#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

inline __host__ __device__ int32_t get_1d_idx(int32_t x, int32_t y, int32_t dimX, int32_t dimY) {
    return y * dimX + x; 
}

inline __host__ __device__ int32_t get_1d_idx(int32_t x, int32_t y, int32_t z, int32_t dimX, 
    int32_t dimY, int32_t dimZ) {
    return z * dimY * dimX + y * dimX + x;
}

inline __host__ __device__ bool check_bounds(int32_t x, int32_t y, int32_t z, int32_t dimX, 
    int32_t dimY, int32_t dimZ) {
    return x >= 0 && y >= 0 && z >= 0 && x < dimX && y < dimY && z < dimZ;
}

inline __host__ __device__ bool check_bounds(int32_t x, int32_t y, int32_t dimX, int32_t dimY) {
    return x >= 0 && y >= 0 && x < dimX && y < dimY;
}

/**
 * Makes sure the given index is "clamped" within the specified minIdx and maxIdx by mirroring the idx with respect
 * to the bounds until it lands in the idx range of the array recursively. It's used to create a "mirroring" effect
 * for the median filter.
 * @param idx desired index of the array to be accessed.
 * @param minIdx the first index of the array. If the idx is smaller than this value, minIdx + (minIdx - idx) will be
 * the potential index within bounds.
 * @param maxIdx the last index of the array. If the idx is larger than this value, maxIdx - (idx - maxIdx) will be the
 * potential index within bounds.
 * @return an index within the specified bounds that satisfies the mirroring behavior of the median filter.
 */
inline __device__ __host__ int32_t clamp_mirror(int32_t idx, int32_t minIdx, int32_t maxIdx)
{
    if(idx < minIdx) return clamp_mirror(minIdx + (minIdx - idx), minIdx, maxIdx);
    else if(idx > maxIdx) return clamp_mirror(maxIdx - (idx - maxIdx), minIdx, maxIdx);
    else return idx;
}

inline __device__ __host__ int32_t get_mirrored_idx(int32_t x, int32_t y, int32_t z, int32_t dimX, 
    int32_t dimY, int32_t dimZ) {
    return clamp_mirror(z, 0, dimZ - 1) * dimY * dimX +
        clamp_mirror(y, 0, dimY - 1) * dimX + clamp_mirror(x, 0, dimX - 1);
}