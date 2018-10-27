/**********************************************************************************************************************
 * Name: em_pre_cuda_kernel.h
 * Author: Matin Raayai Ardakani
 * Email: raayai.matin@gmail.com
 **********************************************************************************************************************/
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define BLOCK_DIM_LEN 8
// This constant is used to create an array of numbers in each GPU kernel to be sorted in the 3d median filter.
// Ideally, it shouldn't be present since it would limit the window size of the 3D median filter and also allocating
// static memory for each kernel is bad. There are implementations of a 3D median filter out there that use heuristics
// to "converge" towards the median after a certain amount of iteration. If proved to work as well as the median filter
// implemented below, one shouldn't hesitate to replace it.
// It's also an obvious fact that specifying a median filter window with a volume bigger than the constant below will
// result in undefined behavior.
#define MAX_GPU_ARRAY_LEN 20


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


/**
 * Calculates the median of a vector by sorting it and returning the middle element of the vector.
 * @tparam scalar_t type of the elements in the array
 * @param vector the desired vector
 * @param size the size of the vector
 * @return the median of the vector.
 * */
template<typename scalar_t>
inline __device__ __host__ scalar_t calculate_median(scalar_t* &vector, int32_t size)
{
    for (int32_t i = 0; i < size; i++){
    for (int32_t j = i + 1; j < size; j++){
        if (vector[i] > vector[j]) {
            scalar_t tmp = vector[i];
            vector[i] = vector[j];
            vector[j] = tmp;
        }
    }}
    return vector[size / 2];
}


/**
 * Applies a median filter to the image stack along the z-axis and returns the middle slice. The filter's
 * z radius is equal to half the z dimension of the imStack.
 * @param imStack a stack of images with z being the inner most dimension.
 * @return the middle slice of the median filter output.
 */
at::Tensor median_filter_cuda(const at::Tensor &imStack);

/**
 * Applies a median filter with window dimensions specified in filtRads.
 * @param imStack The desired input image stack.
 * @param filtRads The radius dims of the median filter.
 * @return the result of the median filter operation.
 */
at::Tensor median_filter_cuda(const at::Tensor &imStack, const at::Tensor &filtRads);

/**
 * Calculates the median filter of stackIn along the z-axis and returns the middle slice of the result in stackOut. The
 * diameter of the median filter window along the z axis is equal the z dimension of the stackIn.
 * @tparam scalar_t data type of the image.
 * @param stackIn pointer to the first element of the input image stack.
 * @param imOut pointer to the first element of the output image (middle slice of the result.)
 * @param dimX X dimension of the input image.
 * @param dimY Y dimension of the input image.
 * @param dimZ Z dimension of  the input image.
 */
template<typename scalar_t>
__global__
void median_filter_kernel(scalar_t* __restrict__ stackIn,
                          scalar_t* __restrict__ imOut,
                          int32_t dimX,
                          int32_t dimY,
                          int32_t dimZ);

/**
 * Calculates the median filter of stackIn using the rad dimensions as median filter rads and returns the
 * result with the same dimensions as stackIn in stackOut.
 * @tparam scalar_t data type of the image stack.
 * @param stackIn Pointer to the beginning of the input image stack.
 * @param stackOut Pointer to the beginning of the output image stack.
 * @param dimX X dimension of the input stack.
 * @param dimY Y dimension of the input stack.
 * @param dimZ Z dimension of the input stack.
 * @param radX X radius of the median filter window.
 * @param radY Y radius of the median filter window.
 * @param radZ Z radius of the median filter window.
 */
template<typename scalar_t>
__global__
void median_filter_kernel(scalar_t* __restrict__ stackIn,
    scalar_t* __restrict__ stackOut,
    int32_t dimX,
    int32_t dimY,
    int32_t dimZ,
    int32_t radX,
    int32_t radY,
    int32_t radZ);
