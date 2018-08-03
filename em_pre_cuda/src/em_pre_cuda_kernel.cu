/********************************`**************************************************************************************
 * Name: em_pre_cuda_kernel.cpp
 * Author: Matin Raayai Ardakani, Tran Minh Quan
 * Email: raayai.matin@gmail.com
 * Where the CUDA magic happens for the em_pre_cuda Python package.
 * Based on the code from Pytorch's tutorials: https://github.com/pytorch/extension-cpp
 * and Tran Minh Quan's 3d_median filter: 
 * https://github.com/tmquan/hetero/blob/ad3c48d1b49b6f79cb06e69ae4199302efd2ffb3/research/ldav14/segment_threshold/median_3d.cu
 **********************************************************************************************************************/
 #include "em_pre_cuda_kernel.h"


at::Tensor cuda_median_3d(
    at::Tensor deviceSrc, 
    at::Tensor deviceDst, 
    int dim_x, 
    int dim_y, 
    int dim_z, 
    int radius, 
    int halo, 
    cudaStream_t stream)
{
    dim3 blockDim(8, 8, 8);
    dim3 gridDim(
            (dim_x/blockDim.x + ((dim_x%blockDim.x)?1:0)),
            (dim_y/blockDim.y + ((dim_y%blockDim.y)?1:0)),
            (dim_z/blockDim.z + ((dim_z%blockDim.z)?1:0)) );
    size_t sharedMemSize  = (blockDim.x+2*halo)*(blockDim.y+2*halo)*(blockDim.z+2*halo)*sizeof(float);
    // auto temp = deviceSrc.data<double>();
    // for (int32_t i = 0; i <= 11; i++) {
    //     if (temp[i] != 0)
    //         printf("%f\n",temp[i]);
    // }    
        
    AT_DISPATCH_FLOATING_TYPES(deviceSrc.type(), "__median_3d", ([&] {
        __median_3d<scalar_t><<<gridDim, blockDim, sharedMemSize, stream>>>(
            deviceSrc.data<scalar_t>(),
            deviceDst.data<scalar_t>(),
            dim_x,
            dim_y,
            dim_z,
            radius,
            halo);
      }));
    return deviceDst;
}

inline __device__ __host__ int clamp_mirror(int f, int a, int b)
{
    if(f<a) return (a+(a-f));
    if(f>b) return (b-(f-b));
    return f;
}


template <typename scalar_t>
__global__ void __median_3d(scalar_t* deviceSrc, 
    scalar_t* deviceDst,
    int dim_x, 
    int dim_y, 
    int dim_z, 
    int radius, 
    int halo) {
    extern __shared__ float sharedMemSrc[];
    int  shared_index_1d, global_index_1d, index_1d;
    int3 shared_index_3d, global_index_3d, index_3d;
    // Multi batch reading here
    int3 sharedMemDim    = make_int3(blockDim.x+2*halo,
                                     blockDim.y+2*halo,
                                     blockDim.z+2*halo);
    int  sharedMemSize   = sharedMemDim.x*sharedMemDim.y*sharedMemDim.z;
    int3 blockSizeDim    = make_int3(blockDim.x+0*halo,
                                     blockDim.y+0*halo,
                                     blockDim.z+0*halo);
    int  blockSize        = blockSizeDim.x*blockSizeDim.y*blockSizeDim.z;
    int  numBatches       = sharedMemSize/blockSize + ((sharedMemSize%blockSize)?1:0);

    auto _3d_mem_idx = [sharedMemDim](int32_t x, int32_t y, int32_t z) { 
        return  clamp_mirror(z, 0, sharedMemDim.z - 1) * sharedMemDim.y * sharedMemDim.x + 
                clamp_mirror(y, 0, sharedMemDim.y - 1) * sharedMemDim.x +
                clamp_mirror(x, 0, sharedMemDim.x - 1); 
    };
    for(int batch=0; batch<numBatches; batch++) {
        shared_index_1d  =  threadIdx.z * blockDim.y * blockDim.x +
                            threadIdx.y * blockDim.x +
                            threadIdx.x +
                            blockSize*batch; //Magic is here quantm@unist.ac.kr
        shared_index_3d  =  make_int3((shared_index_1d % ((blockDim.y+2*halo)*(blockDim.x+2*halo))) % (blockDim.x+2*halo),
                                      (shared_index_1d % ((blockDim.y+2*halo)*(blockDim.x+2*halo))) / (blockDim.x+2*halo),
                                      (shared_index_1d / ((blockDim.y+2*halo)*(blockDim.x+2*halo))) );
        global_index_3d  =  make_int3(clamp_mirror(blockIdx.x * blockDim.x + shared_index_3d.x - halo, 0, dim_x-1),
                                      clamp_mirror(blockIdx.y * blockDim.y + shared_index_3d.y - halo, 0, dim_y-1),
                                      clamp_mirror(blockIdx.z * blockDim.z + shared_index_3d.z - halo, 0, dim_z-1) );

        global_index_1d  =  global_index_3d.z * dim_y * dim_x +
                            global_index_3d.y * dim_x +
                            global_index_3d.x;
        if (shared_index_3d.z < (blockDim.z + 2*halo)) {
            if(global_index_3d.z >= 0 && global_index_3d.z < dim_z &&
               global_index_3d.y >= 0 && global_index_3d.y < dim_y &&
               global_index_3d.x >= 0 && global_index_3d.x < dim_x) {
                sharedMemSrc[_3d_mem_idx(shared_index_3d.x, shared_index_3d.y, shared_index_3d.z)] = deviceSrc[global_index_1d];
            }
            else {
                sharedMemSrc[_3d_mem_idx(shared_index_3d.x, shared_index_3d.y, shared_index_3d.z)] = -100.0f;
            }
        }
        __syncthreads();
    }
    // Stencil  processing here
    float result = sharedMemSrc[_3d_mem_idx(threadIdx.x + halo, threadIdx.y + halo, threadIdx.z + halo)];
    // # Viola's method
    // minval = 0
    // maxval = 255
    // pivot  = (minval + maxval)/2.0

    // count 	= 0
    // val 	= 0
    // for trial in range(0, 8):
    // radius = 3
    // cube_iter = make_cube_iter(x, y, z, radius)
    // count 	= 0
    // for point in cube_iter:
    // val = point_query_3d(volume, point)
    // if val > pivot:
    // count = count + 1

    // if count < (2*radius+1)*(2*radius+1)*(2*radius+1)/2:
    // maxval = floorf(pivot);
    // else:
    // minval = floorf(pivot)+1; 
    // pivot = (minval + maxval)/2.0;

    // return floorf(pivot)
    float minval = 0.0f;
    float maxval = 255.0f;
    float pivot  = (minval + maxval)/2.0f;
    float val;
    int count  = 0;
    const int array_size = (2*radius+1)*(2*radius+1)*(2*radius+1);
    //for(int z=threadIdx.z+halo-radius; z<=threadIdx.z+halo+radius; z++)
    //{
    //    for(int y=threadIdx.y+halo-radius; y<=threadIdx.y+halo+radius; y++)
    //    {
    //        for(int x=threadIdx.x+halo-radius; x<=threadIdx.x+halo+radius; x++)
    //        {
    //            val = sharedMemSrc[at(x, y, z, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)];
    //        }
    //    }
    //}
    int32_t low_index = _3d_mem_idx(threadIdx.x + halo - radius, threadIdx.y + halo - radius, threadIdx.z + halo - radius);
    int32_t high_index = _3d_mem_idx(threadIdx.x + halo + radius, threadIdx.y + halo + radius, threadIdx.z + halo + radius);
    
    //thrust::sort(thrust::device, sharedMemSrc + low_index, sharedMemSrc + high_index, thrust::greater<float>());

    for (int32_t i = 0; i <= 100000000000; i++) {
        if (deviceSrc[i] != 0)
            printf("%f\n",deviceSrc[i]);
    }
    if (sharedMemSize % 2) 
        result = sharedMemSrc[_3d_mem_idx(threadIdx.x + halo, threadIdx.y + halo, threadIdx.z + halo) / 2];
    else {
        int mid = _3d_mem_idx(threadIdx.x + halo, threadIdx.y + halo, threadIdx.z + halo);
        result = (sharedMemSrc[mid] + sharedMemSrc[mid - 1]) / 2.0f;
    }
    //printf("%d\n", result);
    // Single pass writing here
    index_3d       =  make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                                blockIdx.y * blockDim.y + threadIdx.y,
                                blockIdx.z * blockDim.z + threadIdx.z);
    index_1d       =  index_3d.z * dim_y * dim_x +
                      index_3d.y * dim_x +
                      index_3d.x;

    if (index_3d.z < dim_z &&
        index_3d.y < dim_y &&
        index_3d.x < dim_x)
    {
        deviceDst[index_1d] = result;
    }
} 
