#include "em_pre_cuda_kernel.h"

at::Tensor cuda_median_3d(at::Tensor &input, at::Tensor &filter, int halo, cudaStream_t stream)
{
    auto dim_x = input.size(2);
    auto dim_y = input.size(1);
    auto dim_z = input.size(0);
    auto radius_z = filter.size(2);
    auto radius_y = filter.size(1);
    auto radius_x = filter.size(0);
    auto output = at::zeros_like(input);
    dim3 blockDim(8, 8, 8);
    dim3 gridDim(
            (dim_x/blockDim.x + ((dim_x%blockDim.x)?1:0)),
            (dim_y/blockDim.y + ((dim_y%blockDim.y)?1:0)),
            (dim_z/blockDim.z + ((dim_z%blockDim.z)?1:0)) );
    size_t sharedMemSize  = (blockDim.x+2*halo)*(blockDim.y+2*halo)*(blockDim.z+2*halo)*sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "__median_3d", ([&] {
        __median_3d<scalar_t><<<gridDim, blockDim, sharedMemSize, stream>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            dim_x,
            dim_y,
            dim_z,
            radius_z,
            radius_y,
            radius_x,
            halo);
      }));
    return output;
}

inline __device__ __host__ int clamp_mirror(int f, int a, int b)
{
    if(f<a) return (a+(a-f));
    if(f>b) return (b-(f-b));
    return f;
}
#define at(x, y, z, dim_x, dim_y, dim_z) ( clamp_mirror((int)z, 0, dim_z-1)*dim_y*dim_x +       \
                                        clamp_mirror((int)y, 0, dim_y-1)*dim_x +            \
                                        clamp_mirror((int)x, 0, dim_x-1) )

template <typename scalar_t>
__global__ void __median_3d(scalar_t* __restrict__ deviceSrc, 
    scalar_t* __restrict__ deviceDst, 
    int dim_z, int dim_y, int dim_x, int radius_z, int radius_y, int radius_x, int halo)
{
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
    for(int batch=0; batch<numBatches; batch++)
    {
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
        if (shared_index_3d.z < (blockDim.z + 2*halo))
        {
            if(global_index_3d.z >= 0 && global_index_3d.z < dim_z &&
               global_index_3d.y >= 0 && global_index_3d.y < dim_y &&
               global_index_3d.x >= 0 && global_index_3d.x < dim_x)
            {
                sharedMemSrc[at(shared_index_3d.x, shared_index_3d.y, shared_index_3d.z, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)] = deviceSrc[global_index_1d];
            }
            else
            {
                sharedMemSrc[at(shared_index_3d.x, shared_index_3d.y, shared_index_3d.z, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)] = -100.0f;
            }
        }
        __syncthreads();
    }


    // Stencil  processing here
    float result = sharedMemSrc[at(threadIdx.x + halo, threadIdx.y + halo, threadIdx.z + halo, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)];
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
    for(int trial=0; trial<8; trial++)
    {
        count = 0;
        for(int z=threadIdx.z+halo-radius_z; z<=threadIdx.z+halo+radius_z; z++)
        {
            for(int y=threadIdx.y+halo-radius_y; y<=threadIdx.y+halo+radius_y; y++)
            {
                for(int x=threadIdx.x+halo-radius_x; x<=threadIdx.x+halo+radius_x; x++)
                {
                    val = sharedMemSrc[at(x, y, z, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)];
                    if(val>pivot)
                        count++;

                }
            }
        }
        if(count < (2*radius_z+1)*(2*radius_y+1)*(2*radius_x+1)/2)
            maxval = floorf(pivot);
        else
            minval = floorf(pivot)+1;
        pivot = (minval + maxval)/2.0f;
    }

    result = floorf(pivot);

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
