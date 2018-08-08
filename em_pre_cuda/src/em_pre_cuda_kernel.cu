/**********************************************************************************************************************
 * Name: em_pre_cuda_kernel.cpp
 * Author: Matin Raayai Ardakani, Tran Minh Quan
 * Email: raayai.matin@gmail.com
 * Where the CUDA magic happens for the em_pre_cuda Python package.
 * Based on the code from Pytorch's tutorials: https://github.com/pytorch/extension-cpp
 * and Tran Minh Quan's 3d_median filter: 
 * https://github.com/tmquan/hetero/blob/ad3c48d1b49b6f79cb06e69ae4199302efd2ffb3/research/ldav14/segment_threshold/median_3d.cu
 **********************************************************************************************************************/
 #include "em_pre_cuda_kernel.h"
 #include <iostream>
 #define ARRAY_SIZE 5000

//TODO: Make this work.
// at::TensorAccessor getFiltRadsAccessor(const at::Tensor& filtRads) {
//     switch (tensor.type().scalarType()) {
//         case at::ScalarType::Char: return filtRads.accessor<int8_t, 1>();
//         case at::ScalarType::Int: return filtRads.accessor<int32_t, 1>();
//         case at::ScalarType::Long: return filtRads.accessor<int64_t, 1>();
//         case at::ScalarType::Short: return filtRads.accessor<int16_t, 1>();
//         case at::ScalarType::Byte: return filtRads.accessor<uint8_t, 1>();
//         case at::ScalarType::Double: return filtRads.accessor<double, 1>();
//         case at::ScalarType::Float: return filtRads.accessor<float, 1>();
//         default: thtx "The filter Radius Tensor data type not supported.\n";    
//  }

int getCType(const at::Tensor& tensor) {
    switch (tensor.type().scalarType()) {
        case at::ScalarType::Char: return sizeof(int8_t);
        case at::ScalarType::Int: return sizeof(int32_t);
        case at::ScalarType::Long: return sizeof(int64_t);
        case at::ScalarType::Short: return sizeof(int16_t);
        case at::ScalarType::Byte: return sizeof(uint8_t);
        case at::ScalarType::Double: return sizeof(double);
        case at::ScalarType::Float: return sizeof(float);
        default: throw "The Image Stack Tensor data type not supported.\n";   
    }
}

at::Tensor cuda_median_3d(const at::Tensor& imStack, const at::Tensor& filtRads, int32_t halo)
{
    //Input arguments for __median_3d
    at::Tensor imStackOut = at::ones_like(imStack) * 255;
    cudaStream_t stream = 0;
    // at::cuda::getCurrentCUDAStream();
    int32_t dimX = imStack.size(2), dimY = imStack.size(1), dimZ = imStack.size(0);
    std::cout << dimX << " " << dimY << " " << dimZ << "\n";
    auto f_copy = filtRads;
    auto fa = f_copy.accessor<float, 1>();
    int32_t radX = fa[2], radY = fa[1], radZ = fa[0];
    std::cout << radX << " " << radY << " " << radZ << "\n";
    //Dispatch arguments for __median_3d
    dim3 blockDim(8, 8, 8);
    dim3 gridDim(
        (dimX/blockDim.x + ((dimX%blockDim.x)?1:0)),
        (dimY/blockDim.y + ((dimY%blockDim.y)?1:0)),
        (dimZ/blockDim.z + ((dimZ%blockDim.z)?1:0)));
    size_t sharedMemSize  = (blockDim.x+2*radX)*(blockDim.y+2*radY)*(blockDim.z+2*radZ)*getCType(imStack);
    //Kernel Call:
    AT_DISPATCH_FLOATING_TYPES(imStack.type(), "__median_3d", ([&] {
        __median_3d<scalar_t><<<gridDim, blockDim, sharedMemSize, stream>>>(
            imStack.data<scalar_t>(),
            imStackOut.data<scalar_t>(),
            dimX,
            dimY,
            dimZ,
            radX,
            radY,
            radZ,
            halo);
      }));
    return imStackOut;
}

inline __device__ __host__ int clamp_mirror(int f, int a, int b)
{
    if(f<a) return (a+(a-f));
    if(f>b) return (b-(f-b));
    return f;
}
// #define at(x, y, z, dimx, dimy, dimz) ( clamp_mirror((int)z, 0, dimz-1)*dimy*dimx +       \
//                                         clamp_mirror((int)y, 0, dimy-1)*dimx +            \
//                                         clamp_mirror((int)x, 0, dimx-1) )

template<typename scalar_t>
__global__
void __median_3d(scalar_t* __restrict__ imStackIn, 
    scalar_t* __restrict__ imStackOut,
    int32_t dimX,
    int32_t dimY,
    int32_t dimZ,
    int32_t radX,
    int32_t radY,
    int32_t radZ,
    int32_t halo) 
    {
//     for (int i = 1024 * 1024; i <= 1024 * 1024 + (25) * (4); i++)
//     {
//         printf("%f\n", imStackIn[i]);
//     }
// }
    auto get_idx = [&] (int x, int y, int z) {
        return clamp_mirror(z, 0, dimZ-1)*dimY*dimX + clamp_mirror(y, 0, dimY-1)*dimX + clamp_mirror(x, 0, dimX-1);
    };
    //Set tx and colum for thread.
    const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int sht_idx = blockIdx.z * blockDim.z + threadIdx.z;
	float filterVector[200] = {0.};   //Take fiter window
    int count = 0;
    for (int z = -radZ; z <= radZ; z++)
    for (int y = -radY; y <= radY; y++)
    for (int x = -radX; x <= radX; x++)
        filterVector[count++] = imStackIn[get_idx(x + col_idx, y + row_idx, z + sht_idx)];
        
         // setup the filterign window.
    // if (col_idx == row_idx == sht_idx == 0)
    //     printf("%d\n", count);
	for (int i = 0; i < count; i++) {
		for (int j = i + 1; j < count; j++) {
			if (filterVector[i] > filterVector[j]) { 
				//Swap the variables.
				float tmp = filterVector[i];
				filterVector[i] = filterVector[j];
				filterVector[j] = tmp;
			}
		}
    }
    // if(col_idx == row_idx == sht_idx == 0) {
    //     for (int i = 0; i < count; i++)
    //         printf("%f\n", filterVector[i]);
    // }
    
    // if (col_idx == row_idx == sht_idx == 6)
    //     printf(filterVector[0]);
    imStackOut[get_idx(col_idx, row_idx, sht_idx)] = filterVector[count/2];   //Set the output variables.
}
    // extern __shared__ float sharedMemSrc[];
    // int  shared_index_1d, global_index_1d, index_1d;
    // int3 shared_index_3d, global_index_3d, index_3d;
    // // Multi batch reading here
    // int3 sharedMemDim    = make_int3(blockDim.x+2*radX,
    //                                  blockDim.y+2*radY,
    //                                  blockDim.z+2*radZ);
    // int  sharedMemSize   = sharedMemDim.x*sharedMemDim.y*sharedMemDim.z;
    // int3 blockSizeDim    = make_int3(blockDim.x+0*halo,
    //                                  blockDim.y+0*halo,
    //                                  blockDim.z+0*halo);
    // int  blockSize        = blockSizeDim.x*blockSizeDim.y*blockSizeDim.z;
    // int  numBatches       = sharedMemSize/blockSize + ((sharedMemSize%blockSize)?1:0);
    // for(int batch=0; batch<numBatches; batch++)
    // {
    //     shared_index_1d  =  threadIdx.z * blockDim.y * blockDim.x +
    //                         threadIdx.y * blockDim.x +
    //                         threadIdx.x +
    //                         blockSize*batch; //Magic is here quantm@unist.ac.kr
    //     shared_index_3d  =  make_int3((shared_index_1d % ((blockDim.y+2*radY)*(blockDim.x+2*radX))) % (blockDim.x+2*radX),
    //                                   (shared_index_1d % ((blockDim.y+2*radY)*(blockDim.x+2*radX))) / (blockDim.x+2*radX),
    //                                   (shared_index_1d / ((blockDim.y+2*radY)*(blockDim.x+2*radX))) );
    //     global_index_3d  =  make_int3(clamp_mirror(blockIdx.x * blockDim.x + shared_index_3d.x - halo, 0, dimX-1),
    //                                   clamp_mirror(blockIdx.y * blockDim.y + shared_index_3d.y - halo, 0, dimY-1),
    //                                   clamp_mirror(blockIdx.z * blockDim.z + shared_index_3d.z - halo, 0, dimZ-1) );

    //     global_index_1d  =  global_index_3d.z * dimY * dimX +
    //                         global_index_3d.y * dimY +
    //                         global_index_3d.x;
    //     if (shared_index_3d.z < (blockDim.z + 2*radZ))
    //     {
    //         if(global_index_3d.z >= 0 && global_index_3d.z < dimZ &&
    //            global_index_3d.y >= 0 && global_index_3d.y < dimY &&
    //            global_index_3d.x >= 0 && global_index_3d.x < dimX)
    //         {
    //             sharedMemSrc[at(shared_index_3d.x, shared_index_3d.y, shared_index_3d.z, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)] = imStackIn[global_index_1d];
    //         }
    //         else
    //         {
    //             sharedMemSrc[at(shared_index_3d.x, shared_index_3d.y, shared_index_3d.z, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)] = -100.0f;
    //         }
    //     }
    //     __syncthreads();
    // }


    // // Stencil  processing here
    // float result = sharedMemSrc[at(threadIdx.x + halo, threadIdx.y + halo, threadIdx.z + halo, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)];

    // float vals[500] = {255.};
    // int32_t count = 0;
    // int tz = threadIdx.z + halo, ty = threadIdx.y + halo, tx = threadIdx.x + halo;
    // for(int z = tz; z <= tz + 2 * radZ; z++)
    // for(int y = ty; y <= ty + 2 * radY; y++)
    // for(int x = tx; x <= tx + 2 * radX; x++) 
    //     vals[count++] = sharedMemSrc[at(x, y, z, sharedMemDim.x, sharedMemDim.y, sharedMemDim.z)];
        
    // // if(tz == 0 && ty == 0 && tx == 0)
    // //     printf("%d\n", ARRAY_SIZE);

    // //     if(count < (2*radX+1)*(2*radY+1)*(2*radZ+1)/2)
    // //         maxval = floorf(pivot);
    // //     else
    // //         minval = floorf(pivot)+1;
    // //     pivot = (minval + maxval)/2.0f;
    // // }

    // // result = floorf(pivot);
    // for(int32_t i = 0; i <= count; i++) 
    // {
    //     for(int32_t j = i + 1; j <= count; j++)
    //     {
    //         if (vals[i] > vals[j]) {
    //             float tmp = vals[i];
    //             vals[i] = vals[j];
    //             vals[j] = tmp;
    //         }    
    //     }
    // }
    // result = vals[count];
    // // Single pass writing here
    // index_3d       =  make_int3(blockIdx.x * blockDim.x + threadIdx.x,
    //                             blockIdx.y * blockDim.y + threadIdx.y,
    //                             blockIdx.z * blockDim.z + threadIdx.z);
    // index_1d       =  index_3d.z * dimY * dimX +
    //                   index_3d.y * dimX +
    //                   index_3d.x;

    // if (index_3d.z < dimZ &&
    //     index_3d.y < dimY &&
    //     index_3d.x < dimX)
    // {
    //     // if (threadIdx.x < 1 && threadIdx.y < 1)
    //     imStackOut[index_1d] = result;
    // }
// } 
