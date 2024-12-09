// Req_2: Kernel Fusion

#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

__global__ void kernel_fusion(const float *input, float *output, const float * mask,
                            const int Batch, const int Channel, const int Map_out,
                            const int Height, const int Width,
                            const int K) {

    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH]; // mask
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH]; // unrolled input feature
    
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // Geometric Mapping
    // Kernel is launched from the view of output computation
    int row_out = blockIdx.y * blockDim.y + threadIdx.y;
    int col_out = blockIdx.x * blockDim.x + threadIdx.x;

    // Batch x Channel x Height x Width
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    // Batch x Map_out x Height_out x Width_out
    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]

    /////////////////////
    //   [M3] Fusion   //
    /////////////////////

    // Kernel Mapping
    size_t height = col_out / Width_out;
    size_t width = col_out % Width_out;
    size_t map = row_out;
    size_t batch = blockIdx.z;

    size_t Kernel_size = K * K;
    size_t Height_unrolled = Channel * Kernel_size;
    size_t Width_unrolled = Height_out * Width_out;

    // Modified from MatrixMultiplyShared
    int ty = threadIdx.y, tx = threadIdx.x;

    float val = 0;

    for (int tileId = 0; tileId < (Height_unrolled - 1) / TILE_WIDTH + 1; tileId++) {
        // load mask
        if (row_out < Map_out && tileId * TILE_WIDTH + tx < Height_unrolled) {
            tileA[ty][tx] = mask[(size_t) row_out * Height_unrolled + tileId * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = 0;
        }
        // load feature map
        size_t row_in = (size_t) tileId * TILE_WIDTH + ty;
        if (col_out < Width_unrolled && row_in < Height_unrolled) {
            //tileB[ty][tx] = B[((size_t) tileId * TILE_WIDTH + ty) * numBColumns + col]; => unrolled(i*tile+ty, col)
            size_t channel = row_in / Kernel_size;
            int p = (row_in % Kernel_size) / K;
            int q = (row_in % Kernel_size) % K;
            // Batch x Channel x Height x Width
            // X_unroll[b, h_unroll, w_unroll] = X[b, c, h + p, w + q];
            tileB[ty][tx] = in_4d(batch, channel, height+p, width+q);
        } else {
            tileB[ty][tx] = 0;
        }
        __syncthreads();

        if (row_out < Map_out && col_out < Width_unrolled) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val += tileA[ty][i] * tileB[i][tx];
            }
        }
        __syncthreads();
    }

    if (row_out < Map_out && col_out < Width_out * Height_out) {
        // C[row * numCColumns + col] = val;
        // Batch x Map_out x Height_out x Width_out
        // already permuted
        out_4d(batch, map, height, width) = val;
    }

    #undef in_4d
    #undef out_3d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU
    // We pass double pointers for you to initialize the relevant device pointers,
    // which are passed to the other two functions.
    cudaMalloc(device_input_ptr, (size_t) sizeof(float)*Batch*Channel*Height*Width);
    cudaMalloc(device_mask_ptr, (size_t) sizeof(float)*K*K*Channel*Map_out);
    cudaMalloc(device_output_ptr, (size_t) sizeof(float)*Batch*(Height-K+1)*(Width-K+1)*Map_out);

    cudaMemcpy(*device_input_ptr, host_input, (size_t) sizeof(float)*Batch*Channel*Height*Width, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask,  (size_t) sizeof(float)*K*K*Channel*Map_out, cudaMemcpyHostToDevice);

    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // Do Kernel Call
    dim3 Grid_fusion((size_t) (Height_out*Width_out+TILE_WIDTH-1)/TILE_WIDTH, (size_t)(Map_out+TILE_WIDTH-1)/TILE_WIDTH, Batch);
    dim3 Block_fusion(TILE_WIDTH, TILE_WIDTH, 1);
    kernel_fusion<<<Grid_fusion, Block_fusion>>>(device_input, device_output, device_mask, Batch, Channel, Map_out, Height, Width, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMemcpy(host_output, device_output, (size_t) sizeof(float)*Height_out*Width_out*Map_out*Batch, cudaMemcpyDeviceToHost);

    // TODO: Free device memory
    cudaFree(device_input);
    cudaFree(device_mask);
    cudaFree(device_output);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}