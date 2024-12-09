#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256
#define STREAM_NUM 4

const float *host_input_temp;
const float *host_output_temp;

__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    /*
    Modify this function to implement the input matrix unrolling kernel.

    Function paramter definitions:
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    
    // TODO: Insert your input matrix unrolling kernel code here
    // int Height_unrolled = K * K * Channel;
    int Width_unrolled = Batch * Height_out * Width_out;

    #define out_3d(i2, i1, i0) output[(i2) * (Height_out * Width_out) + (i1) * (Width_unrolled) + i0]

    // int H_grid = (Height_out + TILE_WIDTH - 1) / TILE_WIDTH;
    size_t W_grid = (Width_out + TILE_WIDTH - 1) / TILE_WIDTH;

    int b = blockIdx.z;
    int c = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;

    if( h < Height_out && w < Width_out ){
        int w_base = c * (K*K);
        for( int p = 0; p < K; p++ ){
            for( int q = 0; q < K; q++ ){
                size_t h_unroll = w_base + p * K + q;
                size_t w_unroll = h * Width_out + w;
                out_3d(b, h_unroll, w_unroll) = in_4d(b ,c, h+p, w+q);
            }
        }
    } 

    #undef in_4d
    #undef out_3d
}

// Tiled matrix multiplication kernel. Computes C = AB
// You don't need to modify this kernel.
__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
    float val = 0;

    for (int tileId = 0; tileId < (numAColumns - 1) / TILE_WIDTH + 1; tileId++) {
        if (row < numARows && tileId * TILE_WIDTH + tx < numAColumns) {
            tileA[ty][tx] = A[(size_t) row * numAColumns + tileId * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = 0;
        }
        if (col < numBColumns && tileId * TILE_WIDTH + ty < numBRows) {
            tileB[ty][tx] = B[((size_t) tileId * TILE_WIDTH + ty) * numBColumns + col];
        } else {
            tileB[ty][tx] = 0;
        }
        __syncthreads();

        if (row < numCRows && col < numCColumns) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val += tileA[ty][i] * tileB[i][tx];
            }
        }
        __syncthreads();
    }

    if (row < numCRows && col < numCColumns) {
        C[row * numCColumns + col] = val;
    }
}

// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU
    // We pass double pointers for you to initialize the relevant device pointers,
    // which are passed to the other two functions.
    cudaMalloc(device_input_ptr, (size_t) sizeof(float)*Batch*Channel*Height*Width);
    cudaMalloc(device_mask_ptr, (size_t) sizeof(float)*K*K*Channel*Map_out);
    cudaMalloc(device_output_ptr, (size_t) sizeof(float)*Batch*(Height-K+1)*(Width-K+1)*Map_out);

    // small dataset, no need to copy by stream
    cudaMemcpy(*device_mask_ptr, host_mask,  (size_t) sizeof(float)*K*K*Channel*Map_out, cudaMemcpyHostToDevice);

    // prepare for stream
    host_input_temp = host_input;
    host_output_temp = host_output;

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
    const size_t Height_unrolled = Channel * K * K;
    
    float *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
    float *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication
    cudaMalloc((void**)&unrolled_matrix, (size_t) Batch * Channel * K * K * Height_out * Width_out * sizeof(float));
    cudaMalloc((void**)&matmul_output, (size_t) (Batch * Map_out * Height_out * Width_out) * sizeof(float));

    ///////////////////////////////
    // [PM3] Req_0: Stream Start //
    ///////////////////////////////

    cudaStream_t stream [STREAM_NUM]; 

    for (int i = 0; i < STREAM_NUM; i++){

        cudaError_t err = cudaStreamCreate(&stream[i]);
        if (err != cudaSuccess) {
            printf("Stream creation failed for stream[%d]: %s\n", i, cudaGetErrorString(err));
            return;
        }

        size_t  batch_size = (Batch+STREAM_NUM-1) / STREAM_NUM;;
        size_t  offset_input = (size_t) Channel*Height*Width* batch_size;
        size_t  offset_unrolled = (size_t) Channel*K*K*Height_out*Width_out* batch_size;
        size_t  offset_output = (size_t) Height_out*Width_out*Map_out* batch_size;

        size_t  curr_batch = (i==STREAM_NUM-1) ? Batch-(STREAM_NUM-1)*batch_size : batch_size;
        size_t  Width_unrolled = curr_batch * Height_out * Width_out;

        // Copy: Host -> Device
        size_t  size_input = (size_t) Channel*Height*Width* curr_batch;
        cudaMemcpyAsync((void*)(device_input+offset_input*i), (void*)(host_input_temp+offset_input*i), sizeof(float)*size_input, cudaMemcpyHostToDevice, stream[i]);

        // Kernel Call: Unroll
        dim3 Grid_unroll(Channel, (size_t) (Height_out+TILE_WIDTH-1)/TILE_WIDTH * (Width_out+TILE_WIDTH-1)/TILE_WIDTH, curr_batch);
        dim3 Block_unroll(TILE_WIDTH, TILE_WIDTH, 1);
        matrix_unrolling_kernel<<<Grid_unroll, Block_unroll, 0, stream[i]>>>
            (device_input+offset_input*i, unrolled_matrix+offset_unrolled*i, curr_batch, Channel, Height, Width, K);

        // Kernel Call: Matmul
        dim3 Grid_matmul((size_t)(Width_unrolled+TILE_WIDTH-1)/TILE_WIDTH, (size_t)(Map_out+TILE_WIDTH-1)/TILE_WIDTH, 1);
        dim3 Block_matmul(TILE_WIDTH, TILE_WIDTH, 1);
        matrixMultiplyShared<<<Grid_matmul,Block_matmul, 0, stream[i]>>>
            (device_mask, unrolled_matrix+offset_unrolled*i, matmul_output+offset_output*i, Map_out, Height_unrolled, Height_unrolled, Width_unrolled, Map_out, Width_unrolled);

        // Kernel Call: Permute
        const int out_image_size = Height_out * Width_out;
        dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, curr_batch, 1);
        matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE, 0, stream[i]>>>
            (matmul_output+offset_output*i, device_output+offset_output*i, Map_out, curr_batch, out_image_size);

        // Copy: Device -> Host
        size_t  size_output = (size_t) Height_out*Width_out*Map_out* curr_batch;
        cudaMemcpyAsync((void*)(host_output_temp+offset_output*i), (void*)(device_output+offset_output*i), sizeof(float)*size_output, cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < STREAM_NUM; i++) {
        cudaStreamDestroy(stream[i]);
    }

    /////////////////////////////
    // [PM3] Req_0: Stream End //
    /////////////////////////////

    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);
    
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Already done in conv_forward_gpu for Stream
    
    // const int Height_out = Height - K + 1;
    // const int Width_out = Width - K + 1;
    // cudaMemcpy(host_output, device_output, (size_t) sizeof(float)*Height_out*Width_out*Map_out*Batch, cudaMemcpyDeviceToHost);

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