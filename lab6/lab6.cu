// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan_tree(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2*BLOCK_SIZE];
  // each thread load 2 inputs
  int i = 2*(blockIdx.x*blockDim.x+threadIdx.x);
  int j = i+1;
  if( i < len ) T[2*threadIdx.x] = input[i];
  if( j < len ) T[2*threadIdx.x+1] = input[j];
  // first pass
  int stride = 1;
  while( stride < 2*BLOCK_SIZE ){
    __syncthreads();
    int idx = (threadIdx.x+1)*stride*2 - 1;
    if(idx < 2*BLOCK_SIZE && (idx-stride) >= 0)
    T[idx] += T[idx-stride];
    stride = stride*2;
  }
  // second pass: post scan
  stride = BLOCK_SIZE/2;
  while(stride > 0) {
    __syncthreads();
    int idx = (threadIdx.x+1)*stride*2 - 1;
    if ((idx+stride) < 2*BLOCK_SIZE)
    T[idx+stride] += T[idx];
    stride = stride / 2;
  }
  // copy back
  __syncthreads();
  if( i < len ) output[i] = T[2*threadIdx.x];
  if( j < len ) output[j] = T[2*threadIdx.x+1];
}

__global__ void scan_getsum(float *input, float *sums, int numblocks) {
  // 511, 1023, ..., must be odd
  if( (threadIdx.x == BLOCK_SIZE-1) && (blockIdx.x < numblocks) ){
    sums[blockIdx.x] = input[2*(threadIdx.x+blockIdx.x*blockDim.x)+1];
  }
}

__global__ void scan_add(float *input, float *sums, int len) {
  int sum;
  if( blockIdx.x!=0 ){
    sum = sums[blockIdx.x-1];
    int i = 2*(blockIdx.x*blockDim.x+threadIdx.x);
    int j = i+1;
    if( i < len ) input[i] = input[i] + sum;
    if( j < len ) input[j] = input[j] + sum;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));


  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));


  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  int numblocks = (numElements-1)/(BLOCK_SIZE*2)+1;
  dim3 dimgrid(numblocks,1,1);
  dim3 dimblock(BLOCK_SIZE,1,1);

  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  float* tempsums;
  wbCheck(cudaMalloc((void **)&tempsums,sizeof(float)*numblocks));
  scan_tree<<<dimgrid,dimblock>>>(deviceInput,deviceOutput,numElements);
  scan_getsum<<<dimgrid,dimblock>>>(deviceOutput,tempsums,numblocks);
  scan_tree<<<1,numblocks>>>(tempsums,tempsums,numblocks);
  scan_add<<<dimgrid,dimblock>>>(deviceOutput,tempsums,numElements);

  cudaDeviceSynchronize();

  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));


  //@@  Free GPU Memory
  cudaFree(tempsums);
  cudaFree(deviceInput);
  cudaFree(deviceOutput);


  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

