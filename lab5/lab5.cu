// MP5 Reduction
// Input: A num list of length n
// Output: Sum of the list = list[0] + list[1] + ... + list[n-1];

#include <wb.h>

#define BLOCK_SIZE 512 //@@ This value is not fixed and you can adjust it according to the situation

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)
  
__global__ void total(float *input, float *output, int len) {
  //@@ Load a segment of the input vector into shared memory
  __shared__ float partialSum[2*BLOCK_SIZE];
  unsigned int t = threadIdx.x;
  unsigned int idx1 = t + 2*blockDim.x*blockIdx.x;
  unsigned int idx2 = idx1 + blockDim.x;
  partialSum[t] = input[idx1];
  if( idx2 < len ){
    partialSum[blockDim.x+t] = input[idx2];
  } else {
    partialSum[blockDim.x+t] = 0;
  }
  __syncthreads();
  //@@ Traverse the reduction tree
  for( unsigned int stride = blockDim.x; stride >= 1; stride >>= 1 ){
    __syncthreads();
    if( t < stride ){
      partialSum[t] += partialSum[stride+t];
    }
  }
  //@@ Write the computed sum of the block to the output vector at the correct index
  if( t == 0 ){ output[blockIdx.x] = partialSum[0]; }
}

int main(int argc, char **argv) {
  int ii;
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  //@@ Initialize device input and output pointers
  float *deviceInput;
  float *deviceOutput;

  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list

  args = wbArg_read(argc, argv);

  //Import data and create memory on host
  hostInput =
      (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = numInputElements / (BLOCK_SIZE << 1);
  if (numInputElements % (BLOCK_SIZE << 1)) {
    numOutputElements++;
  }
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  // The number of input elements in the input is numInputElements
  // The number of output elements in the input is numOutputElements
  // wbLog(TRACE, "input length: ",numInputElements);
  // wbLog(TRACE, "computed length: ",ceil(numInputElements/BLOCK_SIZE));
  // wbLog(TRACE, "output length: ",numOutputElements);

  //@@ Allocate GPU memory
  cudaMalloc((void**)&deviceInput, sizeof(float)*numInputElements);
  cudaMalloc((void**)&deviceOutput, sizeof(float)*numOutputElements);

  //@@ Copy input memory to the GPU
  cudaMemcpy(deviceInput, hostInput, sizeof(float)*numInputElements, cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 BlockDim (BLOCK_SIZE,1,1);
  dim3 GridDim (numOutputElements,1,1);

  //@@ Launch the GPU Kernel and perform CUDA computation
  total<<<GridDim,BlockDim>>>(deviceInput,deviceOutput,numInputElements);
  
  cudaDeviceSynchronize();  
  //@@ Copy the GPU output memory back to the CPU
  cudaMemcpy(hostOutput,deviceOutput,sizeof(float)*numOutputElements,cudaMemcpyDeviceToHost);
  
  /********************************************************************
   * Reduce output vector on the host
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input. 
   * For simplicity, we do not require that for this lab.
   ********************************************************************/
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }

  // debug
  // wbLog(TRACE, hostOutput[0]);

  //@@ Free the GPU memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);


  wbSolution(args, hostOutput, 1);

  free(hostInput);
  free(hostOutput);

  return 0;
}

