#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float subtileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subtileB[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;
  float cvalue = 0;

  for( int q = 0; q < (numAColumns-1)/TILE_WIDTH+1; q++ ){ // numacol = numbrow

    if (row < numARows && (q*TILE_WIDTH+tx) < numAColumns) {
      subtileA[ty][tx] = A[row*numAColumns + q*TILE_WIDTH+tx];
    } else {
      subtileA[ty][tx] = 0;
    }

    if ( (q*TILE_WIDTH+ty) < numBRows && col < numBColumns ){
      subtileB[ty][tx] = B[(q*TILE_WIDTH+ty)*numBColumns+col];
    } else {
      subtileB[ty][tx] = 0;
    }

    __syncthreads();
    for( int k = 0; k < TILE_WIDTH; k++ ){
      cvalue += subtileA[ty][k] * subtileB[k][tx];
    }
    __syncthreads();
  }

  if( row < numCRows && col < numCColumns ){
    C[row*numCColumns+col] = cvalue;
  } 
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;

  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);

  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;

  //@@ Allocate the hostC matrix
  hostC = (float*)malloc(numCRows*numCColumns*sizeof(float));

  //@@ Allocate GPU memory here
  cudaMalloc((void**)&deviceA, numARows*numAColumns*sizeof(float));
  cudaMalloc((void**)&deviceB, numBRows*numBColumns*sizeof(float));
  cudaMalloc((void**)&deviceC, numCRows*numCColumns*sizeof(float));

  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, numARows*numAColumns*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows*numBColumns*sizeof(float), cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  // num and width are both integers, so can't directly use ceil
  dim3 DimGrid((numCColumns+TILE_WIDTH-1)/TILE_WIDTH,(numCRows+TILE_WIDTH-1)/TILE_WIDTH,1);
  dim3 DimBlock(TILE_WIDTH,TILE_WIDTH,1); // use the default warp size

  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCRows*numCColumns*sizeof(float), cudaMemcpyDeviceToHost);

  // debug
  // for(int i = 0; i < 10; i++){
  //   wbLog(TRACE, "C[", i, "] = ", hostC[numCRows*numCColumns-i]);
  // }

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);

  //@@ Free the hostC matrix
  free(hostC);

  return 0;
}
