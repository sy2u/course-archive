// Histogram Equalization

#include <wb.h>

#define TILE_WIDTH 16
#define HISTOGRAM_LENGTH 256

  //@@ insert code here
  //---------------------------------------------------------------------------------
  // CUDA Code:
  //---------------------------------------------------------------------------------
  __global__ void to_char( float* in, unsigned char* out, int width, int height, int num_channel ){
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int cur_channel = threadIdx.z;
    if( row < height && col < width ){
      int i = (row*width+col)*num_channel+cur_channel;
      out[i] = (unsigned char) (255 * in[i]);
    }
  }

  __global__ void to_gray( unsigned char* ucharImage, unsigned char* grayImage, int width, int height, int num_channel ){
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if( row < height && col < width ){
      int idx = row * width + col;
      int r = ucharImage[num_channel*idx];
      int g = ucharImage[num_channel*idx + 1];
      int b = ucharImage[num_channel*idx + 2];
      grayImage[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
    }
  }

  __global__ void get_hist( unsigned char* grayImage, unsigned int* histogram, int width, int height ){
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int image_idx = row * width + col;
    int hist_idx = threadIdx.x + threadIdx.y * blockDim.x;

    __shared__ unsigned int private_hist[HISTOGRAM_LENGTH];

    if( hist_idx < HISTOGRAM_LENGTH ) private_hist[hist_idx] = 0;
    __syncthreads();

    if( row < height && col < width ) atomicAdd(&(private_hist[grayImage[image_idx]]), 1);      
    __syncthreads();

    if( hist_idx < HISTOGRAM_LENGTH ) atomicAdd(&(histogram[hist_idx]), private_hist[hist_idx]);      
  }

  __global__ void do_equal( unsigned char* ucharImage, float* cdf, int width, int height, int num_channel ){
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int cur_channel = threadIdx.z;
    if( row < height && col < width ){
      int i = (row*width+col)*num_channel+cur_channel;
      ucharImage[i] = min(max(255.0*(cdf[ucharImage[i]] - cdf[0])/(1.0 - cdf[0]),0.0), 255.0);
    }
  }

  __global__ void cast( unsigned char* ucharImage, float* out, int width, int height, int num_channel ){
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int cur_channel = threadIdx.z;
    if( row < height && col < width ){
      int i = (row*width+col)*num_channel+cur_channel;
      out[i] = (float) (ucharImage[i]/255.0);
    }
  }

  //---------------------------------------------------------------------------------
  // Imported from Lab6:
  //---------------------------------------------------------------------------------
  // __global__ void scan_tree(unsigned int *input, unsigned int *output, int len, int image_size) {
  //   __shared__ float T[2*TILE_WIDTH];
  //   // each thread load 2 inputs
  //   int i = 2*(blockIdx.x*blockDim.x+threadIdx.x);
  //   int j = i+1;
  //   if( i < len ) T[2*threadIdx.x] = input[i] / image_size;
  //   if( j < len ) T[2*threadIdx.x+1] = input[j] / image_size;
  //   // first pass
  //   int stride = 1;
  //   while( stride < 2*TILE_WIDTH ){
  //     __syncthreads();
  //     int idx = (threadIdx.x+1)*stride*2 - 1;
  //     if(idx < 2*TILE_WIDTH && (idx-stride) >= 0)
  //     T[idx] += T[idx-stride];
  //     stride = stride*2;
  //   }
  //   // second pass: post scan
  //   stride = TILE_WIDTH/2;
  //   while(stride > 0) {
  //     __syncthreads();
  //     int idx = (threadIdx.x+1)*stride*2 - 1;
  //     if ((idx+stride) < 2*TILE_WIDTH)
  //     T[idx+stride] += T[idx];
  //     stride = stride / 2;
  //   }
  //   // copy back
  //   __syncthreads();
  //   if( i < len ) output[i] = T[2*threadIdx.x];
  //   if( j < len ) output[j] = T[2*threadIdx.x+1];
  // }

  // __global__ void scan_getsum(unsigned int *input, unsigned int *sums, int numblocks) {
  //   // 511, 1023, ..., must be odd
  //   if( (threadIdx.x == TILE_WIDTH-1) && (blockIdx.x < numblocks) ){
  //     sums[blockIdx.x] = input[2*(threadIdx.x+blockIdx.x*blockDim.x)+1];
  //   }
  // }

  // __global__ void scan_add(unsigned int *input, unsigned int *sums, int len) {
  //   int sum;
  //   if( blockIdx.x!=0 ){
  //     sum = sums[blockIdx.x-1];
  //     int i = 2*(blockIdx.x*blockDim.x+threadIdx.x);
  //     int j = i+1;
  //     if( i < len ) input[i] = input[i] + sum;
  //     if( j < len ) input[j] = input[j] + sum;
  //   }
  // }

  //---------------------------------------------------------------------------------
  // CPU Code:
  //---------------------------------------------------------------------------------
int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  //---------------------------------------------------------------------------------
  // Pre-processing:
  //---------------------------------------------------------------------------------
  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  // get parameter
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);

  //@@ insert code here
  //---------------------------------------------------------------------------------
  // Prep for CUDA:
  //---------------------------------------------------------------------------------
  float* deviceInput_float;
  unsigned char* deviceInput_unsigned;
  unsigned char* deviceInput_gray;
  unsigned int* device_histogram;
  float* device_cdf;
  float* deviceOutput;

  int all_size = imageWidth * imageHeight * imageChannels;
  int flat_size = imageWidth * imageHeight;

  cudaMalloc((void**)&deviceInput_float, sizeof(float)*all_size);
  cudaMalloc((void**)&deviceInput_unsigned, sizeof(unsigned char)*all_size);
  cudaMalloc((void**)&deviceInput_gray, sizeof(unsigned char)*all_size);
  cudaMalloc((void**)&device_histogram, sizeof(unsigned int)*HISTOGRAM_LENGTH);
  cudaMalloc((void**)&device_cdf, sizeof(float)*HISTOGRAM_LENGTH);
  cudaMalloc((void**)&deviceOutput, sizeof(float)*all_size);

  cudaMemcpy(deviceInput_float, hostInputImageData, sizeof(float)*all_size, cudaMemcpyHostToDevice);

  //---------------------------------------------------------------------------------
  // Run CUDA:
  //---------------------------------------------------------------------------------
  // convert to unsigned char
  dim3 BlockDim_Tochar(TILE_WIDTH, TILE_WIDTH, imageChannels);
  dim3 GridDim_Tochar(ceil(imageWidth/(1.0*TILE_WIDTH)), ceil(imageHeight/(1.0*TILE_WIDTH)), 1);
  to_char<<<GridDim_Tochar, BlockDim_Tochar>>>(deviceInput_float, deviceInput_unsigned, imageWidth, imageHeight ,imageChannels);

  // convert to gray scale
  dim3 BlockDim_Togray(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 GridDim_Togray(ceil(imageWidth/(1.0*TILE_WIDTH)), ceil(imageHeight/(1.0*TILE_WIDTH)), 1);
  to_gray<<<GridDim_Togray, BlockDim_Togray>>>(deviceInput_unsigned, deviceInput_gray, imageWidth, imageHeight ,imageChannels);

  // get histogram
  dim3 BlockDim_Hist(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 GridDim_Hist(ceil(imageWidth/(1.0*TILE_WIDTH)), ceil(imageHeight/(1.0*TILE_WIDTH)), 1);
  get_hist<<<GridDim_Hist,BlockDim_Hist>>>(deviceInput_gray, device_histogram, imageWidth, imageHeight);

  // compute CDF
  unsigned int cpu_hist[HISTOGRAM_LENGTH];
  float cpu_cdf[HISTOGRAM_LENGTH];
  cudaMemcpy(cpu_hist,device_histogram,sizeof(unsigned int)*HISTOGRAM_LENGTH,cudaMemcpyDeviceToHost);
  cpu_cdf[0] = (float)cpu_hist[0]/flat_size;
  for( int i = 1; i < HISTOGRAM_LENGTH; i++ ){
    cpu_cdf[i] = cpu_cdf[i-1] + (float)cpu_hist[i]/flat_size;
  }
  cudaMemcpy(device_cdf,cpu_cdf,sizeof(float)*HISTOGRAM_LENGTH,cudaMemcpyHostToDevice);
  // int numblocks = (HISTOGRAM_LENGTH-1)/(TILE_WIDTH*2)+1;
  // unsigned int* tempsums;
  // cudaMalloc((void **)&tempsums,sizeof(float)*numblocks);
  // dim3 dimgrid(numblocks,1,1);
  // dim3 dimblock(TILE_WIDTH,1,1);
  // scan_tree<<<dimgrid,dimblock>>>(device_histogram,device_cdf,HISTOGRAM_LENGTH,flat_size);
  // scan_getsum<<<dimgrid,dimblock>>>(device_cdf,tempsums,numblocks);
  // scan_tree<<<1,numblocks>>>(tempsums,tempsums,numblocks,flat_size);
  // scan_add<<<dimgrid,dimblock>>>(device_cdf,tempsums,HISTOGRAM_LENGTH);

  // Apply Equalization
  dim3 BlockDim_Equal(TILE_WIDTH, TILE_WIDTH, imageChannels);
  dim3 GridDim_Equal(ceil(imageWidth/(1.0*TILE_WIDTH)), ceil(imageHeight/(1.0*TILE_WIDTH)), 1);
  do_equal<<<GridDim_Equal,BlockDim_Equal>>>(deviceInput_unsigned,device_cdf,imageWidth,imageHeight,imageChannels);

  // Cast back to float
  dim3 BlockDim_Cast(TILE_WIDTH, TILE_WIDTH, imageChannels);
  dim3 GridDim_Cast(ceil(imageWidth/(1.0*TILE_WIDTH)), ceil(imageHeight/(1.0*TILE_WIDTH)), 1);
  cast<<<GridDim_Cast,BlockDim_Cast>>>(deviceInput_unsigned,deviceOutput,imageWidth,imageHeight,imageChannels);

  //---------------------------------------------------------------------------------
  // Post Run
  //---------------------------------------------------------------------------------

  cudaMemcpy(hostOutputImageData, deviceOutput, sizeof(float)*all_size, cudaMemcpyDeviceToHost);

  cudaFree(deviceInput_float);
  cudaFree(deviceInput_unsigned);
  cudaFree(deviceInput_gray);
  cudaFree(device_histogram);
  cudaFree(device_cdf);
  cudaFree(deviceOutput);

  wbSolution(args, outputImage);

  // for( int i = 0; i < HISTOGRAM_LENGTH; i++ ){
  //   wbLog(TRACE, "cpu_cdf[", i, "] = ", cpu_cdf[i]);
  // }

  return 0;
}

