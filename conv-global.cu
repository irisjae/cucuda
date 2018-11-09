// CUDA runtime
#include <cuda_runtime.h>
#include <stdio.h>
// Helper functions and utilities to work with CUDA
// #include <helper_functions.h>

/**********************************************
 * Check whether we read back the same input
 * The double check is just for debug purposes.
 * We can comment it out when benchmarking the time.
 **********************************************/
#define GPU_DEBUG

/**
 *  
 * PARAMETERS 
 *  
 */
#define IN_Y_DIM 720 // input image height
#define IN_X_DIM 1280 // input image width
#define WINDOW_X_DIM 6 // convolution width
#define WINDOW_Y_DIM 6 // convolution height
#define WINDOW_X_STRIDE 2 // convolution x stride
#define WINDOW_Y_STRIDE 2 // convolution y stride
#define OUT_CHANNEL_NUM 6 // number of filters of convolution

//TODO: warn on stride wider than dim

/**
 * 
 * CUDA UTILS
 * 
 */
#define cuda_try( ans ) { __cuda_try((ans), __FILE__, __LINE__); }
inline void __cuda_try( cudaError_t code, const char * file, int line, bool abort=true ) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/**
 * 
 * UTILS
 * 
 */
#define OUT_Y_DIM ((IN_Y_DIM - WINDOW_Y_DIM) / WINDOW_Y_STRIDE + 1) // output image height
#define OUT_X_DIM ((IN_X_DIM - WINDOW_X_DIM) / WINDOW_X_STRIDE + 1) // output image width
#define WINDOW_SIZE (WINDOW_X_DIM * WINDOW_Y_DIM) // total convolution size
#define IN_IMG_SIZE (IN_Y_DIM * IN_X_DIM) // total input size per image
#define IN_SIZE IN_IMG_SIZE // total input size
#define OUT_IMG_SIZE (OUT_Y_DIM * OUT_X_DIM) // total output size per image
#define OUT_SIZE (OUT_IMG_SIZE * OUT_CHANNEL_NUM) // total output size



/******************************************
 * Device function declaration
 *****************************************/
__global__ void layer1_init_bias(float* d_y, float* d_bias);
__global__ void layer1_feature_maps(float* d_y, unsigned char* d_in_layer, float* d_weight);
__global__ void layer1_sigmoid(float* d_y, unsigned char* d_out_layer);

/************************************************************************************
 * Input   : input image, pointer to output result, coefficients bias and weights
 * Output  : neuron outputs of the feature maps represented as an image
 * Procedure: perform feed forward computation through the feature extraction layers
     *******************************************************************************/
void cuda_convolution_layer1(unsigned char in_layer[], unsigned char out_layer[],
			     const float bias[], const float weight[]) {
  /*********************************
   * allocate device memory on GPU
   *********************************/

  unsigned int size_y = OUT_CHANNEL_NUM*OUT_Y_DIM*OUT_X_DIM;
  unsigned int mem_size_y = sizeof(float) * size_y;
  float *d_y;

  unsigned int size_bias = OUT_CHANNEL_NUM;
  unsigned int mem_size_bias = sizeof(float) * size_bias;
  float *d_bias;

  unsigned int size_weight = OUT_CHANNEL_NUM*WINDOW_SIZE;
  unsigned int mem_size_weight = sizeof(float) * size_weight;
  float *d_weight;

  unsigned int size_in_layer = IN_Y_DIM*IN_X_DIM;
  unsigned int mem_size_in_layer = sizeof(unsigned char) * size_in_layer;
  unsigned char *d_in_layer;

  unsigned int size_out_layer = OUT_CHANNEL_NUM*OUT_Y_DIM*OUT_X_DIM;
  unsigned int mem_size_out_layer = sizeof(unsigned char) * size_out_layer;
  unsigned char *d_out_layer;


  /********************************
   * Allocate device memory on GPU.
   * Check the first cudaMalloc error,
   * in case GPU is busy.
   ********************************/
   cuda_try(cudaMalloc((void **) &d_y, mem_size_y));
   cuda_try(cudaMalloc((void **) &d_in_layer, mem_size_in_layer));
   cuda_try(cudaMalloc((void **) &d_bias, mem_size_bias));
   cuda_try(cudaMalloc((void **) &d_weight, mem_size_weight));
   cuda_try(cudaMalloc((void **) &d_out_layer, mem_size_out_layer));

  /*********************************************
   * copy data from host (CPU) to device (GPU)
   ********************************************/
   cuda_try(cudaMemcpy(d_in_layer, in_layer, mem_size_in_layer, cudaMemcpyHostToDevice));
   cuda_try(cudaMemcpy(d_bias, bias, mem_size_bias, cudaMemcpyHostToDevice));
   cuda_try(cudaMemcpy(d_weight, weight, mem_size_weight, cudaMemcpyHostToDevice));

  /* Synchronize all the cudaMemcpy API before doing the actual computation */
  cuda_try(cudaDeviceSynchronize());

  /*********************************************
   * Layer 1, Step 1: 
   * init values of feature maps at bias value 
   ********************************************/
  /* (16, 16, z) (choose your z dimension) threads per block */
  /* NOTE: threads per block limit is 1024 for K80 */
  /* NOTE: if you use another GPU, check the deviceQuery */

  dim3 bias_dim(16, 16, 1024 / (16 * 16));
  layer1_init_bias<<<1024, bias_dim>>>(d_y, d_bias);

  /* Just in case, put a sync here */
  cuda_try(cudaDeviceSynchronize());

  /*********************************************
   * Layer 1, Step 2: 
   * loop over output feature maps
   ********************************************/
  /* (8, 8, z) (choose your z dimension) threads per block */
  /***********************************************
   * The layer size is not diviadable by 8 either.
   * Mask out extra threads in the kernel.
   **********************************************/  
  
  dim3 feature_maps_dim(8, 8, 1024 / (8 * 8));
  layer1_feature_maps<<<1024, feature_maps_dim>>>(d_y, d_in_layer, d_weight);

  /* Just in case, put a sync here */
  cuda_try(cudaDeviceSynchronize());

  /********************************************
   (14, 14, z) (choose your z dimension) threads per block
   ********************************************
   * Layer 1, Step 3: 
   * sigmoid activation function
   ********************************************/
  
  dim3 sigmoid_dim(14, 14, 1024 / (14 * 14));
  layer1_sigmoid<<<1024, sigmoid_dim>>>(d_y, d_out_layer);

  /* Just in case, put a sync here */
  cuda_try(cudaDeviceSynchronize());

  /* Read back the output from device (GPU) to host (CPU) */
   cuda_try(cudaMemcpy(out_layer, d_out_layer, mem_size_out_layer, cudaMemcpyDeviceToHost));


  /* Just in case, put a sync here */
  cuda_try(cudaDeviceSynchronize());

  /* release device memory */
  cuda_try(cudaFree(d_y));
  cuda_try(cudaFree(d_in_layer));
  cuda_try(cudaFree(d_bias));
  cuda_try(cudaFree(d_weight));
  cuda_try(cudaFree(d_out_layer));

}


/*********************************************
 * GPU kernel
 * Layer 1, Step 1: 
 * init values of feature maps at bias value 
 ********************************************/
__global__ void layer1_init_bias(float* d_y, float* d_bias) {
	int total_work_size = OUT_SIZE;
	int total_workers = (gridDim.x * gridDim.y * gridDim.z) * (blockDim.x * blockDim.y * blockDim.z);
	int worker_id = ((((((blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.z + threadIdx.z) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);

	for (int n = worker_id; n < total_work_size; n += total_workers) {
		int z = (n / OUT_IMG_SIZE);

		d_y[n] = d_bias[z];
	}
}

/*********************************************
 * GPU kernel
 * Layer 1, Step 2: 
 * loop over output feature maps
 ********************************************/
__global__ void layer1_feature_maps(float* d_y, unsigned char* d_in_layer, float* d_weight) {
	int total_work_size = OUT_SIZE;
	int total_workers = (gridDim.x * gridDim.y * gridDim.z) * (blockDim.x * blockDim.y * blockDim.z);
	int worker_id = ((((((blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.z + threadIdx.z) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);

	for (int n = worker_id; n < total_work_size; n += total_workers) {
		int x = (n % OUT_IMG_SIZE) % OUT_X_DIM;
		int y = (n % OUT_IMG_SIZE) / OUT_X_DIM;
		int z = (n / OUT_IMG_SIZE);

		float convolution = 0;
		for (int i = 0; i < WINDOW_X_DIM; i ++) {
			for (int j = 0; j < WINDOW_Y_DIM; j ++) {
				convolution += d_in_layer[(y * WINDOW_Y_STRIDE + j) * IN_X_DIM + (x * WINDOW_X_STRIDE + i)] * d_weight[((z) * WINDOW_Y_DIM + j) * WINDOW_X_DIM + i];
			}
		}
		d_y[n] += convolution;
	}
}

/*********************************************
 * GPU kernel
 * Layer 1, Step 3: 
 * sigmoid activation function
 ********************************************/
__global__ void layer1_sigmoid(float* d_y, unsigned char* d_out_layer){
	int total_work_size = OUT_SIZE;
	int total_workers = (gridDim.x * gridDim.y * gridDim.z) * (blockDim.x * blockDim.y * blockDim.z);
	int worker_id = ((((((blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.z + threadIdx.z) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);

	for (int n = worker_id; n < total_work_size; n += total_workers) {
		d_out_layer[n] = (unsigned char) (255.999f / (1 + expf(- d_y[n] / 256)));
	}
}
