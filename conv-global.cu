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

#define SHARED_X_DIM 256
#define SHARED_Y_DIM 256

/*
  Define all constant variavle below with a REASONABLE name
*/

#define in_y_dim 720
#define in_x_dim 1280
#define window_x_dim 6
#define window_y_dim 6
#define window_x_stride 2
#define window_y_stride 2
#define out_channel_num 6

//do not support stride wider than dim yet
//TODO: check edge cases

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
#define SHARED_DIM (SHARED_X_DIM * SHARED_Y_DIM)
#define out_y_dim ((in_y_dim - window_y_dim) / window_y_stride + 1)
#define out_x_dim ((in_x_dim - window_x_dim) / window_x_stride + 1)
#define window_size (window_x_dim * window_y_dim)
#define in_img_size (in_y_dim * in_x_dim)
#define in_size in_img_size
#define out_img_size (out_y_dim * out_x_dim)
#define out_size (out_img_size * out_channel_num)



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

  unsigned int size_y = out_channel_num*out_y_dim*out_x_dim;
  unsigned int mem_size_y = sizeof(float) * size_y;
  float *d_y;

  unsigned int size_bias = out_channel_num;
  unsigned int mem_size_bias = sizeof(float) * size_bias;
  float *d_bias;

  unsigned int size_weight = out_channel_num*window_size;
  unsigned int mem_size_weight = sizeof(float) * size_weight;
  float *d_weight;

  unsigned int size_in_layer = in_y_dim*in_x_dim;
  unsigned int mem_size_in_layer = sizeof(unsigned char) * size_in_layer;
  unsigned char *d_in_layer;

  unsigned int size_out_layer = out_channel_num*out_y_dim*out_x_dim;
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
	int total_work_size = out_size;
	int total_workers = (gridDim.x * gridDim.y * gridDim.z) * (blockDim.x * blockDim.y * blockDim.z);
	int worker_id = ((((((blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.z + threadIdx.z) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);

	for (int n = worker_id; n < total_work_size; n += total_workers) {
		int z = (n / out_img_size);

		d_y[n] = d_bias[z];
	}
}

/*********************************************
 * GPU kernel
 * Layer 1, Step 2: 
 * loop over output feature maps
 ********************************************/
__global__ void layer1_feature_maps(float* d_y, unsigned char* d_in_layer, float* d_weight) {
	int total_work_size = out_size;
	int total_workers = (gridDim.x * gridDim.y * gridDim.z) * (blockDim.x * blockDim.y * blockDim.z);
	int worker_id = ((((((blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.z + threadIdx.z) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);

	for (int n = worker_id; n < total_work_size; n += total_workers) {
		int x = (n % out_img_size) % out_x_dim;
		int y = (n % out_img_size) / out_x_dim;
		int z = (n / out_img_size);

		float convolution = 0;
		for (int i = 0; i < window_x_dim; i ++) {
			for (int j = 0; j < window_y_dim; j ++) {
				convolution += d_in_layer[(y * window_y_stride + j) * in_x_dim + (x * window_x_stride + i)] * d_weight[((z) * window_y_dim + j) * window_x_dim + i];
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
	int total_work_size = out_size;
	int total_workers = (gridDim.x * gridDim.y * gridDim.z) * (blockDim.x * blockDim.y * blockDim.z);
	int worker_id = ((((((blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.z + threadIdx.z) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);

	for (int n = worker_id; n < total_work_size; n += total_workers) {
		d_out_layer[n] = (unsigned char) (255.999f / (1 + expf(- d_y[n] / 256)));
	}
}
