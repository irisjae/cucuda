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

#define SHARED_X_DIM 128
#define SHARED_Y_DIM 128

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

#define split( n, among ) ((n + (among - 1)) / among)


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

  cudaError_t error;


  /********************************
   * Allocate device memory on GPU.
   * Check the first cudaMalloc error,
   * in case GPU is busy.
   ********************************/
  error = cudaMalloc((void **) &d_y, mem_size_y);
  /* Check the error code of the first CUDA API call */
  if (error != cudaSuccess){
    printf("cudaMalloc returned error code %d, line(%d)\n", error, __LINE__);
    printf("CUDA error: %s\n", cudaGetErrorString(error));
  }else{
    printf("cudaMalloc success.\n");
  }
  /* if no error for the first cudaMalloc, continue other cudaMalloc */
  error = cudaMalloc((void **) &d_in_layer, mem_size_in_layer);
  error = cudaMalloc((void **) &d_bias, mem_size_bias);
  error = cudaMalloc((void **) &d_weight, mem_size_weight);
  error = cudaMalloc((void **) &d_out_layer, mem_size_out_layer);

  /*********************************************
   * copy data from host (CPU) to device (GPU)
   ********************************************/
  error = cudaMemcpy(d_in_layer, in_layer, mem_size_in_layer, cudaMemcpyHostToDevice);
  error = cudaMemcpy(d_bias, bias, mem_size_bias, cudaMemcpyHostToDevice);
  error = cudaMemcpy(d_weight, weight, mem_size_weight, cudaMemcpyHostToDevice);

  /* Synchronize all the cudaMemcpy API before doing the actual computation */
  cudaDeviceSynchronize();

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
  cudaDeviceSynchronize();

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
  cudaDeviceSynchronize();

  /********************************************
   (14, 14, z) (choose your z dimension) threads per block
   ********************************************
   * Layer 1, Step 3: 
   * sigmoid activation function
   ********************************************/
  
  dim3 sigmoid_dim(14, 14, 1024 / (14 * 14));
  layer1_sigmoid<<<1024, sigmoid_dim>>>(d_y, d_out_layer);

  /* Just in case, put a sync here */
  cudaDeviceSynchronize();

  /* Read back the output from device (GPU) to host (CPU) */
  error = cudaMemcpy(out_layer, d_out_layer, mem_size_out_layer, cudaMemcpyDeviceToHost);


  /* Just in case, put a sync here */
  cudaDeviceSynchronize();

  /* release device memory */
  cudaFree(d_y);
  cudaFree(d_in_layer);
  cudaFree(d_bias);
  cudaFree(d_weight);
  cudaFree(d_out_layer);

}


/*********************************************
 * GPU kernel
 * Layer 1, Step 1: 
 * init values of feature maps at bias value 
 ********************************************/
/*__global__ void layer1_init_bias(float* d_y, float* d_bias) {
	int total_image_dim = (out_y_dim * out_x_dim);
	int total_work = (out_y_dim * out_x_dim * out_channel_num);

	int blocks_per_grid = gridDim.z * gridDim.y * gridDim.x;
	int work_per_block = split(total_work, blocks_per_grid);

	int block_id = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
	int min_work_of_block = work_per_block * block_id;
	int max_work_of_block = min_work_of_block + work_per_block - 1;
	if (max_work_of_block > total_work - 1) max_work_of_block = total_work - 1;

	int min_z_of_block = min_work_of_block / total_image_dim;
	int max_z_of_block = max_work_of_block / total_image_dim;

	int threads_per_block = blockDim.z * blockDim.y * blockDim.x;
	int work_per_thread = split(work_per_block, threads_per_block);

	int thread_id = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
	int min_work_of_thread = min_work_of_block +  work_per_thread * thread_id;
	int max_work_of_thread = min_work_of_thread + work_per_thread - 1;
	if (max_work_of_thread - min_work_of_thread > work_per_thread - 1) max_work_of_thread = work_per_thread - 1;

	__shared__ float bias;

	for (int z = min_z_of_block; z <= max_z_of_block; z ++) {
		if (thread_id == 0) {
			bias = d_bias [z];
		}

		__syncthreads();

		for (int work = min_work_of_thread; work <= max_work_of_thread; work ++) {
			d_y[work] = bias;
		}
	}
}/*/
__global__ void layer1_init_bias(float* d_y, float* d_bias) {
	int total_work_size = out_size;
	int total_workers = (gridDim.x * gridDim.y * gridDim.z) * (blockDim.x * blockDim.y * blockDim.z);
	int worker_id = ((((((blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.z + threadIdx.z) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);

	for (int n = worker_id; n < total_work_size; n += total_workers) {
		int z = (n / out_img_size);

		d_y[n] = d_bias[z];
	}
}//*/

/*********************************************
 * GPU kernel
 * Layer 1, Step 2: 
 * loop over output feature maps
 ********************************************/
__global__ void layer1_feature_maps(float* d_y, unsigned char* d_in_layer, float* d_weight) {
	__shared__ unsigned char in_buffer[SHARED_DIM];

	int z_per_layer = split(out_channel_num, gridDim.z);
	int min_z_of_block = blockIdx.z * z_per_layer;
	int max_z_of_block = min_z_of_block + (z_per_layer - 1);
	if (max_z_of_block >= out_channel_num) max_z_of_block = out_channel_num - 1;

	int out_x_per_block = split(out_x_dim, gridDim.x);
	int out_y_per_block = split(out_y_dim, gridDim.y);

	int min_x_of_block = blockIdx.x * out_x_per_block;
	int max_x_of_block = min_x_of_block + (out_x_per_block - 1);
	if (max_x_of_block >= out_x_dim) max_x_of_block = out_x_dim - 1;

	int min_y_of_block = blockIdx.y * out_y_per_block;
	int max_y_of_block = min_y_of_block + (out_y_per_block - 1);
	if (max_y_of_block >= out_y_dim) max_y_of_block = out_y_dim - 1;

	int in_x_per_buffer = SHARED_X_DIM;
	int in_y_per_buffer = SHARED_Y_DIM;

	int out_x_per_buffer = (in_x_per_buffer - window_x_dim) / window_x_stride + 1;
	int out_y_per_buffer = (in_y_per_buffer - window_y_dim) / window_y_stride + 1;

	int thread_id = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
	int threads_per_block = blockDim.z * blockDim.y * blockDim.x;

	for (int z = min_z_of_block; z <= max_z_of_block; z ++ ) {
		for (int n = 0; n < out_x_per_block * out_y_per_block; n += out_x_per_buffer * out_y_per_buffer) {
			int buffer_out_x_offset = min_x_of_block + (n % (out_x_per_buffer * split(out_x_per_block, out_x_per_buffer)));
			int buffer_out_y_offset = min_y_of_block + (n / (out_x_per_buffer * split(out_x_per_block, out_x_per_buffer)));

			int buffer_in_x_offset = buffer_out_x_offset * window_x_stride;
			int buffer_in_y_offset = buffer_out_y_offset * window_y_stride;

			{
				int uv_per_thread = split(in_x_per_buffer * in_y_per_buffer, threads_per_block);
				int uv_thread_offset = thread_id * uv_per_thread;
				for (int uv = 0; uv < uv_per_thread && (uv_thread_offset + uv) < in_x_per_buffer * in_y_per_buffer; uv ++) {
					int u = uv % in_x_per_buffer;
					int v = uv / in_x_per_buffer;
					in_buffer[v * in_y_per_buffer + u] = d_in_layer[z * in_img_size + (v + buffer_in_y_offset) * in_x_dim + (u + buffer_in_x_offset)];
				}
			}

			__syncthreads();

			{
				int uv_per_thread = split(out_x_per_buffer * out_y_per_buffer, threads_per_block);
				int uv_thread_offset = thread_id * uv_per_thread;
				for (int uv = 0; uv < uv_per_thread && (uv_thread_offset + uv) < out_x_per_buffer * out_y_per_buffer; uv ++) {
					int u = uv % out_x_per_buffer;
					int v = uv / out_x_per_buffer;

					float convolution = 0;
					for (int m = 0; m < window_x_dim; m ++) {
						for (int n = 0; n < window_y_dim; n ++) {
							convolution += in_buffer[(v * window_y_stride + n) * in_x_per_buffer + (u * window_x_stride + m)] * d_weight[z * window_size + n * window_x_dim + m];
						}
					}
					//printf("convolution at (%d, %d): %.6f\n", (buffer_out_x_offset + u), (buffer_out_y_offset + v), convolution);
					d_y[(buffer_out_y_offset + v) * out_x_dim + (buffer_out_x_offset + u)] += convolution;
				}
			}
		}
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
