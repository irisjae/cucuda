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
#define SHARED_X_DIM 128 // shared memory buffer width
#define SHARED_Y_DIM 128 // shared memory buffer height
#define IN_Y_DIM 720 // input image height
#define IN_X_DIM 1280 // input image width
#define WINDOW_X_DIM 6 // convolution width
#define WINDOW_Y_DIM 6 // convolution height
#define WINDOW_X_STRIDE 2 // convolution x stride
#define WINDOW_Y_STRIDE 2 // convolution y stride
#define OUT_CHANNEL_NUM 6 // number of filters of convolution

//TODO: warn on stride wider than dim 
//TODO: warn on shared dims minus window dims not divisible by window stride

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
#define SHARED_SIZE (SHARED_X_DIM * SHARED_Y_DIM) // total shared memory buffer size
#define OUT_Y_DIM ((IN_Y_DIM - WINDOW_Y_DIM) / WINDOW_Y_STRIDE + 1) // output image height
#define OUT_X_DIM ((IN_X_DIM - WINDOW_X_DIM) / WINDOW_X_STRIDE + 1) // output image width
#define WINDOW_SIZE (WINDOW_X_DIM * WINDOW_Y_DIM) // total convolution size
#define IN_IMG_SIZE (IN_Y_DIM * IN_X_DIM) // total input size per image
#define IN_SIZE IN_IMG_SIZE // total input size
#define OUT_IMG_SIZE (OUT_Y_DIM * OUT_X_DIM) // total output size per image
#define OUT_SIZE (OUT_IMG_SIZE * OUT_CHANNEL_NUM) // total output size

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
__global__ void layer1_init_bias(float* d_y, float* d_bias) {
	__shared__ float bias_buffer[OUT_CHANNEL_NUM];

	int w_total_work_size = OUT_CHANNEL_NUM;
	int w_total_workers = (blockDim.x * blockDim.y * blockDim.z);
	int w_worker_id = (((threadIdx.z) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);

	for (int w_n = w_worker_id; w_n < w_total_work_size; w_n += w_total_workers) {
		bias_buffer[w_n] = d_bias[w_n];
	}

	__syncthreads();

	int total_work_size = OUT_SIZE;
	int total_workers = (gridDim.x * gridDim.y * gridDim.z) * (blockDim.x * blockDim.y * blockDim.z);
	int worker_id = ((((((blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.z + threadIdx.z) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);

	for (int n = worker_id; n < total_work_size; n += total_workers) {
		int z = (n / OUT_IMG_SIZE);

		d_y[n] = bias_buffer[z];
	}
}

/*********************************************
 * GPU kernel
 * Layer 1, Step 2: 
 * loop over output feature maps
 ********************************************/
#define TILE_X_dim ((SHARED_X_DIM - WINDOW_X_DIM) / WINDOW_X_STRIDE + 1)
#define TILE_Y_dim ((SHARED_Y_DIM - WINDOW_Y_DIM) / WINDOW_Y_STRIDE + 1)
#define X_TILES split(OUT_X_DIM, TILE_X_dim)
#define Y_TILES split(OUT_Y_DIM, TILE_Y_dim)
__global__ void layer1_feature_maps(float* d_y, unsigned char* d_in_layer, float* d_weight) {
	__shared__ unsigned char in_buffer[SHARED_SIZE];
	__shared__ float weight_buffer[WINDOW_SIZE * OUT_CHANNEL_NUM];

	int w_total_work_size = WINDOW_SIZE * OUT_CHANNEL_NUM;
	int w_total_workers = (blockDim.x * blockDim.y * blockDim.z);
	int w_worker_id = (((threadIdx.z) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);

	for (int w_n = w_worker_id; w_n < w_total_work_size; w_n += w_total_workers) {
		weight_buffer[w_n] = d_weight[w_n];
	}

	int total_work_size = X_TILES * Y_TILES * OUT_CHANNEL_NUM;
	int total_workers = (gridDim.x * gridDim.y * gridDim.z);
	int worker_id = (((blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x);

	for (int n = worker_id; n < total_work_size; n += total_workers) {
		int z = (n / (X_TILES * Y_TILES));
		int tile_x = (n % (X_TILES * Y_TILES)) % (X_TILES);
		int tile_y = (n % (X_TILES * Y_TILES)) / (X_TILES);

		int s_x_offset = tile_x * TILE_X_dim * WINDOW_X_STRIDE;
		int s_y_offset = tile_y * TILE_Y_dim * WINDOW_Y_STRIDE;

		int s_total_work_size = SHARED_X_DIM * SHARED_Y_DIM;;
		int s_total_workers = (blockDim.x * blockDim.y * blockDim.z);
		int s_worker_id = (((threadIdx.z) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);

		for (int s_n = s_worker_id; s_n < s_total_work_size; s_n += s_total_workers) {
			int dx = (s_n % SHARED_X_DIM);
			int dy = (s_n / SHARED_X_DIM);
			if ((dy + s_y_offset) < IN_Y_DIM && (dx + s_x_offset) < IN_X_DIM) {
				in_buffer[(dy) * SHARED_X_DIM + dx] = d_in_layer[(dy + s_y_offset) * IN_X_DIM + (dx + s_x_offset)];
			}
		}

		__syncthreads();

		int t_x_offset = tile_x * TILE_X_dim;
		int t_y_offset = tile_y * TILE_Y_dim;

		int t_total_work_size = TILE_X_dim * TILE_Y_dim;;
		int t_total_workers = (blockDim.x * blockDim.y * blockDim.z);
		int t_worker_id = (((threadIdx.z) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);

		for (int t_n = t_worker_id; t_n < t_total_work_size; t_n += t_total_workers) {
			int dx = (t_n % TILE_X_dim);
			int dy = (t_n / TILE_X_dim);
			if ((dy + t_y_offset) < OUT_Y_DIM && (dx + t_x_offset) < OUT_X_DIM) {
				float convolution = 0;
				for (int i = 0; i < WINDOW_X_DIM; i ++) {
					for (int j = 0; j < WINDOW_Y_DIM; j ++) {
						convolution +=
							weight_buffer[((z) * WINDOW_Y_DIM + j) * WINDOW_X_DIM + i]
							* in_buffer[(WINDOW_Y_STRIDE * dy + j) * SHARED_X_DIM + (WINDOW_X_STRIDE * dx + i)];
					}
				}
				d_y[(z * OUT_Y_DIM + (t_y_offset + dy)) * OUT_X_DIM + (t_x_offset + dx)] += convolution;
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
	int total_work_size = OUT_SIZE;
	int total_workers = (gridDim.x * gridDim.y * gridDim.z) * (blockDim.x * blockDim.y * blockDim.z);
	int worker_id = ((((((blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.z + threadIdx.z) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);

	for (int n = worker_id; n < total_work_size; n += total_workers) {
		d_out_layer[n] = (unsigned char) (255.999f / (1 + expf(- d_y[n] / 256)));
	}
}
