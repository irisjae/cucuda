#include <stdio.h>

/**
 * 
 * PARAMETERS
 * 
 */
#define VERBOSE
//#define DRY_RUN

/*#define USE_CPU /*
#define PROFILE_CPU // */
//#define USE_GPU /*
#define PROFILE_GPU // */

#define CPU_OUTPUT_FILE "julia_cpu.ppm"
#define GPU_OUTPUT_FILE "julia_gpu.ppm"


#define JULIA_X -0.8
#define JULIA_Y 0.156

#define SCALE 1.5
#define DIM 1000

/*#define PALE /*
#define WHITE // */

#define GRID_SIZE 1 /*
#define GRID_SIZE_2D DIM,DIM // */
#define BLOCK_SIZE 1 /*
#define BLOCK_SIZE_2D 1,1 // */


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
#if defined(GRID_SIZE) && !defined(GRID_SIZE_2D)
#define GRID_DIM GRID_SIZE
#elif !defined(GRID_SIZE) && defined(GRID_SIZE_2D)
#define GRID_DIM GRID_SIZE_2D
#endif

#if defined(BLOCK_SIZE) && !defined(BLOCK_SIZE_2D)
#define BLOCK_DIM BLOCK_SIZE
#elif !defined(BLOCK_SIZE) && defined(BLOCK_SIZE_2D)
#define BLOCK_DIM BLOCK_SIZE_2D
#endif

#define STR_EXPAND(...) #__VA_ARGS__
#define ARG(...) STR_EXPAND(__VA_ARGS__)





struct cppComplex {
	float r; 
	float i;
	__host__ __device__ cppComplex( float a, float b ) : r(a), i(b) {}
	__host__ __device__ float magnitude2( void ) {
		return r * r + i * i;
	}
	__host__ __device__ cppComplex operator *( const cppComplex& a ) {
		return cppComplex(r * a.r - i * a.i, i * a.r + r * a.i);
	}
	__host__ __device__ cppComplex operator +( const cppComplex& a ) {
		return cppComplex(r + a.r, i + a.i);
	}
};






int julia_cpu( int x, int y ) {
	float jx = SCALE * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = SCALE * (float)(DIM / 2 - y) / (DIM / 2);

	cppComplex c(JULIA_X, JULIA_Y);
	cppComplex a(jx, jy);

	int i = 0;
	for(; i < 200; i ++) {
		a = a * a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}

	return 1;
}

void cpu_draw( unsigned char * pixels ) {
#ifdef VERBOSE
	printf("cpu drawing...\n");
#endif
#ifdef PROFILE_CPU
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
#endif
	for (int x = 0; x < DIM; ++x) {
		for (int y = 0; y < DIM; ++ y) {
			pixels[x + y * DIM] = 255 * julia_cpu(x, y);
		}
	}
#ifdef PROFILE_CPU
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	printf("%f ms\n", time);
#endif
#ifdef VERBOSE
	printf("cpu drawing complete\n");
#endif
}







__global__ void kernel( unsigned char * ptr, int thread_size ) {
	int t_id =
#if defined(GRID_SIZE) && !defined(GRID_SIZE_2D)
		blockIdx.x
#elif !defined(GRID_SIZE) && defined(GRID_SIZE_2D)
		(blockIdx.x + blockIdx.y * gridDim.x)
#endif
#if defined(BLOCK_SIZE) && !defined(BLOCK_SIZE_2D)
		* blockDim.x
		+ threadIdx.x;
#elif !defined(BLOCK_SIZE) && defined(BLOCK_SIZE_2D)
		* (blockDim.x * blockDim.y)
		+ (threadIdx.y * blockDim.x + threadIdx.x);
#endif

	int offset = thread_size * t_id;

	int i = 0;

	cppComplex c(JULIA_X, JULIA_Y);

	int x = (i + offset) % DIM;
	int y = (i + offset) / DIM;

	float jx = SCALE * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = SCALE * (float)(DIM / 2 - y) / (DIM / 2);
		
	for(; i < thread_size && offset + i < DIM * DIM; i ++) {

		cppComplex a(jx, jy);

		int j = 0;
		for(; j < 200; j ++){
			a = a * a + c;
			if (a.magnitude2() > 1000)
				break;
		}

		if (j < 200)
			ptr[offset + i] = 0;
		else
			ptr[offset + i] = 255;

		x ++;
		if (x == DIM) {
			x = 0;
			y ++;
			jy = SCALE * (float)(DIM / 2 - y) / (DIM / 2);
		}
		jx = SCALE * (float)(DIM / 2 - x) / (DIM / 2);
	}
}

void gpu_draw( unsigned char * gpu_pixels ) {
	int n = DIM * DIM;
	dim3 grid_dim(GRID_DIM);
	dim3 block_dim(BLOCK_DIM);
	int grid_size = grid_dim.x * grid_dim.y * grid_dim.z;
	int block_size = block_dim.x * block_dim.y * block_dim.z;
	int thread_size = (n + (grid_size * block_size - 1)) / (grid_size * block_size);

#ifdef VERBOSE
	printf("gpu drawing...\n");
	printf("problem size %d, grid dim "ARG(GRID_DIM)"=%d, block size "ARG(BLOCK_DIM)"=%d, thread size %d\n", n, grid_size, block_size, thread_size);
#endif

	unsigned char * dev_bitmap; 
	cuda_try(cudaMalloc((void **)&dev_bitmap, n * sizeof(unsigned char)));

#ifdef PROFILE_GPU
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
#endif
	kernel<<<grid_dim,block_dim>>>(dev_bitmap, thread_size);
#ifdef PROFILE_GPU
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	printf("%f ms\n", time);
#endif
	cuda_try(cudaPeekAtLastError());

	cuda_try(cudaMemcpy(gpu_pixels, dev_bitmap, n * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	cuda_try(cudaFree(dev_bitmap));

#ifdef VERBOSE
	printf("gpu drawing complete\n");
#endif
}

void draw_file( char * path, unsigned char * pixels ) {
	FILE * f = fopen(path, "wb");
	fprintf(f, "P6\n%i %i 255\n", DIM, DIM);
	for (int y = 0; y < DIM; y ++) {
		for (int x = 0; x < DIM; x ++) {
#if !defined(PALE) && !defined(WHITE)
			fputc(pixels[(y * DIM + x)], f);
			fputc(0, f);
			fputc(0, f);
#elif defined(PALE) && !defined(WHITE)
			fputc(pixels[(y * DIM + x)] * 0.9, f);
			fputc(pixels[(y * DIM + x)] * 0.3, f);
			fputc(pixels[(y * DIM + x)] * 0.3, f);
#elif defined(WHITE) && !defined(PALE)
			fputc(pixels[(y * DIM + x)] * 0.9, f);
			fputc(pixels[(y * DIM + x)] * 0.9, f);
			fputc(pixels[(y * DIM + x)] * 0.9, f);
#else
	#warning Make up your mind on the color!
	#error You must choose either PALE, WHITE, or neither!
#endif
		}
	}
	fclose(f);
}


int main( void ) {
#ifdef VERBOSE
	printf("julia set of "ARG(JULIA_X)","ARG(JULIA_Y)" resolution "ARG(DIM)"*"ARG(DIM)" scale "ARG(SCALE)"\n");
#endif

#if defined(USE_CPU) || defined(PROFILE_CPU)
	unsigned char * pixels = new unsigned char[DIM * DIM]; 
	cpu_draw(pixels);
#if !defined(DRY_RUN)
	draw_file(CPU_OUTPUT_FILE, pixels);
#endif
	delete [] pixels;
#endif

#if defined(USE_GPU) || defined(PROFILE_GPU)
	unsigned char *gpu_pixels = new unsigned char[DIM * DIM]; 
	gpu_draw(gpu_pixels);
#if !defined(DRY_RUN)
	draw_file(GPU_OUTPUT_FILE, gpu_pixels);
#endif
	delete [] gpu_pixels; 
#endif
}
