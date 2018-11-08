#include<stdio.h>
#include<stdlib.h>

/**
 *
 * PARAMETERS
 *
 */
#define VERBOSE
#define PROFILE
//#define DRY_RUN

#define INPUT_FILE "input.ppm"
#define OUTPUT_FILE "output.ppm"

#define FILTER_WIDTH 3
#define FILTER_HEIGHT 3
#define FILTER { \
	{ 0.05, 0.1, 0.05 }, \
	{ 0.1, 0.4, 0.1 }, \
	{ 0.05, 0.1, 0.05 }, \
}
#define CONSTANT_FILTER/*
//#define GLOBAL_FILTER/*
#define LOCAL_FILTER/**/

#define GRID_DIM 32,32
#define BLOCK_DIM 32,32

#define SHARED_DIM 0


/**
 *
 * METADATA
 *
 */
#define CREATOR "Jae"



/**
 *
 * CUDA UTILS
 *
 */
#define cuda_try( ans ) { __cuda_try((ans), __FILE__, __LINE__); }
inline void __cuda_try( cudaError_t code, const char * file, int line, bool abort=true ) {
	if (code != cudaSuccess) {
		fprintf(stderr, "CUDA THROW %s CAUGHT AT %s LINE %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/**
 *
 * UTILS
 *
 */
#define RGB_COMPONENT_COLOR 255

#define STR_EXPAND(...) #__VA_ARGS__
#define ARG(...) STR_EXPAND(__VA_ARGS__)

#define x_radius (FILTER_WIDTH / 2)
#define y_radius (FILTER_HEIGHT / 2)

#define split( n, among ) { ((n + (among - 1)) / among) }

#if defined(CONSTANT_FILTER) && !defined(GLOBAL_FILTER) && !defined(LOCAL_FILTER)
__constant__ float filter[FILTER_HEIGHT][FILTER_WIDTH] = FILTER;
#elif defined(GLOBAL_FILTER) && !defined(CONSTANT_FILTER) && !defined(LOCAL_FILTER)
__device__ float filter[FILTER_HEIGHT][FILTER_WIDTH] = FILTER;
#endif





typedef struct {
	unsigned char red, green, blue;
} PPMPixel;

typedef struct {
	int x, y;
	PPMPixel * data;
} PPMImage;



static PPMImage * readPPM( const char * filename ) {
	char buff[16];
	PPMImage * img;
	FILE * fp;
	int c, rgb_comp_color;
	//open PPM file for reading
	fp = fopen(filename, "rb");
	if (!fp) {
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	//read image format
	if (!fgets(buff, sizeof(buff), fp)) {
		perror(filename);
		exit(1);
	}

	//check the image format
	if (buff[0] != 'P' || buff[1] != '6') {
		fprintf(stderr, "Invalid image format (must be 'P6')\n");
		exit(1);
	}

	//alloc memory form image
	img = (PPMImage *)malloc(sizeof(PPMImage));
	if (!img) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	//check for comments
	c = getc(fp);
	while (c == '#') {
		while (getc(fp) != '\n') ;
		c = getc(fp);
	}

	ungetc(c, fp);
	//read image size information
	if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
		fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
		exit(1);
	}

	//read rgb component
	if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
		fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
		exit(1);
	}

	//check rgb component depth
	if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
		fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
		exit(1);
	}

	while (fgetc(fp) != '\n') ;
	//memory allocation for pixel data
	img->data = (PPMPixel *)malloc(img->x * img->y * sizeof(PPMPixel));

	if (!img) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	//read pixel data from file
	if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
		fprintf(stderr, "Error loading image '%s'\n", filename);
		exit(1);
	}

	fclose(fp);
	return img;
}

void writePPM( const char * filename, PPMImage * img ) {
	FILE * fp;
	//open file for output
	fp = fopen(filename, "wb");
	if (!fp) {
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	//write the header file
	//image format
	fprintf(fp, "P6\n");

	//comments
	fprintf(fp, "# Created by %s\n",CREATOR);

	//image size
	fprintf(fp, "%d %d\n",img->x,img->y);

	// rgb component depth
	fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);

	// pixel data
	fwrite(img->data, 3 * img->x, img->y, fp);
	fclose(fp);
}



__global__ void blur_kernel( PPMImage * img, PPMImage * out ) {
#if defined(LOCAL_FILTER) && !defined(CONSTANT_FILTER) && !defined(GLOBAL_FILTER)
	float filter[FILTER_HEIGHT][FILTER_WIDTH] = FILTER;
#endif

	int img_x = img->x;
	int img_y = img->y;

	int x_per_block = split(img_x, gridDim.x);
	int y_per_block = split(img_y, gridDim.y);

	int min_x_of_block = x_per_block * blockIdx.x;
	int min_y_of_block = y_per_block * blockIdx.y;

	int max_x_of_block = min_x_of_block + x_per_block - 1;
	int max_y_of_block = min_y_of_block + y_per_block - 1;

	if (max_x_of_block > img_x - 1) max_x_of_block = img_x - 1;
	if (max_y_of_block > img_y - 1) max_y_of_block = img_y - 1;

	int work_per_block = (x_per_block * y_per_block);
	int threads_per_block = (blockDim.x * blockDim.y);

	int work_per_thread = split(work_per_block, threads_per_block);
	int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

	int thread_work_offset = work_per_thread * thread_id;
	
#if SHARED_DIM
	int x_sources_per_block = (x_per_block + 2 * x_radius);
	int y_sources_per_block = (y_per_block + 2 * y_radius);
	int sources_per_block = x_sources_per_block * y_sources_per_block;
	int sources_per_thread = split(sources_per_block, threads_per_block);
	int thread_source_offset = sources_per_thread * thread_id;

	__shared__ PPMPixel block_source[SHARED_DIM];

	for (int i = 0; i < sources_per_thread && (thread_source_offset + i) < sources_per_block; i ++) {
		int x_from_block_sources = (thread_source_offset + i) % x_sources_per_block;
		int y_from_block_sources = (thread_source_offset + i) / x_sources_per_block;

		int x_from_block = x_from_block_sources - x_radius;
		int y_from_block = y_from_block_sources - y_radius;

		int x = x_from_block + min_x_of_block;
		int y = y_from_block + min_y_of_block;

		int source_index = y_from_block_sources * x_sources_per_block + x_from_block_sources;

		if (source_index < SHARED_DIM)
			block_source[source_index] =
				(x >= 0 && x < img_x && y >= 0 && y < img_y)
				? img->data[y * img_x + x]
				: (PPMPixel) { 0, 0, 0 };
	}

	__syncthreads();
#endif

	out->x = img_x;
	out->y = img_y;
	for (int i = 0; i < work_per_thread && thread_work_offset + i < work_per_block; i ++) {
		int x_from_block = (thread_work_offset + i) % x_per_block;
		int y_from_block = (thread_work_offset + i) / x_per_block;

		int x = x_from_block + min_x_of_block;
		int y = y_from_block + min_y_of_block;

		if (x <= max_x_of_block && y <= max_y_of_block) {
			float r = 0;
			float g = 0;
			float b = 0;
			for (int x_from_point = - x_radius; x_from_point <= + x_radius; x_from_point ++) {
				for (int y_from_point = - y_radius; y_from_point <= + y_radius; y_from_point ++) {
					int filter_x = x_radius + x_from_point;
					int filter_y = y_radius + y_from_point;

#if SHARED_DIM
					int source_index = (y_from_block + filter_y) * x_sources_per_block + (x_from_block + filter_x);
#endif
					PPMPixel filter_point = 
#if SHARED_DIM
						source_index < SHARED_DIM
						? block_source[source_index]
						: 
#endif
						(x + x_from_point) >= 0 && (x + x_from_point) < img_x
						&& (y + y_from_point) >= 0 && (y + y_from_point) < img_y
						? img->data[(y + y_from_point) * img_x + (x + x_from_point)]
						: (PPMPixel) { 0, 0, 0 };
					float filter_weight = filter[filter_x][filter_y];

					r += filter_weight * filter_point.red;
					g += filter_weight * filter_point.green;
					b += filter_weight * filter_point.blue;
				}
			}

			out->data[y * img_x + x] = (PPMPixel) { r, g, b };
		}
	}
}


dim3 grid_dim(GRID_DIM);
dim3 block_dim(BLOCK_DIM);

void gaussian_blur( PPMImage * img ) {
	int n = img->x * img->y;
#ifdef VERBOSE
	printf("blurring...\n");
	int img_x = img->x;
	int img_y = img->y;

	int x_per_block = split(img_x, grid_dim.x);
	int y_per_block = split(img_y, grid_dim.y);

	int x_sources_per_block = (x_per_block + 2 * x_radius);
	int y_sources_per_block = (y_per_block + 2 * y_radius);
	int sources_per_block = x_sources_per_block * y_sources_per_block;
	int work_per_block = (x_per_block * y_per_block);
	int threads_per_block = (block_dim.x * block_dim.y);

	int sources_per_thread = split(sources_per_block, threads_per_block);
	int work_per_thread = split(work_per_block, threads_per_block);
	printf("problem size %d*%d=%d, shared memory "ARG(SHARED_DIM)", filter size "ARG(FILTER_WIDTH)"*"ARG(FILTER_HEIGHT)"=%d, grid dim "ARG(GRID_DIM)"=%d, block size "ARG(BLOCK_DIM)"=%d - %d,%d - %d->%d, thread size %d->%d\n"
	, img_x, img_y, n, FILTER_WIDTH * FILTER_HEIGHT, grid_dim.x * grid_dim.y * grid_dim.z, threads_per_block
	, x_per_block, y_per_block, sources_per_block, work_per_block, sources_per_thread, work_per_thread);
#endif

	PPMImage * dev_in;
	PPMImage * dev_out;
	PPMImage * host_temp = (PPMImage *) malloc(sizeof(PPMImage));
	* host_temp = (PPMImage) { .x = img->x, .y = img->y };
	cuda_try(cudaMalloc((void **)&(host_temp->data), n * sizeof(PPMPixel)));
	cuda_try(cudaMemcpy(host_temp->data, img->data, n * sizeof(PPMPixel), cudaMemcpyHostToDevice));
	cuda_try(cudaMalloc((void **)&dev_in, sizeof(PPMImage)));
	cuda_try(cudaMemcpy(dev_in, host_temp, sizeof(PPMImage), cudaMemcpyHostToDevice));

	cuda_try(cudaMalloc((void **)&(host_temp->data), n * sizeof(PPMPixel)));
	cuda_try(cudaMalloc((void **)&dev_out, sizeof(PPMImage)));
	cuda_try(cudaMemcpy(dev_out, host_temp, sizeof(PPMImage), cudaMemcpyHostToDevice));

#ifdef PROFILE
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
#endif
	blur_kernel<<<grid_dim,block_dim>>>(dev_in, dev_out);
#ifdef PROFILE
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	printf("%f ms\n", time);
#endif
	cuda_try(cudaPeekAtLastError());

	cuda_try(cudaMemcpy(host_temp, dev_out, sizeof(PPMImage), cudaMemcpyDeviceToHost));
	cuda_try(cudaMemcpy(img->data, host_temp->data, n * sizeof(PPMPixel), cudaMemcpyDeviceToHost));
	cuda_try(cudaFree(host_temp->data));
	cuda_try(cudaFree(dev_out));

	cuda_try(cudaMemcpy(host_temp, dev_in, sizeof(PPMImage), cudaMemcpyDeviceToHost));
	cuda_try(cudaFree(host_temp->data));
	cuda_try(cudaFree(dev_in));
#ifdef VERBOSE
	printf("blurring complete\n");
#endif
}


int main( void ) {
	PPMImage * image = readPPM(INPUT_FILE);
	gaussian_blur(image);
#if !defined(DRY_RUN)
	writePPM(OUTPUT_FILE, image);
#endif
}
