#ifndef DHASH_CALCULATOR_H
#define DHASH_CALCULATOR_H

#include <dirent.h>
#include <stdbool.h>
#include <stdint.h>

// Constants
#define MAX_FILES 4096
#define MAX_FILENAME 256
#define DHASH_WIDTH 9
#define DHASH_HEIGHT 8
#define THREADS_PER_BLOCK 128

#define MAX_PATH_LENGTH 1024
#define SHARED_MEMORY_THRESHOLD 1024
#define MAX_BLOCKS_SHARED 32

// Structs
typedef struct {
  char filename[MAX_FILENAME];
  uint64_t hash;
} ImageHash;

typedef struct {
  int img1_idx;
  int img2_idx;
  int distance;
} ComparisonResult;

// Kernels
#ifdef __CUDACC__
__global__ void compute_dHash_kernel(unsigned char *images, uint64_t *hashes,
                                     int numImages);

__global__ void compare_hashes_kernel(uint64_t *hashes,
                                      ComparisonResult *results,
                                      int *result_count, int numImages,
                                      int threshold);

__global__ void compare_hashes_kernel_shared(uint64_t *hashes,
                                             ComparisonResult *results,
                                             int *result_count, int numImages,
                                             int threshold);
#endif

// Host functions
int is_image_file(const char *filename);

void resize_to_dHash(unsigned char *input, int width, int height, int channels,
                     unsigned char *output);

int count_images(struct dirent **entries, int n_entries);

int process_image(const char *filepath, const char *filename,
                  unsigned char *batch_data, int batch_index,
                  char filenames[][MAX_FILENAME]);

int hamming_distance(uint64_t hash1, uint64_t hash2);

int process_directory(const char *directory, ImageHash *results, bool gpu);

int process_dir_cpu(int count, unsigned char *h_batch_data,
                    char filenames[][MAX_FILENAME], ImageHash *results);

int process_dir_gpu(int count, unsigned char *h_batch_data,
                    char filenames[][MAX_FILENAME], ImageHash *results);

void print_results(ImageHash *hashes, int count);

int find_similar_images_gpu(ImageHash *hashes, int count, int threshold);

int find_similar_images_cpu(ImageHash *hashes, int count, int threshold);

int find_similar_images(ImageHash *hashes, int count, int threshold, bool gpu);

// Macros
#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                   \
      fprintf(stderr, "code: %d, reason: %s\n", error,                         \
              cudaGetErrorString(error));                                      \
    }                                                                          \
  }

#define CUDA_CHECK_MALLOC_WITH_CODE(ptr, size, cleanup_code, error_code)       \
  do {                                                                         \
    cudaError_t err = cudaMalloc(&ptr, size);                                  \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));             \
      cleanup_code;                                                            \
      return error_code;                                                       \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_MALLOC(ptr, size, cleanup_code)                             \
  CUDA_CHECK_MALLOC_WITH_CODE(ptr, size, cleanup_code, 0)

#define CUDA_CHECK_MEMCPY_WITH_CODE(dst, src, size, kind, cleanup_code,        \
                                    error_code)                                \
  do {                                                                         \
    cudaError_t err = cudaMemcpy(dst, src, size, kind);                        \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA memcpy failed: %s\n", cudaGetErrorString(err));             \
      cleanup_code;                                                            \
      return error_code;                                                       \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_MEMCPY(dst, src, size, kind, cleanup_code)                  \
  CUDA_CHECK_MEMCPY_WITH_CODE(dst, src, size, kind, cleanup_code, 0)

#define RGB_TO_GRAY(r, g, b)                                                   \
  ((unsigned char)(0.299f * (r) + 0.587f * (g) + 0.114f * (b)))

#endif // DHASH_CALCULATOR_H
