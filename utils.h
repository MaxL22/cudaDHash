#ifndef _UTILS_H
#define _UTILS_H

// Named constants
#define MAX_PATH_LENGTH 1024
#define SHARED_MEMORY_THRESHOLD 1024
#define MAX_BLOCKS_SHARED 32

#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                   \
      fprintf(stderr, "code: %d, reason: %s\n", error,                         \
              cudaGetErrorString(error));                                      \
    }                                                                          \
  }

// Enhanced malloc macro with specific error codes
#define CUDA_CHECK_MALLOC_WITH_CODE(ptr, size, cleanup_code, error_code)       \
  do {                                                                         \
    cudaError_t err = cudaMalloc(&ptr, size);                                  \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));             \
      cleanup_code;                                                            \
      return error_code;                                                       \
    }                                                                          \
  } while (0)

// Original macro
#define CUDA_CHECK_MALLOC(ptr, size, cleanup_code)                             \
  CUDA_CHECK_MALLOC_WITH_CODE(ptr, size, cleanup_code, 0)

// Enhanced memcpy macro with specific error codes
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

// Original macro
#define CUDA_CHECK_MEMCPY(dst, src, size, kind, cleanup_code)                  \
  CUDA_CHECK_MEMCPY_WITH_CODE(dst, src, size, kind, cleanup_code, 0)

#define RGB_TO_GRAY(r, g, b)                                                   \
  ((unsigned char)(0.299f * (r) + 0.587f * (g) + 0.114f * (b)))

#endif // _COMMON_H
