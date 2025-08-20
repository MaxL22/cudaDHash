#include "utils.h"
#include <ctype.h>
#include <cuda_runtime.h>
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define MAX_FILES 4096
#define MAX_FILENAME 256
#define DHASH_WIDTH 9
#define DHASH_HEIGHT 8
#define THREADS_PER_BLOCK 256

// Image hash struct
typedef struct {
  char filename[MAX_FILENAME];
  uint64_t hash;
} ImageHash;

// Structure to store comparison results
typedef struct {
  int img1_idx;
  int img2_idx;
  int distance;
} ComparisonResult;

/**
 * compute_dHash_kernel(): CUDA kernel to compute dHash for multiple images
 * using shared memory
 * @images: image vector
 * @hashes: hashes vector (to be filled)
 * @numImages: number of images to be computed
 */
__global__ void compute_dHash_kernel(unsigned char *images, uint64_t *hashes,
                                     int numImages) {
  // thread id, return if too much
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numImages)
    return;

  // Use shared memory to load the image data, each thread loads its own image
  extern __shared__ unsigned char shared_img[];
  unsigned char *local_img =
      &shared_img[threadIdx.x * DHASH_WIDTH * DHASH_HEIGHT];

  // Load image pointed by global data in shared memory
  unsigned char *global_img = images + idx * DHASH_WIDTH * DHASH_HEIGHT;
  for (int i = 0; i < DHASH_WIDTH * DHASH_HEIGHT; i++) {
    local_img[i] = global_img[i];
  }

  __syncthreads();

  uint64_t hash = 0;

  // Process 8x8 grid (comparing adjacent horizontal pixels)
  for (int row = 0; row < DHASH_HEIGHT; row++) {
    for (int col = 0; col < DHASH_WIDTH - 1; col++) {
      int pixelIdx = row * DHASH_WIDTH + col;
      // Compare adjacent horizontal pixels
      if (local_img[pixelIdx] > local_img[pixelIdx + 1]) {
        int bitPos = row * (DHASH_WIDTH - 1) + col; // Hash bit to be filled
        // Write the bit, if left is brighter than right
        hash |= (1ULL << bitPos);
      }
    }
  }
  hashes[idx] = hash;
}

/**
 * is_image_file(): checks if file has image extension
 * @filename: self-explainatory
 *
 * Returns: 1 if the extension of @filename is jpg|jpeg|png|bmp, 0 otherwise
 */
int is_image_file(const char *filename) {
  const char *ext = strrchr(filename, '.'); // Find last . in str
  if (!ext)                                 // No extension
    return 0;

  char lower_ext[10];
  int i;
  for (i = 0; i < 9 && ext[i]; i++) {
    lower_ext[i] = tolower(ext[i]);
  }
  lower_ext[i] = '\0';

  return (strcmp(lower_ext, ".jpg") == 0 || strcmp(lower_ext, ".jpeg") == 0 ||
          strcmp(lower_ext, ".png") == 0 || strcmp(lower_ext, ".bmp") == 0);
}

/**
 * resize_to_dHash(): Simple bilinear resize to 9x8 grayscale
 * @input: input image
 * @width: width of input image
 * @height: height of image
 * @channels: number of channels
 * @output: output image
 *
 * Writes into @output a 9x8 grayscale version of @input
 */
void resize_to_dHash(unsigned char *input, int width, int height, int channels,
                     unsigned char *output) {
  for (int y = 0; y < DHASH_HEIGHT; y++) {
    for (int x = 0; x < DHASH_WIDTH; x++) {
      // Compute source coordinates from the ones in the grayscale image
      float src_x = ((float)x / (DHASH_WIDTH - 1)) * (width - 1);
      float src_y = ((float)y / (DHASH_HEIGHT - 1)) * (height - 1);

      int cx[2], cy[2];

      cx[0] = (int)src_x;
      cy[0] = (int)src_y;
      cx[1] = cx[0] + 1 < width ? cx[0] + 1 : cx[0];
      cy[1] = cy[0] + 1 < height ? cy[0] + 1 : cy[0];

      // Get pixel values
      int idx[4];
      for (int i = 0; i < 4; i++) {
        idx[i] = (cy[i >> 1] * width + cx[i & 0x1]) * channels;
      }

      unsigned char gray[4];

      // Convert to gray
      if (channels == 1) { // If already gray
        for (int i = 0; i < 4; i++) {
          gray[i] = input[idx[i]];
        }
      } else { // If RGB
        for (int i = 0; i < 4; i++) {
          gray[i] =
              RGB_TO_GRAY(input[idx[i]], input[idx[i] + 1], input[idx[i] + 2]);
        }
      }

      // Bilinear interpolation
      float fx = src_x - cx[0];
      float fy = src_y - cy[0];

      float top = gray[0] * (1.0f - fx) + gray[1] * fx;
      float bottom = gray[2] * (1.0f - fx) + gray[3] * fx;
      float result = top * (1.0f - fy) + bottom * fy;

      output[y * DHASH_WIDTH + x] = (unsigned char)result;
    }
  }
}

/**
 * count_images(): Count images in directory without closing the directory
 * @entries: directory
 * @n_entries: number of entries
 *
 * Returns the number of entries in @entries, up to MAX_FILES
 */
int count_images(struct dirent **entries, int n_entries) {
  int count = 0;
  for (int i = 0; i < n_entries && count < MAX_FILES; i++) {
    // If it's a regular file and an image, increase the count by 1
    if (entries[i]->d_type == DT_REG && is_image_file(entries[i]->d_name)) {
      count++;
    }
  }
  return count;
}

/**
 * process_image(): Process a single image file and add to batch
 * @filepath: path of the original image
 * @filename: name of the original image
 * @batch_data: batch of grayscale imgs
 * @batch_index: index of the current img
 * @filenames_ array of filenames
 *
 * Returns 0 if failed, 1 if successful
 */
int process_image(const char *filepath, const char *filename,
                  unsigned char *batch_data, int batch_index,
                  char filenames[][MAX_FILENAME]) {
  // Load the image and its parameters
  int width, height, channels;
  unsigned char *image_data =
      stbi_load(filepath, &width, &height, &channels, 0);

  if (!image_data) {
    printf("Failed to load image: %s\n", filepath);
    return 0;
  }

  // Resize the loaded img to 9x8 grayscale
  unsigned char *output_ptr =
      batch_data + batch_index * DHASH_WIDTH * DHASH_HEIGHT;
  resize_to_dHash(image_data, width, height, channels, output_ptr);

  // Store filename in batch, padded to MAX_FILENAME
  strncpy(filenames[batch_index], filename, MAX_FILENAME - 1);
  filenames[batch_index][MAX_FILENAME - 1] = '\0';

  // Free loaded image
  stbi_image_free(image_data);
  return 1;
}

/**
 * hamming_distance_device(): CUDA device function to calculate Hamming distance
 * between two hashes
 * @hash1: first hash
 * @hash2: second hash
 *
 * Returns the distance between the two hashes
 */
__device__ int hamming_distance_device(uint64_t hash1, uint64_t hash2) {
  uint64_t xor_result = hash1 ^ hash2;
  return __popcll(xor_result); // CUDA built-in
}

/**
 * hamming_distance(): Host function to calculate Hamming distance between two
 * hashes
 * @hash1: first hash
 * @hash2: second hash
 *
 * Returns the distance between the two hashes
 */
int hamming_distance(uint64_t hash1, uint64_t hash2) {
  uint64_t xor_result = hash1 ^ hash2;
  int count = 0;
  while (xor_result) {
    count += xor_result & 1;
    xor_result >>= 1;
  }
  return count;
}

/**
 * compare_hashes_kernel(): CUDA kernel to compare all hash pairs in parallel
 * @hashes: device array of hashes to compare
 * @results: device array for the result of the comparisons
 * @result_count: device counter for results written
 * @numImages: image count
 * @threshold: max Hamming distance for similarity
 */
__global__ void compare_hashes_kernel(uint64_t *hashes,
                                      ComparisonResult *results,
                                      int *result_count, int numImages,
                                      int threshold) {
  // Idk why it's a long long, don't remember
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;

  // Calculate total number of unique pairs: n*(n-1)/2
  long long total_pairs = ((long long)numImages * (numImages - 1)) / 2;
  if (idx >= total_pairs)
    return;

  // Convert linear index to (i,j) pair where i < j
  // Use double precision to avoid overflow, convert to have decimals
  double idx_d = (double)idx;
  /*
   * Short explanation: this is computing the coordinates starting from the
   * index, as if it was inside a triangular matrix (coordinates i and j); don't
   * bother calculating it again, you don't remember
   */
  int i = (int)((-1.0 + sqrt(1.0 + 8.0 * idx_d)) / 2.0);
  long long temp = ((long long)i * (i + 1)) / 2;
  int j = (int)(idx - temp + i + 1);

  // Ensure we don't go out of bounds
  if (i >= numImages || j >= numImages || i < 0 || j < 0)
    return;

  // Calculate Hamming distance
  int distance = hamming_distance_device(hashes[i], hashes[j]);

  // If distance is within threshold, add to results
  if (distance <= threshold) {
    int result_idx = atomicAdd(result_count, 1);

    // Make sure we don't overflow the results buffer (should never happen)
    if (result_idx < total_pairs) {
      results[result_idx].img1_idx = i;
      results[result_idx].img2_idx = j;
      results[result_idx].distance = distance;
    }
  }
}

/**
 * compare_hashes_kernel_shared(): compare hashes pairs using shared memory, for
 * smaller datasets
 * @hashes: device array of hashes to compare
 * @results: device array for the result of the comparisons
 * @result_count: device counter for results written
 * @numImages: image count
 * @threshold: max Hamming distance for similarity
 */
__global__ void compare_hashes_kernel_shared(uint64_t *hashes,
                                             ComparisonResult *results,
                                             int *result_count, int numImages,
                                             int threshold) {
  extern __shared__ uint64_t shared_hashes[];

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int block_size = blockDim.x;

  // Load hashes into shared memory
  for (int i = tid; i < numImages; i += block_size) {
    shared_hashes[i] = hashes[i];
  }
  __syncthreads();

  // Calculate which pairs this thread will handle
  long long total_pairs = ((long long)numImages * (numImages - 1)) / 2;
  long long pairs_per_block = (total_pairs + gridDim.x - 1) / gridDim.x;
  // Indexes of start and end pair inside the block
  long long start_pair = (long long)bid * pairs_per_block;
  long long end_pair = min(start_pair + pairs_per_block, total_pairs);

  // Process the pairs assigned to the thread, should be even across threads
  for (long long pair_idx = start_pair + tid; pair_idx < end_pair;
       pair_idx += block_size) {
    // Convert linear index to (i,j) pair with bounds checking
    double pair_d = (double)pair_idx;
    int i = (int)((-1.0 + sqrt(1.0 + 8.0 * pair_d)) / 2.0);
    int j = (int)(pair_idx - (((long long)i * (i + 1)) / 2.0) + i + 1);
    // Bounds check
    if (i >= numImages || j >= numImages || i < 0 || j < 0)
      continue;

    // Calculate Hamming distance using shared memory
    int distance = hamming_distance_device(shared_hashes[i], shared_hashes[j]);

    // If distance is within threshold, add to results

    // Check condition and create bitmask for the threads which satisfy the
    // condition
    unsigned mask = __ballot_sync(0xFFFFFFFF, distance <= threshold);
    // Only continue if at least one thread qualifies
    if (mask == 0)
      continue;

    int lane_id = threadIdx.x & 31; // Position within the warp
    bool thread_qualifies = (distance <= threshold);

    // Only continue if thread qualifies
    if (!thread_qualifies)
      continue;

    // Count qualifying threads before this one in the warp
    // (__popc() counts the number of bits set to 1)
    int warp_offset = __popc(mask & ((1U << lane_id) - 1));

    // First qualifying thread does atomic add for entire warp
    // (only one atomic add per warp)
    int base_idx;
    // __ffs() gets the position of the first set bit (1-indexed)
    if (lane_id == __ffs(mask) - 1) {
      int warp_count = __popc(mask);
      base_idx = atomicAdd(result_count, warp_count);
    }

    // Broadcast base index to all qualifying threads
    // __shfl_sync() copies value base_idx from lane __ffs(mask) - 1 to all bits
    // in mask
    base_idx = __shfl_sync(mask, base_idx, __ffs(mask) - 1);
    // Position in which to write the result
    int result_idx = base_idx + warp_offset;

    // Bounds check (should never be out)
    if (result_idx < total_pairs) {
      results[result_idx].img1_idx = i;
      results[result_idx].img2_idx = j;
      results[result_idx].distance = distance;
    }
  }
}

//
/**
 * process_directory_cuda(): Process all images in directory using CUDA
 * @directory: directory
 * @results: resulting hashes vector
 *
 * Returns: the count of images processed, otherwise -1 if can't open directory,
 * -2 if can't scan directory, -3/-4 if host memory error, -5 to -10 CUDA errors
 */
int process_directory_cuda(const char *directory, ImageHash *results) {
  DIR *dir = opendir(directory);
  if (!dir) {
    printf("Cannot open directory: %s\n", directory);
    return -1; // Negative for errors
  }

  // Read all directory entries
  struct dirent **entries;
  int n_entries = scandir(directory, &entries, NULL, alphasort);
  if (n_entries < 0) {
    printf("Failed to scan directory: %s\n", directory);
    closedir(dir);
    return -2;
  }

  // Count actual images
  int actual_count = count_images(entries, n_entries);
  if (actual_count == 0) {
    printf("No images found\n");
    for (int i = 0; i < n_entries; i++)
      free(entries[i]);
    free(entries);
    closedir(dir);
    return 0;
  }

  // Allocate host memory for batch processing
  unsigned char *h_batch_data =
      (unsigned char *)malloc(actual_count * DHASH_WIDTH * DHASH_HEIGHT);
  char (*filenames)[MAX_FILENAME] =
      (char (*)[MAX_FILENAME])malloc(actual_count * MAX_FILENAME);
  // I don't remember why I padded all names

  if (!h_batch_data || !filenames) {
    printf("Failed to allocate host memory\n");
    if (h_batch_data)
      free(h_batch_data);
    if (filenames)
      free(filenames);
    for (int i = 0; i < n_entries; i++)
      free(entries[i]);
    free(entries);
    closedir(dir);
    return -3;
  }

  int count = 0;
  char filepath[MAX_PATH_LENGTH];
  printf("Scanning directory: %s\n", directory);

  // Process all image files
  for (int i = 0; i < n_entries && count < actual_count; i++) {
    // If it's an image
    if (entries[i]->d_type == DT_REG && is_image_file(entries[i]->d_name)) {
      // Write the file path
      int ret = snprintf(filepath, sizeof(filepath), "%s/%s", directory,
                         entries[i]->d_name);
      // Check that name wasn't too long
      if (ret >= sizeof(filepath)) {
        printf("Warning: Path too long for %s, skipping\n", entries[i]->d_name);
        continue;
      }
      // Process image
      printf("Loading: %s\n", entries[i]->d_name);
      if (process_image(filepath, entries[i]->d_name, h_batch_data, count,
                        filenames))
        count++;
    }
  }

  // Clean up directory entries
  for (int i = 0; i < n_entries; i++)
    free(entries[i]);
  free(entries);
  closedir(dir);

  printf("Loaded %d images, processing with CUDA...\n", count);

  // Allocate GPU memory
  unsigned char *d_batch_data;
  uint64_t *d_hashes;
  uint64_t *h_hashes = (uint64_t *)malloc(count * sizeof(uint64_t));

  if (!h_hashes) {
    printf("Failed to allocate host hash array\n");
    free(h_batch_data);
    free(filenames);
    return -4;
  }

  size_t batch_size =
      count * DHASH_WIDTH * DHASH_HEIGHT * sizeof(unsigned char);
  size_t hash_size = count * sizeof(uint64_t);

  CUDA_CHECK_MALLOC_WITH_CODE(
      d_batch_data, batch_size,
      {
        free(h_batch_data);
        free(filenames);
        free(h_hashes);
      },
      -5);
  CUDA_CHECK_MALLOC_WITH_CODE(
      d_hashes, hash_size,
      {
        cudaFree(d_batch_data);
        free(h_batch_data);
        free(filenames);
        free(h_hashes);
      },
      -6);
  CUDA_CHECK_MEMCPY_WITH_CODE(
      d_batch_data, h_batch_data, batch_size, cudaMemcpyHostToDevice,
      {
        cudaFree(d_batch_data);
        cudaFree(d_hashes);
        free(h_batch_data);
        free(filenames);
        free(h_hashes);
      },
      -7);

  // Launch CUDA kernel with shared memory
  int blocks_per_grid = (count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  size_t shared_mem_size =
      THREADS_PER_BLOCK * DHASH_WIDTH * DHASH_HEIGHT * sizeof(unsigned char);

  printf("Launching CUDA kernel with %d blocks, %d threads per block\n",
         blocks_per_grid, THREADS_PER_BLOCK);

  compute_dHash_kernel<<<blocks_per_grid, THREADS_PER_BLOCK, shared_mem_size>>>(
      d_batch_data, d_hashes, count);

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_batch_data);
    cudaFree(d_hashes);
    free(h_batch_data);
    free(filenames);
    free(h_hashes);
    return -8;
  }

  // Wait for kernel to complete
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA synchronization failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_batch_data);
    cudaFree(d_hashes);
    free(h_batch_data);
    free(filenames);
    free(h_hashes);
    return -9;
  }

  CUDA_CHECK_MEMCPY_WITH_CODE(
      h_hashes, d_hashes, hash_size, cudaMemcpyDeviceToHost,
      {
        cudaFree(d_batch_data);
        cudaFree(d_hashes);
        free(h_batch_data);
        free(filenames);
        free(h_hashes);
      },
      -10);

  // Store results
  for (int i = 0; i < count; i++) {
    strncpy(results[i].filename, filenames[i], MAX_FILENAME - 1);
    results[i].filename[MAX_FILENAME - 1] = '\0';
    results[i].hash = h_hashes[i];
  }

  // Cleanup
  cudaFree(d_batch_data);
  cudaFree(d_hashes);
  free(h_batch_data);
  free(filenames);
  free(h_hashes);

  printf("CUDA processing completed successfully!\n\n");
  return count;
}

/**
 * print_results(): Print all hash results
 * @hashes: array of hashes
 * @count: how many hashes
 */
void print_results(ImageHash *hashes, int count) {
  printf("=== dHash Results ===\n");
  for (int i = 0; i < count; i++) {
    printf("%s: %016lx\n", hashes[i].filename, hashes[i].hash);
  }
  printf("\n");
}

/**
 * find_similar_images_gpu(): Find and print similar images using GPU
 * acceleration
 * @hashes: array of hashes
 * @count: how many hashes
 * @threshold: similarity threshold
 *
 * Returns: 0 if successful, 1-4 host errors, 5-9 CUDA errors
 */
int find_similar_images_gpu(ImageHash *hashes, int count, int threshold) {
  printf("=== Similar Images (Hamming distance <= %d) ===\n", threshold);
  if (count <= 1) {
    printf("Not enough images for comparison.\n\n");
    return 1;
  }

  // Calculate total number of unique pairs
  long long total_pairs = ((long long)count * (count - 1)) / 2;
  // Check if total_pairs fits in int for results array
  if (total_pairs > INT_MAX) {
    printf("Too many image pairs (%lld) for current implementation\n",
           total_pairs);
    return 2;
  }
  // This is useful only for >2^16 images (i.e. not really useful)

  int total_pairs_int = (int)total_pairs;
  // Allocate GPU memory
  uint64_t *d_hashes;
  ComparisonResult *d_results;
  int *d_result_count;
  int h_result_count = 0;

  // Host memory for results
  ComparisonResult *h_results =
      (ComparisonResult *)malloc(total_pairs_int * sizeof(ComparisonResult));
  if (!h_results) {
    printf("Failed to allocate host memory for comparison results\n");
    return 3;
  }

  // Extract hashes from ImageHash struct to a simple array
  uint64_t *hash_array = (uint64_t *)malloc(count * sizeof(uint64_t));
  if (!hash_array) {
    printf("Failed to allocate hash array\n");
    free(h_results);
    return 4;
  }
  for (int i = 0; i < count; i++) {
    hash_array[i] = hashes[i].hash;
  }

  // Allocate and copy hashes to GPU
  CUDA_CHECK_MALLOC_WITH_CODE(
      d_hashes, count * sizeof(uint64_t),
      {
        free(h_results);
        free(hash_array);
      },
      5);
  CUDA_CHECK_MEMCPY_WITH_CODE(
      d_hashes, hash_array, count * sizeof(uint64_t), cudaMemcpyHostToDevice,
      {
        cudaFree(d_hashes);
        free(h_results);
        free(hash_array);
      },
      6);
  // Allocate GPU memory for results
  CUDA_CHECK_MALLOC_WITH_CODE(
      d_results, total_pairs_int * sizeof(ComparisonResult),
      {
        cudaFree(d_hashes);
        free(h_results);
        free(hash_array);
      },
      7);
  CUDA_CHECK_MALLOC_WITH_CODE(
      d_result_count, sizeof(int),
      {
        cudaFree(d_hashes);
        cudaFree(d_results);
        free(h_results);
        free(hash_array);
      },
      8);
  // Initialize result count to 0
  CUDA_CHECK_MEMCPY_WITH_CODE(
      d_result_count, &h_result_count, sizeof(int), cudaMemcpyHostToDevice,
      {
        cudaFree(d_hashes);
        cudaFree(d_results);
        cudaFree(d_result_count);
        free(h_results);
        free(hash_array);
      },
      9);

  // Should be below, but throws an variable initialization skip error
  cudaError_t err = cudaSuccess;

  printf("Comparing %lld image pairs on GPU...\n", total_pairs);
  // Choose kernel based on dataset size using named constants
  if (count <= SHARED_MEMORY_THRESHOLD) { // Should always be here
    int blocks = (total_pairs_int + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks > MAX_BLOCKS_SHARED) {
      blocks = MAX_BLOCKS_SHARED;
    }

    size_t shared_mem_size = count * sizeof(uint64_t);
    printf("Using shared memory kernel with %d blocks\n", blocks);
    compare_hashes_kernel_shared<<<blocks, THREADS_PER_BLOCK,
                                   shared_mem_size>>>(
        d_hashes, d_results, d_result_count, count, threshold);
  } else { // You got a lot of images boy
    long long blocks_ll =
        (total_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks_ll > INT_MAX) { // A LOOOOOOOT
      printf("Too many blocks required (%lld) for kernel launch\n", blocks_ll);
      goto cleanup;
    }
    int blocks = (int)blocks_ll;

    printf("Using regular kernel with %d blocks\n", blocks);
    compare_hashes_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_hashes, d_results, d_result_count, count, threshold);
  }

  // Check for kernel launch errors
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    goto cleanup;
  }

  // Wait for kernel to complete
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA synchronization failed: %s\n", cudaGetErrorString(err));
    goto cleanup;
  }

  // Copy result count back to host
  CUDA_CHECK_MEMCPY(&h_result_count, d_result_count, sizeof(int),
                    cudaMemcpyDeviceToHost, { goto cleanup; });

  printf("Found %d similar image pairs\n", h_result_count);

  if (h_result_count > 0) {
    // Copy results back to host
    CUDA_CHECK_MEMCPY(h_results, d_results,
                      h_result_count * sizeof(ComparisonResult),
                      cudaMemcpyDeviceToHost, { goto cleanup; });

    // Print results
    for (int i = 0; i < h_result_count; i++) {
      printf("%s <-> %s (distance: %d)\n",
             hashes[h_results[i].img1_idx].filename,
             hashes[h_results[i].img2_idx].filename, h_results[i].distance);
    }
  } else {
    printf("No similar images found.\n");
  }

cleanup:
  // Cleanup GPU memory
  cudaFree(d_hashes);
  cudaFree(d_results);
  cudaFree(d_result_count);
  free(h_results);
  free(hash_array);
  printf("\n");

  return 0;
}

/**
 * find_similar_images_cpu(): CPU version for comparisons
 * @hashes: array of hashes
 * @count: how many hashes
 * @threshold: similarity threshold
 *
 * Returns: 0 if succesful
 */
int find_similar_images_cpu(ImageHash *hashes, int count, int threshold) {
  printf("=== Similar Images - CPU Version (Hamming distance <= %d) ===\n",
         threshold);

  int found_similar = 0;
  for (int i = 0; i < count; i++) {
    for (int j = i + 1; j < count; j++) {
      int distance = hamming_distance(hashes[i].hash, hashes[j].hash);
      if (distance <= threshold) {
        printf("%s <-> %s (distance: %d)\n", hashes[i].filename,
               hashes[j].filename, distance);
        found_similar = 1;
      }
    }
  }

  if (!found_similar) {
    printf("No similar images found.\n");
  }
  printf("\n");

  return 0;
}

int main(int argc, char *argv[]) {
  // Handle command line arguments including optional threshold parameter and
  // comparison mode
  int threshold = 10;             // Default threshold
  bool use_gpu_comparison = true; // Default to GPU comparison

  if (argc < 2 || argc > 4) {
    printf("Usage: %s <image_directory> [threshold] [--cpu]\n", argv[0]);
    printf("\nThis program calculates dHash for all images in a directory "
           "using CUDA\n");
    printf("and finds visually similar images.\n\n");
    printf("Parameters:\n");
    printf("  image_directory: Path to directory containing images\n");
    printf("  threshold:       Optional Hamming distance threshold (default: "
           "10)\n");
    printf("  --cpu:           Optional flag to use CPU for hash comparison "
           "instead of GPU\n\n");
    printf("Supported formats: JPG, JPEG, PNG, BMP\n");
    return 1;
  }

  // First parameter is directory
  const char *directory = argv[1];

  // Parse optional arguments
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--cpu") == 0) {
      use_gpu_comparison = false;
    } else {
      threshold = atoi(argv[i]);
      if (threshold < 0 || threshold > 64) {
        printf("Warning: Threshold should be between 0 and 64. Using default: "
               "10\n");
        threshold = 10;
      }
    }
  }
  printf("Comparison mode: %s\n", use_gpu_comparison ? "GPU" : "CPU");

  // Check if directory exists
  struct stat st;
  if (stat(directory, &st) != 0 || !S_ISDIR(st.st_mode)) {
    fprintf(stderr, "Error: Directory does not exist: %s\n", directory);
    return 2;
  }

  // Initialize CUDA
  int device_count;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    fprintf(stderr, "No CUDA devices found or CUDA not available: %s\n",
            cudaGetErrorString(err));
    return 3;
  }
  printf("Found %d CUDA device(s)\n", device_count);

  // Get device properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Using device: %s\n", prop.name);
  printf("Compute capability: %d.%d\n", prop.major, prop.minor);
  printf("Max threads per block: %d\n\n", prop.maxThreadsPerBlock);

  // Process all images in directory using CUDA
  // Allocate memory for hash results - use a reasonable estimate first
  ImageHash *hashes = (ImageHash *)malloc(MAX_FILES * sizeof(ImageHash));
  if (!hashes) {
    printf("Memory allocation failed!\n");
    return 4;
  }

  int actual_image_count = process_directory_cuda(directory, hashes);

  if (actual_image_count > 0) {
    print_results(hashes, actual_image_count);

    // Choose comparison method based on user preference
    if (use_gpu_comparison) {
      int gpu_result =
          find_similar_images_gpu(hashes, actual_image_count, threshold);
      if (gpu_result != 0) {
        printf("GPU comparison failed, falling back to CPU...\n");
        find_similar_images_cpu(hashes, actual_image_count, threshold);
      }
    } else {
      find_similar_images_cpu(hashes, actual_image_count, threshold);
    }
  } else if (actual_image_count == 0) {
    printf("No images found or processed.\n");
  } else {
    // Handle error cases (negative return values)
    printf("Error processing directory (code: %d)\n", actual_image_count);
    free(hashes);
    return 5;
  }

  free(hashes);
  return 0;
}
