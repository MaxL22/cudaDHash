# Makefile for CUDA dHash Calculator (Pure C)

# Compiler and flags
NVCC = nvcc
BASE_FLAGS = -O2
LIBS = -lm

# Target executable
TARGET = dhash_calc_cuda
SOURCE = dhash_calculator.cu

# Auto-detect GPU compute capabilities
# This creates a temporary CUDA program to query GPU properties
define DETECT_GPU_COMPUTE
#include <stdio.h>
#include <cuda_runtime.h>
int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("5.0"); // Default fallback
        return 0;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Get first GPU
    printf("%d.%d", prop.major, prop.minor);
    return 0;
}
endef

export DETECT_GPU_COMPUTE

# Get compute capability
COMPUTE_CAPABILITY := $(shell echo "$$DETECT_GPU_COMPUTE" > /tmp/detect_gpu.cu && \
                      $(NVCC) -o /tmp/detect_gpu /tmp/detect_gpu.cu 2>/dev/null && \
                      /tmp/detect_gpu 2>/dev/null || echo "5.0")

# Clean up temporary files
$(shell rm -f /tmp/detect_gpu.cu /tmp/detect_gpu 2>/dev/null)

# Set architecture flags based on detected capability
ARCH_FLAGS = -arch=sm_$(subst .,,$(COMPUTE_CAPABILITY))

# Full CUDA flags
CUDA_FLAGS = $(BASE_FLAGS) $(ARCH_FLAGS)

# Default target
all: show-compute $(TARGET)

# Show detected compute capability
show-compute:
	@echo "Detected GPU Compute Capability: $(COMPUTE_CAPABILITY)"
	@echo "Using architecture flags: $(ARCH_FLAGS)"

# Build the executable
$(TARGET): $(SOURCE) stb_image.h
	@echo "Building with compute capability $(COMPUTE_CAPABILITY)..."
	$(NVCC) $(CUDA_FLAGS) -o $(TARGET) $(SOURCE) $(LIBS)

# Download stb_image.h if it doesn't exist
stb_image.h:
	@echo "Downloading stb_image.h..."
	curl -o stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h

# Manual override for compute capability
set-compute:
	@echo "Current compute capability: $(COMPUTE_CAPABILITY)"
	@echo "To manually set compute capability, use:"
	@echo "  make COMPUTE_CAPABILITY=7.5 all"
	@echo "  make COMPUTE_CAPABILITY=8.6 all"
	@echo "Available compute capabilities: 5.0, 5.2, 6.0, 6.1, 7.0, 7.5, 8.0, 8.6, 8.9, 9.0"

# Force redetection of compute capability
redetect:
	@echo "Forcing redetection of GPU compute capability..."
	@$(MAKE) clean
	@$(MAKE) all

# Check CUDA installation and show GPU info
cuda-check:
	@echo "Checking CUDA installation..."
	@nvcc --version || { echo "CUDA not found! Please install CUDA toolkit."; exit 1; }
	@nvidia-smi || { echo "NVIDIA driver not found!"; exit 1; }
	@echo ""
	@echo "GPU Information:"
	@nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "Could not query GPU details"

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Clean everything including downloaded headers
clean-all: clean
	rm -f stb_image.h

# Debug build with more verbose output
debug: CUDA_FLAGS += -g -G --ptxas-options=-v
debug: show-compute $(TARGET)

# Profile build for performance analysis
profile: CUDA_FLAGS += -lineinfo
profile: show-compute $(TARGET)

# Build for maximum compatibility (multiple architectures)
universal:
	@echo "Building universal binary for multiple GPU architectures..."
	$(NVCC) $(BASE_FLAGS) \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_70,code=sm_70 \
		-gencode arch=compute_75,code=sm_75 \
		-gencode arch=compute_80,code=sm_80 \
		-gencode arch=compute_86,code=sm_86 \
		-gencode arch=compute_89,code=sm_89 \
		-gencode arch=compute_90,code=sm_90 \
		-o $(TARGET) $(SOURCE) $(LIBS)

# Detailed GPU info
gpu-info:
	@echo "=== Detailed GPU Information ==="
	@nvidia-smi -q -d MEMORY,UTILIZATION,TEMPERATURE,COMPUTE 2>/dev/null || echo "nvidia-smi not available"
	@echo ""
	@echo "=== CUDA Device Properties ==="
	@echo "$$DETECT_GPU_COMPUTE" > /tmp/gpu_info.cu
	@echo 'printf("Compute Capability: %d.%d\\n", prop.major, prop.minor);' >> /tmp/gpu_info.cu
	@echo 'printf("Multiprocessors: %d\\n", prop.multiProcessorCount);' >> /tmp/gpu_info.cu
	@echo 'printf("Max Threads per Block: %d\\n", prop.maxThreadsPerBlock);' >> /tmp/gpu_info.cu
	@echo 'printf("Shared Memory per Block: %zu bytes\\n", prop.sharedMemPerBlock);' >> /tmp/gpu_info.cu
	@echo 'return 0; }' >> /tmp/gpu_info.cu
	@sed -i '6i\    printf("GPU Name: %s\\n", prop.name);' /tmp/gpu_info.cu
	@$(NVCC) -o /tmp/gpu_info /tmp/gpu_info.cu 2>/dev/null && /tmp/gpu_info 2>/dev/null || echo "Could not compile GPU info program"
	@rm -f /tmp/gpu_info.cu /tmp/gpu_info 2>/dev/null

# Help
help:
	@echo "Available targets:"
	@echo "  all         - Build the dHash calculator with auto-detected compute capability"
	@echo "  show-compute- Show detected compute capability"
	@echo "  set-compute - Show manual compute capability override options"
	@echo "  redetect    - Force redetection of GPU compute capability"
	@echo "  universal   - Build for multiple GPU architectures (broader compatibility)"
	@echo "  clean       - Remove build artifacts"
	@echo "  clean-all   - Remove build artifacts and downloaded files"
	@echo "  debug       - Build with debug information"
	@echo "  profile     - Build with profiling information"
	@echo "  gpu-info    - Show detailed GPU information"
	@echo "  cuda-check  - Check CUDA installation"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Manual override example:"
	@echo "  make COMPUTE_CAPABILITY=8.6 all"
	@echo ""
	@echo "Requirements:"
	@echo "  - CUDA Toolkit (nvcc compiler)"
	@echo "  - NVIDIA GPU with compute capability 2.0+"
	@echo "  - NVIDIA drivers"

.PHONY: all show-compute set-compute redetect universal clean clean-all debug profile gpu-info cuda-check help
