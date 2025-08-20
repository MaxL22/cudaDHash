# Makefile for CUDA dHash Calculator (Pure C)

# Compiler and flags
NVCC = nvcc
CUDA_FLAGS = -O2 
LIBS = -lm

# Target executable
TARGET = dhash_calc_cuda
SOURCE = dhash_calculator.cu

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SOURCE) stb_image.h
	$(NVCC) $(CUDA_FLAGS) -o $(TARGET) $(SOURCE) $(LIBS)

# Download stb_image.h if it doesn't exist
stb_image.h:
	@echo "Downloading stb_image.h..."
	curl -o stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h

# Check CUDA installation
cuda-check:
	@echo "Checking CUDA installation..."
	@nvcc --version || { echo "CUDA not found! Please install CUDA toolkit."; exit 1; }
	@nvidia-smi || { echo "NVIDIA driver not found!"; exit 1; }

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Clean everything including downloaded headers
clean-all: clean
	rm -f stb_image.h

# Debug build with more verbose output
debug: CUDA_FLAGS += -g -G --ptxas-options=-v
debug: $(TARGET)

# Profile build for performance analysis
profile: CUDA_FLAGS += -lineinfo
profile: $(TARGET)

# Info about GPU
gpu-info:
	@echo "GPU Information:"
	@nvidia-smi -q -d MEMORY,UTILIZATION,TEMPERATURE || echo "nvidia-smi not available"

# Help
help:
	@echo "Available targets:"
	@echo "  all       - Build the dHash calculator with CUDA"
	@echo "  clean     - Remove build artifacts"
	@echo "  clean-all - Remove build artifacts and downloaded files"
	@echo "  debug     - Build with debug information"
	@echo "  profile   - Build with profiling information"
	@echo "  gpu-info  - Show GPU information"
	@echo "  cuda-check- Check CUDA installation"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - CUDA Toolkit (nvcc compiler)"
	@echo "  - NVIDIA GPU with compute capability 2.0+"
	@echo "  - NVIDIA drivers"

.PHONY: all clean clean-all debug profile gpu-info cuda-check help
