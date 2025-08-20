# CUDA dHash Calculator

A high-performance **image similarity detector** using **perceptual hashing (dHash)**, accelerated with **CUDA**.  
This tool scans a directory of images, computes their difference-hash fingerprints on the GPU, and compares them to find duplicates or visually similar files.

---

## Features
- Computes **dHash** (difference hash) in parallel using CUDA.
- Supports common image formats: **JPG, JPEG, PNG, BMP**.
- Resizes images to **9×8 grayscale** for standardized hashing.
- Finds visually similar images using **Hamming distance**.
- Optionally runs comparisons on **CPU or GPU**.
- Handles up to **4096 images per run** by default.

---

## Requirements
- **CUDA-capable GPU** (Compute Capability ≥ 3.0 recommended)
- **CUDA Toolkit** installed
- **C compiler** (`gcc`/`g++`)
- [stb_image.h](https://github.com/nothings/stb) for image loading (the Makefile takes care of it)

---

## Build

Just `make` it.

---

## Usage

Run the program on a directory of images:
```
./dhash_calc_cuda <directory> [options]
````

**Options**
* `--cpu`
Use CPU for image comparison (default is GPU).

* `--threshold <N>`
Set maximum Hamming distance (N) for considering two images as similar. Default 10.

**Example**
    ./dhash_calc_cuda ./images 10 --cpu

---

## How it Works

1. Each image is resized to 9×8 grayscale.
2. The dHash algorithm compares adjacent pixels horizontally to produce a 64-bit fingerprint.
3. Hashes are compared using Hamming distance:
  * Distance = number of differing bits.
  * Smaller distance = more similar images.
4. Results are printed, showing pairs of similar images.
