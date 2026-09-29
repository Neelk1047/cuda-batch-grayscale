# CUDA Batch Image Grayscale Processing

A CUDA C++ project that processes a large batch of RGB images on the GPU and converts each pixel to grayscale.

## Project objective

The project demonstrates GPU-based image processing at scale. It generates/processes a batch of 150 RGB images and uses a custom CUDA kernel where one GPU thread handles one pixel.

## CUDA implementation

The pipeline is:

1. Read RGB PPM images on the host.
2. Allocate RGB and grayscale buffers with `cudaMalloc`.
3. Copy RGB image data from host to device with `cudaMemcpy`.
4. Launch a 2D CUDA grid with 16x16 blocks.
5. Execute `rgbToGrayscale`, mapping one CUDA thread to one pixel.
6. Copy grayscale pixels back with `cudaMemcpy`.
7. Save the result as a PGM image.
8. Repeat for the complete batch.

Boundary checks prevent threads outside the image dimensions from accessing memory.

## Dataset

The included dataset generator creates 150 deterministic RGB PPM images. This avoids requiring third-party image libraries while still providing a reproducible batch workload.

Generate the dataset:

```bash
python3 scripts/generate_dataset.py --count 150
```

For a larger image workload:

```bash
python3 scripts/generate_dataset.py --count 150 --width 1024 --height 768
```

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit / `nvcc`
- C++17 compiler support through `nvcc`
- Python 3 for dataset generation

Check CUDA:

```bash
nvcc --version
nvidia-smi
```

## Build

Linux/WSL:

```bash
make
```

Or directly:

```bash
nvcc -O2 -std=c++17 src/batch_grayscale.cu -o batch_grayscale
```

Windows with `nvcc`:

```powershell
nvcc -O2 -std=c++17 src/batch_grayscale.cu -o batch_grayscale.exe
```

## Run

Linux/WSL:

```bash
./batch_grayscale --input images/input --output images/output --limit 150
```

Windows:

```powershell
.\batch_grayscale.exe --input images/input --output images/output --limit 150
```

The program prints the detected GPU, number of images, successful/failed images, and measured GPU kernel time.

## Command-line arguments

- `--input <folder>`: input PPM directory
- `--output <folder>`: output PGM directory
- `--limit <N>`: maximum number of images to process
- `--help`: display usage

## Output

For an input file such as:

```text
image_0000.ppm
```

the output is:

```text
image_0000_gray.pgm
```

## Proof of execution

After running on a CUDA-capable GPU, save the terminal output to:

```text
results/execution_log.txt
```

and keep representative input/output images in the repository.

Do not place fabricated GPU names or timing values in the execution log. The values in the final submission should come from an actual run.

## Repository structure

```text
cuda-batch-grayscale/
├── README.md
├── Makefile
├── src/
│   └── batch_grayscale.cu
├── scripts/
│   └── generate_dataset.py
├── images/
│   ├── input/
│   └── output/
└── results/
    └── execution_log.txt
```

## Academic note

This project is intended as an independent CUDA image-processing demonstration. The final execution evidence should be produced by running the project in the student's CUDA environment.
