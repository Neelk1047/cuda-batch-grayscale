#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "  \
                << cudaGetErrorString(err__) << std::endl;                  \
      return 1;                                                              \
    }                                                                        \
  } while (0)

// Simple PPM (P6) image reader/writer.
// This keeps the project self-contained and avoids third-party dependencies.
struct Image {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> rgb;
};

static bool ReadToken(std::ifstream& in, std::string& token) {
  token.clear();
  char c;
  while (in.get(c)) {
    if (c == '#') {
      std::string ignored;
      std::getline(in, ignored);
      continue;
    }
    if (!std::isspace(static_cast<unsigned char>(c))) {
      token.push_back(c);
      break;
    }
  }
  if (token.empty()) return false;
  while (in.get(c)) {
    if (std::isspace(static_cast<unsigned char>(c))) break;
    token.push_back(c);
  }
  return true;
}

static bool ReadPPM(const fs::path& path, Image& image) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;

  std::string magic, sw, sh, maxv;
  if (!ReadToken(in, magic) || magic != "P6") return false;
  if (!ReadToken(in, sw) || !ReadToken(in, sh) || !ReadToken(in, maxv)) return false;

  image.width = std::stoi(sw);
  image.height = std::stoi(sh);
  int max_value = std::stoi(maxv);
  if (image.width <= 0 || image.height <= 0 || max_value != 255) return false;

  image.rgb.resize(static_cast<size_t>(image.width) * image.height * 3);
  in.read(reinterpret_cast<char*>(image.rgb.data()), image.rgb.size());
  return in.gcount() == static_cast<std::streamsize>(image.rgb.size());
}

static bool WritePGM(const fs::path& path, int width, int height,
                     const std::vector<uint8_t>& gray) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  out << "P5\n" << width << " " << height << "\n255\n";
  out.write(reinterpret_cast<const char*>(gray.data()), gray.size());
  return static_cast<bool>(out);
}

// One CUDA thread processes one RGB pixel.
__global__ void rgbToGrayscale(const uint8_t* rgb, uint8_t* gray,
                               int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  const size_t pixel = static_cast<size_t>(y) * width + x;
  const size_t rgb_index = pixel * 3;

  // ITU-R BT.601 luma approximation.
  const float value = 0.299f * rgb[rgb_index] +
                      0.587f * rgb[rgb_index + 1] +
                      0.114f * rgb[rgb_index + 2];
  gray[pixel] = static_cast<uint8_t>(value);
}

static void PrintUsage(const char* program) {
  std::cout << "Usage: " << program
            << " --input <folder> --output <folder> [--limit N]\n";
  std::cout << "Example: " << program
            << " --input images/input --output images/output --limit 150\n";
}

int main(int argc, char** argv) {
  fs::path input_dir = "images/input";
  fs::path output_dir = "images/output";
  int limit = 150;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      input_dir = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      output_dir = argv[++i];
    } else if (arg == "--limit" && i + 1 < argc) {
      limit = std::stoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      PrintUsage(argv[0]);
      return 1;
    }
  }

  if (limit <= 0) {
    std::cerr << "--limit must be positive.\n";
    return 1;
  }

  fs::create_directories(output_dir);

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cerr << "No CUDA-capable GPU detected.\n";
    return 1;
  }

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  CUDA_CHECK(cudaSetDevice(0));

  std::cout << "CUDA Batch Image Grayscale Processing\n";
  std::cout << "GPU: " << prop.name << "\n";
  std::cout << "Input: " << input_dir << "\n";
  std::cout << "Output: " << output_dir << "\n";
  std::cout << "Requested limit: " << limit << "\n";

  std::vector<fs::path> inputs;
  for (const auto& entry : fs::directory_iterator(input_dir)) {
    if (!entry.is_regular_file()) continue;
    if (entry.path().extension() == ".ppm" ||
        entry.path().extension() == ".PPM") {
      inputs.push_back(entry.path());
    }
  }
  std::sort(inputs.begin(), inputs.end());

  if (inputs.empty()) {
    std::cerr << "No PPM images found in " << input_dir << ".\n";
    std::cerr << "Run: python3 scripts/generate_dataset.py --count 150\n";
    return 1;
  }

  if (static_cast<int>(inputs.size()) > limit) {
    inputs.resize(limit);
  }

  size_t processed = 0;
  size_t failed = 0;
  double total_ms = 0.0;

  for (const auto& input : inputs) {
    Image image;
    if (!ReadPPM(input, image)) {
      std::cerr << "Skipping invalid image: " << input << "\n";
      ++failed;
      continue;
    }

    const size_t pixels = static_cast<size_t>(image.width) * image.height;
    const size_t rgb_bytes = pixels * 3;
    const size_t gray_bytes = pixels;

    uint8_t* d_rgb = nullptr;
    uint8_t* d_gray = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rgb, rgb_bytes));
    CUDA_CHECK(cudaMalloc(&d_gray, gray_bytes));
    CUDA_CHECK(cudaMemcpy(d_rgb, image.rgb.data(), rgb_bytes,
                          cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((image.width + block.x - 1) / block.x,
              (image.height + block.y - 1) / block.y);

    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    rgbToGrayscale<<<grid, block>>>(d_rgb, d_gray, image.width, image.height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    total_ms += elapsed_ms;

    std::vector<uint8_t> gray(gray_bytes);
    CUDA_CHECK(cudaMemcpy(gray.data(), d_gray, gray_bytes,
                          cudaMemcpyDeviceToHost));

    fs::path output = output_dir / (input.stem().string() + "_gray.pgm");
    if (!WritePGM(output, image.width, image.height, gray)) {
      std::cerr << "Could not write: " << output << "\n";
      ++failed;
    } else {
      ++processed;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_rgb));
    CUDA_CHECK(cudaFree(d_gray));
  }

  std::cout << "Images discovered: " << inputs.size() << "\n";
  std::cout << "Images processed successfully: " << processed << "\n";
  std::cout << "Images failed: " << failed << "\n";
  std::cout << "Total GPU kernel time (ms): " << total_ms << "\n";
  if (processed > 0) {
    std::cout << "Average GPU kernel time per image (ms): "
              << (total_ms / processed) << "\n";
  }
  std::cout << "Processing complete.\n";

  return failed == 0 ? 0 : 1;
}
