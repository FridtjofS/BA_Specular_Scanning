
#define CHECK_CUDA(call)                                        \
    if ((call) != cudaSuccess)                                  \
    {                                                           \
        std::cerr << "CUDA error at " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                     \
    }

struct Point3D {
    float x;
    float y;
    float z;
};

struct EdgePixel {
    int x;
    int theta;
    bool gradient_sign; // true for positive slope, false for negative slop
    
};

void write_points_to_ply(const std::vector<Point3D>& points_3d, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Write PLY header
    fprintf(file, "ply\n");
    fprintf(file, "format ascii 1.0\n");
    fprintf(file, "element vertex %zu\n", points_3d.size());
    fprintf(file, "property float x\n");
    fprintf(file, "property float y\n");
    fprintf(file, "property float z\n");
    fprintf(file, "property float r\n");
    fprintf(file, "property float g\n");
    fprintf(file, "property float b\n");
    fprintf(file, "end_header\n");

    // Write vertex data
    for (const auto& point : points_3d) {

        fprintf(file, "%f %f %f %f %f %f\n", point.x / 1010, point.y / 1010, (point.z - 54.0f) / 1010);
    }

    fclose(file);
}

void print_progress(int current, int total, auto start_time, double copy_time, double edge_time, double pack_time, double hough_time, double peak_time) {
    auto now = std::chrono::high_resolution_clock::now();

    int bar_width = 50;
    
    // use milliseconds for better precision
    double elapsed_ms = std::chrono::duration<double, std::milli>(now - start_time).count();
    double elapsed_s = elapsed_ms / 1000.0;
    
    float progress = (float)current / total;
    int filled = (int)(progress * bar_width);
    
    double imgs_per_sec = (current > 0) ? (current / elapsed_s) : 0.0;
    double eta_s = (imgs_per_sec > 0) ? ((total - current) / imgs_per_sec) : 0.0;
    int eta_m = std::floor(eta_s / 60.0);
    eta_s = std::fmod(eta_s, 60.0);

    printf("\r[");
    for (int i = 0; i < bar_width; ++i)
        printf(i < filled ? "=" : (i == filled ? ">" : " "));
    printf("] %d/%d (%.1f%%) imgs/s %.5f, ETA: %ld m %ld s", current, total, progress * 100.0f, imgs_per_sec, (long)eta_m, (long)eta_s);

    // print mean times for each stage
    //if (current > 0) {
    //    printf("\nCopy: %.2f ms, Edge: %.2f ms, Pack: %.2f ms, Hough: %.2f ms, Peak: %.2f ms", 
    //        copy_time / current, edge_time / current, pack_time / current, hough_time / current, peak_time / current);
    //}

    fflush(stdout);
    
    if (current == total) printf("\n");
}

__global__ void conv2d(const float* __restrict__ in,
                             float* __restrict__ out,
                             int W, int H, int radius, float* c_weights)
{

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < W && y < H)
  {
    float sum = 0.0f;
    int count = 0;

    for (int j = -radius; j <= radius; ++j)
    {
      for (int i = -radius; i <= radius; ++i)
      {
        int neighbor_x = x + i;
        int neighbor_y = y + j;

        if (neighbor_x >= 0 && neighbor_x < W && neighbor_y >= 0 && neighbor_y < H)
        {
          sum += in[neighbor_y * W + neighbor_x] * c_weights[(j + radius) * (2 * radius + 1) + (i + radius)];
          count++;
        }
      }
    }
    out[y * W + x] = sum;
  }
}

void launch_conv2d(const float* d_in, float* d_out, int W, int H, int radius, float* d_c_weights, cudaStream_t stream)
{

  dim3 blockSize(16, 16);
  dim3 gridSize((W + blockSize.x - 1) / blockSize.x, (H + blockSize.y - 1) / blockSize.y);
  conv2d<<<gridSize, blockSize, 0, stream>>>(d_in, d_out, W, H, radius, d_c_weights);
}

__global__ void conv_sobel(const float* __restrict__ in,
                             float* __restrict__ mag,
                             float* __restrict__ dir,
                             int W, int H, float* c_weights_x, float* c_weights_y)
{

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < W && y < H)
  {
    float sum_x = 0.0f;
    float sum_y = 0.0f;

    for (int j = -1; j <= 1; ++j)
    {
      for (int i = -1; i <= 1; ++i)
      {
        int neighbor_x = x + i;
        int neighbor_y = y + j;

        if (neighbor_x >= 0 && neighbor_x < W && neighbor_y >= 0 && neighbor_y < H)
        {
          sum_x += in[neighbor_y * W + neighbor_x] * c_weights_x[(j + 1) * 3 + (i + 1)];
          sum_y += in[neighbor_y * W + neighbor_x] * c_weights_y[(j + 1) * 3 + (i + 1)];
        }
      }
    }
    mag[y * W + x] = sqrtf(sum_x * sum_x + sum_y * sum_y);
    dir[y * W + x] = atan2f(sum_y, sum_x);
  }
}

void launch_conv_sobel(const float* d_in, float* d_mag, float* d_dir, int W, int H, float* d_c_weights_x, float* d_c_weights_y, cudaStream_t stream)
{

  dim3 blockSize(16, 16);
  dim3 gridSize((W + blockSize.x - 1) / blockSize.x, (H + blockSize.y - 1) / blockSize.y);
  conv_sobel<<<gridSize, blockSize, 0, stream>>>(d_in, d_mag, d_dir, W, H, d_c_weights_x, d_c_weights_y);
}




__global__ void nms(const float* __restrict__ mag,
                             const float* __restrict__ dir,
                             float* __restrict__ out,
                             int W, int H)
{

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= W || y >= H) return;

  float angle = dir[y * W + x];
  float magnitude = mag[y * W + x];

  float deg = angle * 180.0f / M_PI;
  if (deg < 0) deg += 180.0f;

  auto fetch = [&](int nx, int ny) -> float {
        if (nx < 0 || nx >= W || ny < 0 || ny >= H) return 0.f;
        return mag[ny * W + nx];
  };

  float neighbor1 = 0.0f;
  float neighbor2 = 0.0f;
  if ((deg >= 0 && deg < 22.5) || (deg >= 157.5 && deg <= 180))
  {
    neighbor1 = fetch(x + 1, y);
    neighbor2 = fetch(x - 1, y);
  }
  else if (deg >= 22.5 && deg < 67.5)
  {
    neighbor1 = fetch(x + 1, y - 1);
    neighbor2 = fetch(x - 1, y + 1);
  }
  else if (deg >= 67.5 && deg < 112.5)
  {
    neighbor1 = fetch(x, y - 1);
    neighbor2 = fetch(x, y + 1);
  }
  else if (deg >= 112.5 && deg < 157.5)
  {
    neighbor1 = fetch(x - 1, y - 1);
    neighbor2 = fetch(x + 1, y + 1);
  }
  out[y * W + x] = (magnitude >= neighbor1 && magnitude >= neighbor2) ? magnitude : 0.0f;
  
}

void launch_non_maximum_suppression(const float* d_mag, const float* d_dir, float* d_out, int W, int H, cudaStream_t stream)
{

  dim3 blockSize(16, 16);
  dim3 gridSize((W + blockSize.x - 1) / blockSize.x, (H + blockSize.y - 1) / blockSize.y);
  nms<<<gridSize, blockSize, 0, stream>>>(d_mag, d_dir, d_out, W, H);
}


__global__ void hysteresis_threshold(const float* __restrict__ in,
                             float* __restrict__ out,
                             int W, int H, float low_thresh, float high_thresh)
{

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < W && y < H)
  {
    if (in[y * W + x] >= high_thresh) {
      out[y * W + x] = 1.0f;  // strong edge
      return;
    }
    if (in[y * W + x] < low_thresh) {
      out[y * W + x] = 0.0f;  // non-edge
      return;
    }

    for (int j = -1; j <= 1; ++j)
    {
      for (int i = -1; i <= 1; ++i)
      {
        int neighbor_x = x + i;
        int neighbor_y = y + j;

        if (neighbor_x >= 0 && neighbor_x < W && neighbor_y >= 0 && neighbor_y < H)
        {
          if (in[neighbor_y * W + neighbor_x] >= high_thresh) {
            out[y * W + x] = 1.0f;  // weak edge connected to strong edge
            return;
          }
        }
      }
    }
  }
}

void launch_hysteresis_threshold(const float* d_in, float* d_out, int W, int H, float low_thresh, float high_thresh, cudaStream_t stream)
{

  dim3 blockSize(16, 16);
  dim3 gridSize((W + blockSize.x - 1) / blockSize.x, (H + blockSize.y - 1) / blockSize.y);
  hysteresis_threshold<<<gridSize, blockSize, 0, stream>>>(d_in, d_out, W, H, low_thresh, high_thresh);
}





void canny_edge(const float* in, float* out, int W, int H, float low_thresh, float high_thresh, cudaStream_t stream = 0)
{

    float* d_in;
    float* d_blurred;
    cudaMalloc(&d_in, W * H * sizeof(float));
    cudaMalloc(&d_blurred, W * H * sizeof(float));
    cudaMemcpy(d_in, in, W * H * sizeof(float), cudaMemcpyHostToDevice);

    // 1. Gaussian blur
    float* blurred = new float[W * H];
    float c_weights[25] = {
        2.0/159.0, 4.0/159.0, 5.0/159.0, 4.0/159.0, 2.0/159.0,
        4.0/159.0, 9.0/159.0, 12.0/159.0, 9.0/159.0, 4.0/159.0,
        5.0/159.0, 12.0/159.0, 15.0/159.0, 12.0/159.0, 5.0/159.0,
        4.0/159.0, 9.0/159.0, 12.0/159.0, 9.0/159.0, 4.0/159.0,
        2.0/159.0, 4.0/159.0, 5.0/159.0, 4.0/159.0, 2.0/159.0
    };
    float* d_c_weights_gaussian;
    cudaMalloc(&d_c_weights_gaussian, 25 * sizeof(float));
    cudaMemcpy(d_c_weights_gaussian, c_weights, 25 * sizeof(float), cudaMemcpyHostToDevice);


    launch_conv2d(d_in, d_blurred, W, H, 2, d_c_weights_gaussian, stream);

    cudaFree(d_in);
    cudaFree(d_c_weights_gaussian);

    // 2. Sobel filter
    float* d_sobel_mag;
    float* d_sobel_dir;
    cudaMalloc(&d_sobel_mag, W * H * sizeof(float));
    cudaMalloc(&d_sobel_dir, W * H * sizeof(float));


    float c_weights_sobel_x[9] = {-1, 0, 1,
                                 -2, 0, 2,
                                 -1, 0, 1};
    float c_weights_sobel_y[9] = {1, 2, 1,
                                 0, 0, 0,
                                 -1, -2, -1};
    float* d_c_weights_sobel_x;
    float* d_c_weights_sobel_y;
    cudaMalloc(&d_c_weights_sobel_x, 9 * sizeof(float));
    cudaMalloc(&d_c_weights_sobel_y, 9 * sizeof(float));
    cudaMemcpy(d_c_weights_sobel_x, c_weights_sobel_x, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_weights_sobel_y, c_weights_sobel_y, 9 * sizeof(float), cudaMemcpyHostToDevice);


    launch_conv_sobel(d_blurred, d_sobel_mag, d_sobel_dir, W, H, d_c_weights_sobel_x, d_c_weights_sobel_y, stream);

    cudaFree(d_blurred);
    cudaFree(d_c_weights_sobel_x);
    cudaFree(d_c_weights_sobel_y);

    // 3. Non-maximum suppression
    float* d_nms;
    cudaMalloc(&d_nms, W * H * sizeof(float));

    launch_non_maximum_suppression(d_sobel_mag, d_sobel_dir, d_nms, W, H, stream);
    cudaFree(d_sobel_mag);
    cudaFree(d_sobel_dir);

    // 4. Hysteresis thresholding
    float* d_hysteresis;
    cudaMalloc(&d_hysteresis, W * H * sizeof(float));
    launch_hysteresis_threshold(d_nms, d_hysteresis, W, H, low_thresh, high_thresh, stream);
    cudaFree(d_nms);
    cudaMemcpy(out, d_hysteresis, W * H * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_hysteresis);


    
}

__global__ void conv_sobel_grad(const float* __restrict__ in,
                             float* __restrict__ grad,
                             int W, int H, float* c_weights_x, float* c_weights_y)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 2 && x < W-2 && y >= 2 && y < H-2) // radius=2, skip border
    {
        float sum_x = 0.0f;
        float sum_y = 0.0f;

        for (int j = -2; j <= 2; ++j)
        {
            for (int i = -2; i <= 2; ++i)
            {
                float px = in[(y+j) * W + (x+i)];
                sum_x += px * c_weights_x[(j + 2) * 5 + (i + 2)]; // 5x5 indexing
                sum_y += px * c_weights_y[(j + 2) * 5 + (i + 2)];
            }
        }

        // normalize by kernel sum to make scale-independent
        grad[y * W + x] = (sum_x * sum_y) / (96.0f * 96.0f); // max possible value per channel
    } else if (x < W && y < H) {
        grad[y * W + x] = 0.0f; // zero border explicitly
    }
}

void launch_conv_sobel_grad(const float* d_in, float* d_grad, int W, int H, float* d_c_weights_x, float* d_c_weights_y, cudaStream_t stream)
{

  dim3 blockSize(16, 16);
  dim3 gridSize((W + blockSize.x - 1) / blockSize.x, (H + blockSize.y - 1) / blockSize.y);
  conv_sobel_grad<<<gridSize, blockSize, 0, stream>>>(d_in, d_grad, W, H, d_c_weights_x, d_c_weights_y);
}

void sobel_grad(const float* in, float* out, int W, int H, cudaStream_t stream = 0) {
    float c_weights_sobel_x[25] = {
      -1,  -4,  0,  4,  1,
      -4, -16,  0, 16,  4,
      -6, -24,  0, 24,  6,
      -4, -16,  0, 16,  4,
      -1,  -4,  0,  4,  1
    };

    float c_weights_sobel_y[25] = {
      -1,  -4,  -6,  -4, -1,
      -4, -16, -24, -16, -4,
       0,   0,   0,   0,  0,
       4,  16,  24,  16,  4,
       1,   4,   6,   4,  1
    };
    float* d_in;
    float* d_out;
    float* d_c_weights_sobel_x;
    float* d_c_weights_sobel_y;
    cudaMalloc(&d_in, W * H * sizeof(float));
    cudaMalloc(&d_out, W * H * sizeof(float));
    cudaMalloc(&d_c_weights_sobel_x, 25 * sizeof(float));
    cudaMalloc(&d_c_weights_sobel_y, 25 * sizeof(float));
    cudaMemcpy(d_in, in, W * H * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_weights_sobel_x, c_weights_sobel_x, 25 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_weights_sobel_y, c_weights_sobel_y, 25 * sizeof(float), cudaMemcpyHostToDevice);


    launch_conv_sobel_grad(d_in, d_out, W, H, d_c_weights_sobel_x, d_c_weights_sobel_y, stream);

    cudaMemcpy(out, d_out, W * H * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_c_weights_sobel_x);
    cudaFree(d_c_weights_sobel_y);
}

// Device-pointer variants for use inside multi-stream pipelines: operate entirely on
// caller-supplied device buffers (no internal cudaMalloc/cudaFree/cudaMemcpy per call),
// so they don't force a whole-device sync on every invocation. Filter weights are
// constant, so they're uploaded once (lazily) instead of once per image.

void canny_edge_device(const float* d_in, float* d_out,
                        float* d_blurred, float* d_sobel_mag, float* d_sobel_dir, float* d_nms,
                        int W, int H, float low_thresh, float high_thresh, cudaStream_t stream)
{
    static float* d_c_weights_gaussian = nullptr;
    static float* d_c_weights_sobel3_x = nullptr;
    static float* d_c_weights_sobel3_y = nullptr;
    if (!d_c_weights_gaussian) {
        float c_weights[25] = {
            2.0/159.0, 4.0/159.0, 5.0/159.0, 4.0/159.0, 2.0/159.0,
            4.0/159.0, 9.0/159.0, 12.0/159.0, 9.0/159.0, 4.0/159.0,
            5.0/159.0, 12.0/159.0, 15.0/159.0, 12.0/159.0, 5.0/159.0,
            4.0/159.0, 9.0/159.0, 12.0/159.0, 9.0/159.0, 4.0/159.0,
            2.0/159.0, 4.0/159.0, 5.0/159.0, 4.0/159.0, 2.0/159.0
        };
        float c_weights_sobel_x[9] = {-1, 0, 1,
                                     -2, 0, 2,
                                     -1, 0, 1};
        float c_weights_sobel_y[9] = {1, 2, 1,
                                     0, 0, 0,
                                     -1, -2, -1};
        cudaMalloc(&d_c_weights_gaussian, 25 * sizeof(float));
        cudaMalloc(&d_c_weights_sobel3_x, 9 * sizeof(float));
        cudaMalloc(&d_c_weights_sobel3_y, 9 * sizeof(float));
        cudaMemcpy(d_c_weights_gaussian, c_weights, 25 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_c_weights_sobel3_x, c_weights_sobel_x, 9 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_c_weights_sobel3_y, c_weights_sobel_y, 9 * sizeof(float), cudaMemcpyHostToDevice);
    }

    launch_conv2d(d_in, d_blurred, W, H, 2, d_c_weights_gaussian, stream);
    launch_conv_sobel(d_blurred, d_sobel_mag, d_sobel_dir, W, H, d_c_weights_sobel3_x, d_c_weights_sobel3_y, stream);
    launch_non_maximum_suppression(d_sobel_mag, d_sobel_dir, d_nms, W, H, stream);
    launch_hysteresis_threshold(d_nms, d_out, W, H, low_thresh, high_thresh, stream);
}

void sobel_grad_device(const float* d_in, float* d_out, int W, int H, cudaStream_t stream)
{
    static float* d_c_weights_sobel5_x = nullptr;
    static float* d_c_weights_sobel5_y = nullptr;
    if (!d_c_weights_sobel5_x) {
        float c_weights_sobel_x[25] = {
          -1,  -4,  0,  4,  1,
          -4, -16,  0, 16,  4,
          -6, -24,  0, 24,  6,
          -4, -16,  0, 16,  4,
          -1,  -4,  0,  4,  1
        };
        float c_weights_sobel_y[25] = {
          -1,  -4,  -6,  -4, -1,
          -4, -16, -24, -16, -4,
           0,   0,   0,   0,  0,
           4,  16,  24,  16,  4,
           1,   4,   6,   4,  1
        };
        cudaMalloc(&d_c_weights_sobel5_x, 25 * sizeof(float));
        cudaMalloc(&d_c_weights_sobel5_y, 25 * sizeof(float));
        cudaMemcpy(d_c_weights_sobel5_x, c_weights_sobel_x, 25 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_c_weights_sobel5_y, c_weights_sobel_y, 25 * sizeof(float), cudaMemcpyHostToDevice);
    }

    launch_conv_sobel_grad(d_in, d_out, W, H, d_c_weights_sobel5_x, d_c_weights_sobel5_y, stream);
}

__global__ void real_to_complex(const float* __restrict__ in,
                                 cufftComplex* __restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[idx].x = in[idx];
    out[idx].y = 0.0f;
}

__global__ void fftshift_2d(cufftComplex* __restrict__ data, int rows, int cols) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= cols || v >= rows) return;

    // flip sign of elements where (u+v) is odd — equivalent to fftshift
    if ((u + v) % 2 == 1) {
        data[v * cols + u].x = -data[v * cols + u].x;
        data[v * cols + u].y = -data[v * cols + u].y;
    }
}

__global__ void apply_lowpass_mask(cufftComplex* __restrict__ freq,
                                    int rows, int cols, const float* __restrict__ radius) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= cols || v >= rows) return;

    float r = *radius;

    float du = u - cols / 2.0f;
    float dv = v - rows / 2.0f;
    float dist2 = du * du + dv * dv;

    if (dist2 > r * r) {
        freq[v * cols + u].x = 0.0f;
        freq[v * cols + u].y = 0.0f;
    }
}

__global__ void subtract_and_clip(float* __restrict__ original,
                                   const cufftComplex* __restrict__ lowpass,
                                   int N, float norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float lp = lowpass[idx].x / norm; // take real part, normalize
    float val = original[idx] - lp;
    original[idx] = val < 0.0f ? 0.0f : val; // clip to 0
}


