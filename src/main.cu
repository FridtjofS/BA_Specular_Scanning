/*
Robust Hough Transform Based 3D Reconstruction from Circular Light Fields
*/

#include <iostream>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <vector>
#include <filesystem>
namespace fs = std::filesystem;

#include <cufft.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "helper.cu"

#define MAX_PEAKS 1000 // per image
#define PEAK_THRESHOLD 0.65f // relative to global max in hough space
#define ABS_MIN_VOTES 10.0f
#define COARSE_FACTOR 4
#define BATCH_SIZE 8 // number of images to process in parallel

void extract_3d_points_from_hough_space(const float* hough_space, int num_rotations, int x_center, int z, std::vector<Point3D>& points_3d, float max_vote) {
    // find peaks in hough space
    for (int amplitude = 0; amplitude < x_center; ++amplitude) {
        for (int phi_bin = 0; phi_bin < num_rotations; ++phi_bin) {
            float vote = hough_space[amplitude * num_rotations + phi_bin];
            if (vote > PEAK_THRESHOLD * max_vote) { // threshold for peak detection
                double phi = (phi_bin / (float)(num_rotations - 1)) * 2.0 * M_PI; // convert back to radians
                double x = amplitude * std::cos(phi);
                double y = amplitude * std::sin(phi);
                points_3d.push_back({static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)});
            }
        }
    }
}

void calculate_hough_spaces(const char*path) {
    // For each edge pixel, we will calculate the corresponding (z, amplitude) values for the Hough space.
    // z = height of the circle center
    // amplitude = radius of the circle

    // get first image to determine dimensions
    int W, H, channels;
    float* img = stbi_loadf("../renders/orthographic.png", &W, &H, &channels, 1);
    // printf("Image dimensions: %d x %d\n", W, H);
    stbi_image_free(img);  // free when done

    // number of images = number of heights = number of z values in hough space
    int max_z = 1;


    int x_center = std::floor(W / 2.0f);  // maximum possible radius based on image width
    int num_rotations = H;


    
    // create empty vector for 3d points
    std::vector<Point3D> points_3d; // vector to store 3D points

    for(int z = 0; z < max_z; ++z) { // = image_id
        float* hough_space = new float[num_rotations * x_center](); // initialize to 0

        // load image for this height
        float* img = stbi_loadf("../renders/orthographic.png", &W, &H, &channels, 1);
        if (!img) {
            fprintf(stderr, "Failed to load image: %s\n", stbi_failure_reason());
            exit(EXIT_FAILURE);
        }

        // canny edge detection
        float* edge_img = new float[W * H];
        canny_edge(img, edge_img, W, H, 0.0001f, 0.0002f);

        // sobel
        float* gradient = new float[W * H];
        sobel_grad(img, gradient, W, H);



        cudaDeviceSynchronize(); // ensure edge detection is complete before proceeding

        // write to test.png for visualization
        uint8_t* h_img = new uint8_t[W * H];
        for (int i = 0; i < W * H; ++i) {
            h_img[i] = static_cast<uint8_t>(std::min(std::max(edge_img[i] * 255.0f, 0.0f), 255.0f));  // clamp to [0, 255]
        }
        stbi_write_png("edges.png", W, H, 1, h_img, W * sizeof(uint8_t));
        delete[] h_img; // free when done

        int edge_count = 0;
        float max_vote = 0.0f; // track max vote for normalization
        // go through edge pixels and calculate corresponding (amplitude, phi) values for hough space
        for (int x = 0; x < W; ++x) {
            if (x == x_center) continue; // degenerate case, x=center contributes no directional info
            for (int theta = 1; theta < num_rotations - 1; ++theta) {
                if (edge_img[theta * W + x] > 0) { // edge pixel
                    edge_count++;
                    // only iterate over amplitudes greater than the distance from the center to avoid redundant calculations (since circles with smaller amplitudes would not reach this far)
                    for (int amplitude = std::abs(x - x_center); amplitude < x_center; ++amplitude) { // maximum possible amplitude based on distance from center
                        if (amplitude <= 0) continue; // skip non-positive amplitudes
                        double angle = (theta / (float)num_rotations) * 2.0 * M_PI; // convert to radians

                        double phi = 0.0; // initialize phi
                        
                        if (gradient[theta * W + x] < 0) { // positive slope
                            phi = std::asin((x - x_center) / amplitude) - angle; // calculate phi based on x coordinate, amplitude, and theta
                        } else { // negative slope            
                            phi = std::acos((x - x_center) / amplitude) - angle + M_PI / 2.0; // calculate phi based on x coordinate, amplitude, and theta
                        }
                        
                        phi = std::fmod(phi, 2.0 * M_PI); // ensure phi is in [0, 2pi]
                        if (phi < 0) phi += 2.0 * M_PI;
                        int phi_bin = static_cast<int>((phi / (2.0 * M_PI)) * (num_rotations - 1));
                        phi_bin = phi_bin % num_rotations; // wrap around if necessary
                        phi_bin = std::max(0, std::min(phi_bin, num_rotations - 1)); // clamp to valid range
                        
                        
                        hough_space[amplitude * num_rotations + phi_bin] += std::exp(-0.001f * (x_center - amplitude - 1)); // accumulate votes in hough space with slight decay for larger amplitudes to prioritize smaller circles
                        if (hough_space[amplitude * num_rotations + phi_bin] > max_vote) {
                            max_vote = hough_space[amplitude * num_rotations + phi_bin]; // update max vote
                        }
                        
                    }
                }
            }
        }

        // extract 3d points from hough space
        extract_3d_points_from_hough_space(hough_space, num_rotations, x_center, z, points_3d, max_vote);

        //print stats
        std::cout << "Processed image " << z << ", max vote: " << max_vote << ", number of 3D points: " << points_3d.size() << std::endl;

        // write hough space to image for visualization
        uint8_t* hough_img = new uint8_t[num_rotations * x_center];
        max_vote = max_vote;
        for (int i = 0; i < num_rotations * x_center; ++i) {
            hough_img[i] = static_cast<uint8_t>(std::min(hough_space[i] / max_vote * 255.0 , 255.0)); // scale for visualization
        }
        int result = stbi_write_png("hough_space_z.png", num_rotations, x_center, 1, hough_img, num_rotations * sizeof(uint8_t));
        if (!result) {
            fprintf(stderr, "Failed to write image: %s\n", stbi_failure_reason());
            exit(EXIT_FAILURE);
        }


        delete[] hough_img; // free when done
        
        delete[] hough_space; // free when done
        stbi_image_free(img); // free when done
    }

    // write 3d points to ply file
    write_points_to_ply(points_3d, "points_3d.ply");
}







__global__ void pack_edge_pixels(const float* edge_img, const float* gradient, EdgePixel* edge_pixels, int* __restrict__ count, int W, int H, int coarse_factor) {
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * coarse_factor;
    int theta = (blockIdx.y * blockDim.y + threadIdx.y) * coarse_factor;
    if (x >= W || theta >= H) return; // bounds check
    for(int dx = 0; dx < coarse_factor; ++dx) {
        for(int dtheta = 0; dtheta < coarse_factor; ++dtheta) {
            int x_idx = x + dx;
            int theta_idx = theta + dtheta;
            if (x_idx < W && theta_idx < H) { // bounds check
                if (edge_img[theta_idx * W + x_idx] > 0) { // edge pixel
                    int idx = atomicAdd(count, 1); // get index for this edge pixel
                    edge_pixels[idx].x = x_idx;
                    edge_pixels[idx].theta = theta_idx;
                    edge_pixels[idx].gradient_sign = gradient[theta_idx * W + x_idx] > 0;                   
                }
            }
        }
    }
}


__global__ void compute_hough_space(const EdgePixel* edge_pixels, int num_edge_pixels, 
                                     float* hough_space, int num_rotations, int x_center) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_edge_pixels) return; // bounds check
    
    EdgePixel edge_pixel = edge_pixels[idx];
    int x = edge_pixel.x;
    int theta = edge_pixel.theta;
    bool gradient_sign = edge_pixel.gradient_sign;

    if (x < 0 || x >= x_center * 2) return; // sanity check
    if (x == x_center) return;
    if (theta < 0 || theta >= num_rotations) return;

    for (int amplitude = abs(x - x_center); amplitude < x_center; ++amplitude) {
        if (amplitude <= 0) continue;

        float arg = (float)(x - x_center) / (float)amplitude;
        if (arg < -1.0 || arg > 1.0) continue; // guard asin/acos domain

        float angle = (theta / (float)num_rotations) * 2.0 * M_PI;
        float phi;

        if (gradient_sign) {
            phi = asinf(arg) - angle;
        } else {
            phi = acosf(arg) - angle + M_PI / 2.0;
        }

        phi = fmodf(phi, 2.0f * M_PI);
        if (phi < 0.0) phi += 2.0f * M_PI;

        int phi_bin = (int)((phi / (2.0f * M_PI)) * (num_rotations - 1));
        phi_bin = max(0, min(phi_bin, num_rotations - 1));

        // bounds check before atomicAdd
        int flat_idx = amplitude * num_rotations + phi_bin;
        if (flat_idx < 0 || flat_idx >= x_center * num_rotations) {
            continue;
        }

        atomicAdd(&hough_space[flat_idx], (float)expf(-0.001f * amplitude)); // accumulate votes with slight decay for larger amplitudes to prioritize smaller circles
    }
}

/*
Go through the amplitudes, with a shared memory for each amplitude, and accumulate votes for each amplitude in the edge pixels.
*/

extern __shared__ float shared_votes[]; // sized num_rotations floats at launch

__global__ void compute_hough_space_amplitude(const EdgePixel* edge_pixels, int num_edge_pixels,
                                               float* hough_space, int num_rotations, int x_center,
                                               int amplitude) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // zero-init shared memory
    for (int i = tid; i < num_rotations; i += blockDim.x) shared_votes[i] = 0.0f;
    __syncthreads();

    if (idx < num_edge_pixels) {
        EdgePixel edge_pixel = edge_pixels[idx];
        int x = edge_pixel.x;
        int theta = edge_pixel.theta;
        bool gradient_sign = edge_pixel.gradient_sign;

        bool valid = !(x < 0 || x >= x_center * 2) && x != x_center &&
                     !(theta < 0 || theta >= num_rotations);

        if (valid) {
            float arg = (float)(x - x_center) / (float)amplitude;
            valid = !(arg < -1.0f || arg > 1.0f);

            if (valid) {
                float angle = (theta / (float)num_rotations) * 2.0f * (float)M_PI;
                float phi = gradient_sign ? (asinf(arg) - angle)
                                           : (acosf(arg) - angle + (float)M_PI / 2.0f);
                phi = fmodf(phi, 2.0f * (float)M_PI);
                if (phi < 0.0f) phi += 2.0f * (float)M_PI;

                int phi_bin = (int)((phi / (2.0f * (float)M_PI)) * (num_rotations - 1));
                phi_bin = max(0, min(phi_bin, num_rotations - 1));

                atomicAdd(&shared_votes[phi_bin], expf(-0.001f * amplitude));
            }
        }
    }

    __syncthreads(); // ensure all threads have finished accumulating votes before writing to global memory

    // write shared votes to global, parallelized across threads
    for (int i = tid; i < num_rotations; i += blockDim.x) {
        if (shared_votes[i] > 0.0f) {
            atomicAdd(&hough_space[amplitude * num_rotations + i], shared_votes[i]);
        }
    }
}

__global__ void compute_radius_kernel(float* d_radius, float hough_sum, int N, int x_center, int num_rotations) {
    float temp = (hough_sum > 0.0f) ? (float)N / hough_sum : 0.9f;
    if (!isfinite(temp)) temp = 0.9f;
    *d_radius = temp * min(x_center, num_rotations);
}

__global__ void compute_threshold_kernel(float* d_threshold, float global_max, float peak_threshold, float abs_min_votes) {
    *d_threshold = max(peak_threshold * global_max, abs_min_votes);
}



__global__ void peak_local_max(
    const float* __restrict__ hough_space,
    int* __restrict__ peak_count,
    Point3D* __restrict__ points_3d,
    int rows, int cols,
    const float* __restrict__ threshold_abs_ptr,
    int min_distance,
    int z,
    int num_rotations,
    int max_points)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= cols || row >= rows) return;

    // skip border region — peaks here are artifacts not real circles
    if (row < min_distance || row >= rows - min_distance) return;
    if (col < min_distance || col >= cols - min_distance) return;

    // additionally skip very small amplitudes — radius=0,1,2 are never real
    if (row < 5) return;

    float val = hough_space[row * cols + col];
    if (val < *threshold_abs_ptr) return;

    int r_start = max(0,    row - min_distance);
    int r_end   = min(rows, row + min_distance + 1);
    int c_start = max(0,    col - min_distance);
    int c_end   = min(cols, col + min_distance + 1);

    for (int r = r_start; r < r_end; ++r) {
        for (int c = c_start; c < c_end; ++c) {
            if (hough_space[r * cols + c] > val) return;
            if (hough_space[r * cols + c] == val && 
                (r < row || (r == row && c < col))) return;
        }
    }

    int amplitude = row;
    float phi = (col / (float)(num_rotations - 1)) * 2.0 * M_PI;

    int idx = atomicAdd(peak_count, 1);
    if (idx < max_points) {
        points_3d[idx] = {
            (float)(amplitude * cosf(phi)),
            (float)(amplitude * sinf(phi)),
            (float)z,
        };
    }
}

// The heart of this project
void calculate_hough_spaces_parallel(const char*dir, int batch_size) {
    // For each edge pixel, we will calculate the corresponding (z, amplitude) values for the Hough space.
    // z = height of the circle center
    // amplitude = radius of the circle

    // get path list of images in directory
    std::vector<std::string> image_paths;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            image_paths.push_back(entry.path().string());
        }
    }
    // numeric sort by filename stem (e.g. "2.png" before "10.png") — lexicographic
    // sort would otherwise order "10.png" before "2.png"
    std::sort(image_paths.begin(), image_paths.end(), [](const std::string& a, const std::string& b) {
        return std::stol(fs::path(a).stem().string()) < std::stol(fs::path(b).stem().string());
    });
    int num_images = image_paths.size();

    //debug print filenames
    // for (int i = 0; i < std::min(30, num_images); ++i) {
    //     std::cout << image_paths[i] << std::endl;
    //}

    // get first image to determine dimensions
    int W, H, channels;
    float* img = stbi_loadf(image_paths[0].c_str(), &W, &H, &channels, 1);
    if (!img) {
        fprintf(stderr, "Failed to load image: %s\n", stbi_failure_reason());
        exit(EXIT_FAILURE);
    }
    stbi_image_free(img);  // free when done


    int x_center = std::floor(W / 2.0f);  // maximum possible radius based on image width
    int num_rotations = H;
    int N = x_center * num_rotations; // total number of elements in hough space
    // printf("Image dimensions: %d x %d\n", W, H);
    // printf("Hough space dimensions: %d x %d\n", num_rotations, x_center);

    std::vector<Point3D> points_3d; // accumulates results, appended to after every batch

    int max_points_per_batch = MAX_PEAKS * batch_size;
    Point3D* d_points_3d;
    int*     d_point_count;
    CHECK_CUDA(cudaMalloc(&d_points_3d,  max_points_per_batch * sizeof(Point3D)));
    CHECK_CUDA(cudaMalloc(&d_point_count, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_point_count, 0, sizeof(int)));

    // create streams and events for each image in the batch
    std::vector<cudaStream_t> streams(batch_size);
    std::vector<cudaEvent_t> start_events(batch_size);
    std::vector<cudaEvent_t> copy_events(batch_size);
    std::vector<cudaEvent_t> edge_events(batch_size);
    std::vector<cudaEvent_t> pack_events(batch_size);
    std::vector<cudaEvent_t> hough_events(batch_size);
    std::vector<cudaEvent_t> peak_events(batch_size);

    // timing variables for each stage
    double copy_time = 0.0;
    double edge_time = 0.0;
    double pack_time = 0.0;
    double hough_time = 0.0;
    double peak_time = 0.0;

    // additional host and device buffers for each image in the batch
    std::vector<float*> h_imgs(batch_size, nullptr);

    std::vector<float*> d_img(batch_size);
    std::vector<float*> d_edge_img(batch_size);
    std::vector<float*> d_gradient(batch_size);
    std::vector<float*> d_blurred(batch_size);
    std::vector<float*> d_sobel_mag(batch_size);
    std::vector<float*> d_sobel_dir(batch_size);
    std::vector<float*> d_nms(batch_size);
    std::vector<float*> d_hough_space(batch_size);
    std::vector<EdgePixel*> d_edge_pixels(batch_size);
    std::vector<int*> d_edge_count(batch_size);
    std::vector<int> h_edge_count(batch_size, 0);

    // FFT plans and frequency buffers for each image in the batch
    std::vector<cufftHandle> plans_fwd(batch_size);
    std::vector<cufftHandle> plans_inv(batch_size);
    std::vector<cufftComplex*> d_freq(batch_size);

    std::vector<float*> d_radius(batch_size);
    std::vector<float*> d_threshold(batch_size);

    // create streams, events, and allocate device memory for each image in the batch
    for (int i = 0; i < batch_size; ++i) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
        CHECK_CUDA(cudaEventCreate(&start_events[i]));
        CHECK_CUDA(cudaEventCreate(&copy_events[i]));
        CHECK_CUDA(cudaEventCreate(&edge_events[i]));
        CHECK_CUDA(cudaEventCreate(&pack_events[i]));
        CHECK_CUDA(cudaEventCreate(&hough_events[i]));
        CHECK_CUDA(cudaEventCreate(&peak_events[i]));

        CHECK_CUDA(cudaMalloc(&d_img[i], W * H * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_edge_img[i], W * H * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_gradient[i], W * H * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_blurred[i], W * H * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_sobel_mag[i], W * H * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_sobel_dir[i], W * H * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_nms[i], W * H * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_hough_space[i], num_rotations * x_center * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_edge_pixels[i], W * H * sizeof(EdgePixel))); // worst case: every pixel is an edge
        CHECK_CUDA(cudaMalloc(&d_edge_count[i], sizeof(int)));

        // FFT plans for hough space post-processing — created once, reused for every image
        CHECK_CUDA(cudaMalloc(&d_freq[i], N * sizeof(cufftComplex)));
        cufftPlan2d(&plans_fwd[i], x_center, num_rotations, CUFFT_C2C);
        cufftPlan2d(&plans_inv[i], x_center, num_rotations, CUFFT_C2C);
        cufftSetStream(plans_fwd[i], streams[i]);
        cufftSetStream(plans_inv[i], streams[i]);

        CHECK_CUDA(cudaMalloc(&d_radius[i], sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_threshold[i], sizeof(float)));
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // start Batch Processing
    for (int batch = 0; batch < num_images; batch += batch_size) {
        int current_batch_size = std::min(batch_size, num_images - batch);

        print_progress(batch, num_images, start_time, copy_time, edge_time, pack_time, hough_time, peak_time);

        // load image, copy to device
        for (int i = 0; i < current_batch_size; ++i) {

            // load image for this height
            int img_idx = batch + i;
            h_imgs[i] = stbi_loadf(image_paths[img_idx].c_str(), &W, &H, nullptr, 1);
            if (!h_imgs[i]) {
                fprintf(stderr, "Failed to load image: %s\n", stbi_failure_reason());
                exit(EXIT_FAILURE);
            }
            
            CHECK_CUDA(cudaEventRecord(start_events[i], streams[i]));

            CHECK_CUDA(cudaMemsetAsync(d_edge_count[i], 0, sizeof(int), streams[i])); // initialize to 0
            CHECK_CUDA(cudaMemsetAsync(d_hough_space[i], 0, num_rotations * x_center * sizeof(float), streams[i])); // initialize to 0

            // copy to device
            CHECK_CUDA(cudaMemcpyAsync(d_img[i], h_imgs[i], W * H * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
            CHECK_CUDA(cudaEventRecord(copy_events[i], streams[i]));
        }

        // compute edges and sobel gradients for each image in the batch
        for (int i = 0; i < current_batch_size; ++i) {

            CHECK_CUDA(cudaEventSynchronize(copy_events[i])); // ensure copy is complete before freeing host image
            
            stbi_image_free(h_imgs[i]); // free host image after copy to device

            // compute edges (device-native: no internal malloc/free/memcpy per image)
            canny_edge_device(d_img[i], d_edge_img[i], d_blurred[i], d_sobel_mag[i], d_sobel_dir[i], d_nms[i], W, H, 0.001f, 0.01f, streams[i]);
            sobel_grad_device(d_img[i], d_gradient[i], W, H, streams[i]);

            CHECK_CUDA(cudaEventRecord(edge_events[i], streams[i]));
        }

        // pack edge pixels into tuples for warp divergence
        for (int i = 0; i < current_batch_size; ++i) {
            CHECK_CUDA(cudaStreamWaitEvent(streams[i], edge_events[i], 0)); // ensure edge detection is complete before packing edge pixels

            
            int coarse_factor = COARSE_FACTOR; // adjust as needed to balance parallelism and warp divergence
            dim3 block = dim3(16, 16); // adjust block size as needed
            dim3 grid = dim3((W + block.x * coarse_factor - 1) / (block.x * coarse_factor), (H + block.y * coarse_factor - 1) / (block.y * coarse_factor));

            // pack edge pixels into tuples for warp divergence
            pack_edge_pixels<<<grid, block, 0, streams[i]>>>(d_edge_img[i], d_gradient[i], d_edge_pixels[i], d_edge_count[i], W, H, coarse_factor);

            CHECK_CUDA(cudaEventRecord(pack_events[i], streams[i]));
        }


        std::vector<bool> processed(batch_size, false); // track which images were processed
        
        // copy edge count back to host for each image in the batch
        for (int i = 0; i < current_batch_size; ++i) {
            // ensure pack kernel is complete before copying edge count
            CHECK_CUDA(cudaStreamWaitEvent(streams[i], pack_events[i], 0));
            
            // copy edge count back to host
            CHECK_CUDA(cudaMemcpyAsync(&h_edge_count[i], d_edge_count[i], sizeof(int), cudaMemcpyDeviceToHost, streams[i]));
        }

        // Hough space computation
        for (int i = 0; i < current_batch_size; ++i) {
            CHECK_CUDA(cudaEventSynchronize(pack_events[i])); // ensure pack kernel is complete before copying edge count
            int num_edge_pixels = h_edge_count[i];
            
            // Skip if no edge pixels found
            if (num_edge_pixels == 0) {
                // printf("Warning: No edge pixels found for image %d\n", i);
                processed[i] = false;
                continue;
            }
            
            processed[i] = true;

            
            
            // compute hough space
            dim3 block = dim3(256); // adjust block size as needed
            dim3 grid = dim3((num_edge_pixels + block.x - 1) / block.x);
            //compute_hough_space<<<grid, block, 0, streams[i]>>>(d_edge_pixels[i], num_edge_pixels, d_hough_space[i], num_rotations, x_center);

            // Share memory across threads in a block for each amplitude, to reduce atomicAdds on global memory
            size_t shared_bytes = num_rotations * sizeof(float);
            for (int amplitude = 1; amplitude < x_center; ++amplitude) {
                compute_hough_space_amplitude<<<grid, block, shared_bytes, streams[i]>>>(
                    d_edge_pixels[i], num_edge_pixels, d_hough_space[i], num_rotations, x_center, amplitude);
            }


            CHECK_CUDA(cudaEventRecord(hough_events[i], streams[i]));

        }

        // Peak calculation and post-processing
        for (int i = 0; i < current_batch_size; ++i) {
            CHECK_CUDA(cudaStreamWaitEvent(streams[i], hough_events[i], 0)); // ensure hough space is computed before post-processing
            

            dim3 block1(256);
            dim3 grid1((N + 255) / 256);
            dim3 block2(16, 16);
            dim3 grid2((num_rotations + 15) / 16, (x_center + 15) / 16);

            // compute radius from hough space sum using thrust
            thrust::device_ptr<float> h_ptr(d_hough_space[i]);
            float hough_sum = thrust::reduce(thrust::cuda::par.on(streams[i]),
                                              h_ptr, h_ptr + N, 0.0f, thrust::plus<float>());

            compute_radius_kernel<<<1, 1, 0, streams[i]>>>(d_radius[i], hough_sum, N, x_center, num_rotations);

            // real -> complex
            real_to_complex<<<grid1, block1, 0, streams[i]>>>(
                d_hough_space[i], d_freq[i], N);
            
            // fftshift (multiply by (-1)^(u+v) before FFT)
            fftshift_2d<<<grid2, block2, 0, streams[i]>>>(
                d_freq[i], x_center, num_rotations);
            
            // forward FFT
            cufftExecC2C(plans_fwd[i], d_freq[i], d_freq[i], CUFFT_FORWARD);
            
            // apply circular lowpass mask
            apply_lowpass_mask<<<grid2, block2, 0, streams[i]>>>(
                d_freq[i], x_center, num_rotations, d_radius[i]);
            
            // inverse FFT
            cufftExecC2C(plans_inv[i], d_freq[i], d_freq[i], CUFFT_INVERSE);
            
            // ifftshift
            fftshift_2d<<<grid2, block2, 0, streams[i]>>>(
                d_freq[i], x_center, num_rotations);
            
            // subtract lowpass from original and clip negative values
            subtract_and_clip<<<grid1, block1, 0, streams[i]>>>(
                d_hough_space[i], d_freq[i], N, (float)N);

            // global max
            thrust::device_ptr<float> ptr(d_hough_space[i]);
            float global_max = thrust::reduce(thrust::cuda::par.on(streams[i]),
                                               ptr, ptr + N,
                                               0.0f, thrust::maximum<float>());

            compute_threshold_kernel<<<1, 1, 0, streams[i]>>>(d_threshold[i], global_max, PEAK_THRESHOLD, ABS_MIN_VOTES);
            

            // compute 3d points from hough space (threshold read on-device — no host round-trip)
            int z_val = num_images - (batch + i);
            peak_local_max<<<grid2, block2, 0, streams[i]>>>(
                d_hough_space[i],
                d_point_count,       // counter shared across this batch's slices, drained/reset after the batch
                d_points_3d,         // buffer shared across this batch's slices, drained/reset after the batch
                x_center, num_rotations,
                d_threshold[i],
                5,
                z_val,
                num_rotations,
                max_points_per_batch);
                
            CHECK_CUDA(cudaEventRecord(peak_events[i], streams[i]));
            
        }

        // Synchronize before copying
        for (int i = 0; i < current_batch_size; ++i) {
            CHECK_CUDA(cudaEventSynchronize(peak_events[i]));

            float ms;

            CHECK_CUDA(cudaEventElapsedTime(&ms, start_events[i], copy_events[i]));
            copy_time += ms;

            CHECK_CUDA(cudaEventElapsedTime(&ms, copy_events[i], edge_events[i]));
            edge_time += ms;

            CHECK_CUDA(cudaEventElapsedTime(&ms, edge_events[i], pack_events[i]));
            pack_time += ms;

            if (processed[i]) {
                CHECK_CUDA(cudaEventElapsedTime(&ms, pack_events[i], hough_events[i]));
                hough_time += ms;

                CHECK_CUDA(cudaEventElapsedTime(&ms, hough_events[i], peak_events[i]));
                peak_time += ms;
            }
        }
        // clculate number of new 3d points found in this batch and copy to host
        int h_batch_point_count = 0;
        CHECK_CUDA(cudaMemcpy(&h_batch_point_count, d_point_count, sizeof(int), cudaMemcpyDeviceToHost));
        int copy_count = std::min(h_batch_point_count, max_points_per_batch);

        size_t old_size = points_3d.size();
        points_3d.resize(old_size + copy_count);

        CHECK_CUDA(cudaMemcpy(points_3d.data() + old_size, d_points_3d,
                        copy_count * sizeof(Point3D), cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaMemset(d_point_count, 0, sizeof(int))); // reset for next batch
        CHECK_CUDA(cudaMemset(d_points_3d, 0, max_points_per_batch * sizeof(Point3D))); // reset for next batch
    }

    printf("Total number of 3D points: %zu\n", points_3d.size());
    // write 3d points to ply file
    write_points_to_ply(points_3d, "../points_3d_parallel.ply");

    // Free device memory and destroy streams/events
    for (int i = 0; i < batch_size; ++i) {
        CHECK_CUDA(cudaFree(d_img[i]));
        CHECK_CUDA(cudaFree(d_edge_img[i]));
        CHECK_CUDA(cudaFree(d_gradient[i]));
        CHECK_CUDA(cudaFree(d_blurred[i]));
        CHECK_CUDA(cudaFree(d_sobel_mag[i]));
        CHECK_CUDA(cudaFree(d_sobel_dir[i]));
        CHECK_CUDA(cudaFree(d_nms[i]));
        CHECK_CUDA(cudaFree(d_hough_space[i]));
        CHECK_CUDA(cudaFree(d_edge_pixels[i]));
        CHECK_CUDA(cudaFree(d_edge_count[i]));

        cufftDestroy(plans_fwd[i]);
        cufftDestroy(plans_inv[i]);
        CHECK_CUDA(cudaFree(d_freq[i]));

        CHECK_CUDA(cudaFree(d_radius[i]));
        CHECK_CUDA(cudaFree(d_threshold[i]));

        CHECK_CUDA(cudaEventDestroy(copy_events[i]));
        CHECK_CUDA(cudaEventDestroy(edge_events[i]));
        CHECK_CUDA(cudaEventDestroy(pack_events[i]));
        CHECK_CUDA(cudaEventDestroy(hough_events[i]));
        CHECK_CUDA(cudaEventDestroy(peak_events[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }

    // Free shared device memory for 3D points and point count
    CHECK_CUDA(cudaFree(d_points_3d));
    CHECK_CUDA(cudaFree(d_point_count));
}

int main()
{   

    auto start = std::chrono::system_clock::now();

    calculate_hough_spaces_parallel("example_data/", BATCH_SIZE);

    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(elapsed_seconds);
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed_seconds - minutes);
    std::cout << "Elapsed time: " << minutes.count() << "m " << seconds.count() << "s" << std::endl;

    


    // load images

    // canny-edge
    // convert non-zeros to list of indices (x0,y0,x1,y1,x2,y2,...) for warp divergence
    // sort tuples by x value (higher x values have more work, thus keeping it similar across blocks)

    // create hough spaces (z, amplitute, phi)
    // z=height
    // amplitude = radius of circle
    // phi = angle of circle

    // launch thread for each edge tuple, creating a line in the Hough accumulator for each tuple
    // - use shared memory to accumulate the hough entries for each block, then atomically add to global memory
    
    // finished hough space.
    // find peaks in hough space, convert to 3D points, and write to ply file
    
}
