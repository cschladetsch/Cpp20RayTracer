#include <iostream>
#include <cuda_runtime.h>
#include "vec3.h"  // Include the new BMP writer header
#include "bmp_writer.h"  // Include the new BMP writer header

// Sphere struct
struct Sphere {
    Vec3 center;
    float radius;
    Vec3 color;
    float reflectivity;

    __host__ __device__ Sphere() {}
    __host__ __device__ Sphere(const Vec3& c, float r, const Vec3& col, float refl) 
        : center(c), radius(r), color(col), reflectivity(refl) {}

    __device__ bool intersect(const Vec3& origin, const Vec3& direction, float& t) const {
        Vec3 oc = origin - center;
        float a = direction.dot(direction);
        float b = 2.0f * oc.dot(direction);
        float c = oc.dot(oc) - radius * radius;
        float discriminant = b * b - 4 * a * c;
        if (discriminant > 0) {
            t = (-b - sqrtf(discriminant)) / (2.0f * a);
            return true;
        }
        return false;
    }
};

// Kernel for rendering
__device__ Vec3 trace_ray(const Vec3& origin, const Vec3& direction, const Sphere* spheres, int num_spheres, int depth) {
    Vec3 color(0.5f, 0.7f, 1.0f);  // Sky blue color
    float t_min = 1e20f;
    int hit_sphere = -1;

    for (int k = 0; k < num_spheres; ++k) {
        float t = 0;
        if (spheres[k].intersect(origin, direction, t) && t < t_min) {
            t_min = t;
            hit_sphere = k;
        }
    }

    if (hit_sphere != -1) {
        Vec3 hit_point = origin + direction * t_min;
        Vec3 normal = (hit_point - spheres[hit_sphere].center).normalize();
        color = spheres[hit_sphere].color;

        // Reflection
        if (spheres[hit_sphere].reflectivity > 0 && depth < 5) {
            Vec3 reflected_dir = direction.reflect(normal).normalize();
            Vec3 reflected_color = trace_ray(hit_point, reflected_dir, spheres, num_spheres, depth + 1);
            color = color * (1.0f - spheres[hit_sphere].reflectivity) + reflected_color * spheres[hit_sphere].reflectivity;
        }
    }

    return color;
}

__global__ void render_kernel(Vec3* framebuffer, int width, int height, const Sphere* spheres, int num_spheres) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height) return;

    int pixel_index = j * width + i;

    Vec3 origin(0.0f, 0.0f, 0.0f);
    Vec3 direction(float(i) / float(width) - 0.5f, float(height - j) / float(height) - 0.5f, -1.0f);
    direction = direction.normalize();

    framebuffer[pixel_index] = trace_ray(origin, direction, spheres, num_spheres, 0);
}

int main() {
    int width = 800, height = 600;
    Vec3* framebuffer;
    cudaMallocManaged(&framebuffer, width * height * sizeof(Vec3));

    // Define spheres
    Sphere* spheres;
    cudaMallocManaged(&spheres, 3 * sizeof(Sphere));
    spheres[0] = Sphere(Vec3(-0.5f, 0.0f, -1.0f), 0.5f, Vec3(1.0f, 0.0f, 0.0f), 0.8f); // Red reflective sphere
    spheres[1] = Sphere(Vec3(0.5f, 0.0f, -1.5f), 0.3f, Vec3(0.0f, 1.0f, 0.0f), 0.3f); // Green semi-reflective sphere
    spheres[2] = Sphere(Vec3(0.0f, -1000.5f, -1.0f), 1000.0f, Vec3(0.5f, 0.5f, 0.5f), 0.0f); // Ground plane

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    render_kernel<<<numBlocks, threadsPerBlock>>>(framebuffer, width, height, spheres, 3);
    cudaDeviceSynchronize();

    writeBMP("output.bmp", width, height, framebuffer);

    cudaFree(framebuffer);
    cudaFree(spheres);

    return 0;
}
