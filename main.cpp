#include <vector>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "light.h"
#include "cuda_utils.h"
#include "stb_image_write.h"

int main() {
    int nx = 800;
    int ny = 600;
    int num_pixels = nx * ny;
    int max_depth = 10;
    
    Vec3 *fb;
    checkCudaErrors(cudaMallocManaged(&fb, num_pixels * sizeof(Vec3)));
    
    // Camera setup
    Vec3 camera_pos(0, 2, -5);
    Vec3 camera_dir = Vec3(0, 0, 1).normalize();
    Vec3 camera_up = Vec3(0, 1, 0);
    Vec3 camera_right = camera_up.cross(camera_dir).normalize();
    float fov = 60.0f; // Field of view in degrees
    float focal_length = 1.0f;
    float viewport_height = 2.0f * focal_length * tanf(fov * 0.5f * PI / 180.0f);
    float viewport_width = viewport_height * ((float)nx / ny);
    
    // Scene setup
    std::vector<Sphere> h_spheres = {
        Sphere(Vec3(0, 1, 0), 1, Material(Vec3(0.7f, 0.3f, 0.3f), 0.8f, 50)),
        Sphere(Vec3(-2.5f, 0.5f, 2), 0.5f, Material(Vec3(0.3f, 0.7f, 0.3f), 0.3f, 10)),
        Sphere(Vec3(2.5f, 0.5f, -2), 0.5f, Material(Vec3(0.3f, 0.3f, 0.7f), 0.5f, 30))
    };
    int num_spheres = h_spheres.size();
    
    Sphere *d_spheres;
    checkCudaErrors(cudaMalloc(&d_spheres, num_spheres * sizeof(Sphere)));
    checkCudaErrors(cudaMemcpy(d_spheres, h_spheres.data(), num_spheres * sizeof(Sphere), cudaMemcpyHostToDevice));
    
    // Light setup
    std::vector<Light> h_lights = {
        Light{Vec3(-5, 5, -5), Vec3(1, 1, 1), 1.0f, 0.5f}
    };
    int num_lights = h_lights.size();

    Light *d_lights;
    checkCudaErrors(cudaMalloc(&d_lights, num_lights * sizeof(Light)));
    checkCudaErrors(cudaMemcpy(d_lights, h_lights.data(), num_lights * sizeof(Light), cudaMemcpyHostToDevice));

    // CUDA block and grid size setup
    int tx = 16;
    int ty = 16;
    dim3 blocks((nx + tx - 1) / tx, (ny + ty - 1) / ty);
    dim3 threads(tx, ty);

    // Launch the CUDA kernel
    render_kernel<<<blocks, threads>>>(fb, nx, ny, camera_pos, camera_dir, camera_right, camera_up, viewport_width, viewport_height, focal_length, d_spheres, num_spheres, d_lights, num_lights, max_depth, time(0));
    checkCudaErrors(cudaDeviceSynchronize());

    // Save the rendered image
    std::vector<unsigned char> image(num_pixels * 3);
    for (int i = 0; i < num_pixels; i++) {
        image[i * 3 + 0] = static_cast<unsigned char>(255.99f * fb[i].x);
        image[i * 3 + 1] = static_cast<unsigned char>(255.99f * fb[i].y);
        image[i * 3 + 2] = static_cast<unsigned char>(255.99f * fb[i].z);
    }
    
    stbi_write_png("output.png", nx, ny, 3, image.data(), nx * 3);

    // Free CUDA memory
    cudaFree(fb);
    cudaFree(d_spheres);
    cudaFree(d_lights);

    return 0;
}

