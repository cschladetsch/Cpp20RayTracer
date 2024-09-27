#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>
#include <fstream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct Vec3 {
    double x, y, z;
    Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator-() const { return Vec3(-x, -y, -z); } // Add this line
    Vec3 operator*(double d) const { return Vec3(x * d, y * d, z * d); }
    Vec3 operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
    Vec3 normalize() const {
        double mg = sqrt(x * x + y * y + z * z);
        return Vec3(x / mg, y / mg, z / mg);
    }
    double dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    Vec3 cross(const Vec3& v) const {
        return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }
};
struct Ray {
    Vec3 origin, direction;
    Ray(const Vec3& origin, const Vec3& direction) : origin(origin), direction(direction) {}
};

struct Light {
    Vec3 position;
    Vec3 color;
    double intensity;
    double radius;  // For soft shadows
};

class Material {
public:
    Vec3 color;
    double reflectivity;
    double specularity;
    Material(const Vec3& c, double r, double s) : color(c), reflectivity(r), specularity(s) {}
};

struct Sphere {
    Vec3 center;
    double radius;
    Material material;
    Sphere(const Vec3& c, double r, const Material& m) : center(c), radius(r), material(m) {}
    bool intersect(const Ray& ray, double& t) const {
        Vec3 oc = ray.origin - center;
        double b = oc.dot(ray.direction);
        double c = oc.dot(oc) - radius * radius;
        double discriminant = b * b - c;
        if (discriminant < 0) return false;
        double sqrtd = sqrt(discriminant);
        double root = (-b - sqrtd);
        if (root < 0) {
            root = (-b + sqrtd);
            if (root < 0) return false;
        }
        t = root;
        return true;
    }
};

Vec3 sky_color(const Ray& ray) {
    Vec3 unit_direction = ray.direction.normalize();
    double t = 0.5 * (unit_direction.y + 1.0);
    return Vec3(0.5, 0.7, 1.0) * t + Vec3(1.0, 1.0, 1.0) * (1.0 - t);
}

bool check_ground_intersection(const Ray& ray, double& t) {
    if (std::abs(ray.direction.y) > 1e-6) {
        t = -ray.origin.y / ray.direction.y;
        return t > 0;
    }
    return false;
}

Vec3 soft_shadow(const Vec3& point, const Vec3& normal, const Light& light, const std::vector<Sphere>& spheres, int samples) {
    Vec3 shadow(0, 0, 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int i = 0; i < samples; ++i) {
        double u = dis(gen);
        double v = dis(gen);
        double theta = 2 * M_PI * u;
        double phi = acos(2 * v - 1);
        Vec3 light_point = light.position + Vec3(light.radius * sin(phi) * cos(theta),
                                                 light.radius * sin(phi) * sin(theta),
                                                 light.radius * cos(phi));
        
        Vec3 light_dir = (light_point - point).normalize();
        Ray shadow_ray(point + normal * 0.001, light_dir);
        
        bool in_shadow = false;
        for (const auto& sphere : spheres) {
            double t;
            if (sphere.intersect(shadow_ray, t) && t > 0.001) {
                in_shadow = true;
                break;
            }
        }
        
        if (!in_shadow) {
            shadow = shadow + Vec3(1, 1, 1);
        }
    }
    
    return shadow * (1.0 / samples);
}

Vec3 calculate_lighting(const Vec3& point, const Vec3& normal, const Vec3& view_dir, 
                        const Material& material, const std::vector<Light>& lights,
                        const std::vector<Sphere>& spheres) {
    Vec3 color(0, 0, 0);
    for (const auto& light : lights) {
        Vec3 light_dir = (light.position - point).normalize();
        
        Vec3 shadow_factor = soft_shadow(point, normal, light, spheres, 16);
        
        double diffuse = std::max(0.0, normal.dot(light_dir));
        Vec3 reflect_dir = (normal * 2 * normal.dot(light_dir) - light_dir).normalize();
        double specular = std::pow(std::max(0.0, view_dir.dot(reflect_dir)), material.specularity);
        
        color = color + material.color * light.color * light.intensity * (diffuse + specular) * shadow_factor;
    }
    return color;
}

Vec3 get_color(const Ray& ray, const std::vector<Sphere>& spheres, const std::vector<Light>& lights, int bounces) {
    if (bounces <= 0) return sky_color(ray);

    double closest_t = std::numeric_limits<double>::infinity();
    const Sphere* hit_sphere = nullptr;
    double t_ground;
    bool hit_ground = check_ground_intersection(ray, t_ground);

    for (const auto& sphere : spheres) {
        double t;
        if (sphere.intersect(ray, t) && t < closest_t) {
            closest_t = t;
            hit_sphere = &sphere;
        }
    }

    if (hit_sphere && (!hit_ground || closest_t < t_ground)) {
        Vec3 hit_point = ray.origin + ray.direction * closest_t;
        Vec3 normal = (hit_point - hit_sphere->center).normalize();
        Vec3 view_dir = -ray.direction;
        
        Vec3 direct_color = calculate_lighting(hit_point, normal, view_dir, hit_sphere->material, lights, spheres);
        
        if (hit_sphere->material.reflectivity > 0) {
            Vec3 reflect_dir = (ray.direction - normal * 2 * ray.direction.dot(normal)).normalize();
            Ray reflect_ray(hit_point + normal * 0.001, reflect_dir);
            Vec3 reflect_color = get_color(reflect_ray, spheres, lights, bounces - 1);
            return direct_color * (1 - hit_sphere->material.reflectivity) + reflect_color * hit_sphere->material.reflectivity;
        }
        
        return direct_color;
    }

    if (hit_ground) {
        Vec3 hit_point = ray.origin + ray.direction * t_ground;
        int check_x = static_cast<int>(std::floor(hit_point.x)) % 2;
        int check_z = static_cast<int>(std::floor(hit_point.z)) % 2;
        Vec3 ground_color = (check_x + check_z) % 2 == 0 ? Vec3(0.8, 0.8, 0.8) : Vec3(0.3, 0.3, 0.3);
        return calculate_lighting(hit_point, Vec3(0, 1, 0), -ray.direction, Material(ground_color, 0, 10), lights, spheres);
    }

    return sky_color(ray);
}
int main(int argc, char* argv[]) {
    int max_bounces = 10;
    if (argc > 1) {
        max_bounces = std::stoi(argv[1]);
    }

    const int width = 800;
    const int height = 600;
    const double aspect_ratio = static_cast<double>(width) / height;

    std::vector<unsigned char> image(width * height * 3);

    // Camera setup with more realistic FOV
    double fov = 60.0; // Field of view in degrees
    double focal_length = 1.0;
    double viewport_height = 2.0 * focal_length * tan(fov * 0.5 * M_PI / 180.0);
    double viewport_width = viewport_height * aspect_ratio;

    Vec3 camera_pos(0, 2, -5);
    Vec3 camera_dir = Vec3(0, 0, 1).normalize();
    Vec3 camera_up = Vec3(0, 1, 0);
    Vec3 camera_right = camera_up.cross(camera_dir).normalize();

    std::vector<Sphere> spheres = {
        Sphere(Vec3(0, 1, 0), 1, Material(Vec3(0.7, 0.3, 0.3), 0.8, 50)),
        Sphere(Vec3(-2.5, 0.5, 2), 0.5, Material(Vec3(0.3, 0.7, 0.3), 0.3, 10)),
        Sphere(Vec3(2.5, 0.5, -2), 0.5, Material(Vec3(0.3, 0.3, 0.7), 0.5, 30)),
        // Additional spheres
        Sphere(Vec3(-1.5, 0.3, -1), 0.3, Material(Vec3(0.9, 0.8, 0.2), 0.1, 5)),
        Sphere(Vec3(1.5, 0.3, 1), 0.3, Material(Vec3(0.2, 0.8, 0.9), 0.6, 40)),
        Sphere(Vec3(0, 0.2, 2), 0.2, Material(Vec3(0.8, 0.2, 0.8), 0.7, 60)),
        Sphere(Vec3(-1, 0.2, 3), 0.2, Material(Vec3(0.1, 0.9, 0.1), 0.4, 20)),
        Sphere(Vec3(1, 0.2, 3), 0.2, Material(Vec3(0.9, 0.1, 0.1), 0.2, 15))
    };

    std::vector<Light> lights = {
        Light{Vec3(-5, 5, -5), Vec3(1, 1, 1), 1.0, 0.2},
        Light{Vec3(5, 3, -3), Vec3(0.8, 0.8, 1), 0.8, 0.1},
        // Additional light
        Light{Vec3(0, 5, 5), Vec3(1, 0.9, 0.8), 0.6, 0.15}
    };

    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double u = (x + 0.5) / width * 2 - 1;
            double v = -((y + 0.5) / height * 2 - 1);

            Vec3 ray_dir = camera_dir * focal_length +
                           camera_right * (u * viewport_width * 0.5) +
                           camera_up * (v * viewport_height * 0.5);

            Ray ray(camera_pos, ray_dir.normalize());

            Vec3 color = get_color(ray, spheres, lights, max_bounces);

            int index = (y * width + x) * 3;
            image[index] = static_cast<unsigned char>(std::min(color.x * 255.0, 255.0));
            image[index + 1] = static_cast<unsigned char>(std::min(color.y * 255.0, 255.0));
            image[index + 2] = static_cast<unsigned char>(std::min(color.z * 255.0, 255.0));
        }
    }

    stbi_write_png("output.png", width, height, 3, image.data(), width * 3);
    std::cout << "Image saved as output.png" << std::endl;

    return 0;
}
