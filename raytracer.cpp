#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <fstream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct Vec3 {
    double x, y, z;
    Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
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

struct Sphere {
    Vec3 center;
    double radius;
    Vec3 color;
    bool reflective;
    Sphere(const Vec3& c, double r, const Vec3& col, bool refl) : center(c), radius(r), color(col), reflective(refl) {}
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

Vec3 get_color(const Ray& ray, const std::vector<Sphere>& spheres, int bounces) {
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
        
        if (hit_sphere->reflective) {
            Vec3 reflected = (ray.direction - normal * 2 * ray.direction.dot(normal)).normalize();
            return get_color(Ray(hit_point + normal * 0.001, reflected), spheres, bounces - 1);
        } else {
            double diffuse = std::max(0.0, normal.dot(Vec3(0, 1, 0)));
            return hit_sphere->color * diffuse;
        }
    }

    if (hit_ground) {
        Vec3 hit_point = ray.origin + ray.direction * t_ground;
        int check_x = static_cast<int>(std::floor(hit_point.x)) % 2;
        int check_z = static_cast<int>(std::floor(hit_point.z)) % 2;
        if ((check_x + check_z) % 2 == 0) {
            return Vec3(0.8, 0.8, 0.8);  // Light square
        } else {
            return Vec3(0.3, 0.3, 0.3);  // Dark square
        }
    }

    return sky_color(ray);
}
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <number_of_bounces>" << std::endl;
        return 1;
    }

    int max_bounces = std::stoi(argv[1]);

    const int width = 800;
    const int height = 600;
    const double aspect_ratio = static_cast<double>(width) / height;

    std::vector<unsigned char> image(width * height * 3);

    Vec3 camera_pos(0, 2, -5);
    Vec3 camera_dir = Vec3(0, 0, 1).normalize();
    Vec3 camera_up = Vec3(0, 1, 0);
    Vec3 camera_right = camera_up.cross(camera_dir).normalize();

    std::vector<Sphere> spheres = {
        Sphere(Vec3(0, 1, 0), 2, Vec3(1, 0, 0), true),    // Large red reflective sphere
        Sphere(Vec3(-2.5, 0.5, 2), 0.5, Vec3(0, 1, 0), false), // Smaller green non-reflective sphere
        Sphere(Vec3(2.5, 0.5, -2), 0.5, Vec3(0, 0, 1), true)   // Smaller blue reflective sphere
    };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double u = (x + 0.5) / width * 2 - 1;
            double v = -((y + 0.5) / height * 2 - 1);

            Vec3 ray_dir = camera_dir + camera_right * u * aspect_ratio + camera_up * v;
            Ray ray(camera_pos, ray_dir.normalize());

            Vec3 color = get_color(ray, spheres, max_bounces);

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
