#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <execution>
#include <fstream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct Vec3 {
    double x, y, z;
    Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(double d) const { return Vec3(x * d, y * d, z * d); }
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
    Sphere(const Vec3& c, double r, const Vec3& col) : center(c), radius(r), color(col) {}
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

Vec3 get_color(const Ray& ray, const Sphere& sphere) {
    double t_sphere = std::numeric_limits<double>::infinity();
    double t_ground;
    bool hit_sphere = sphere.intersect(ray, t_sphere);
    bool hit_ground = check_ground_intersection(ray, t_ground);

    if (!hit_sphere && !hit_ground) {
        return sky_color(ray);
    }

    if (hit_sphere && (!hit_ground || t_sphere < t_ground)) {
        Vec3 hit_point = ray.origin + ray.direction * t_sphere;
        Vec3 normal = (hit_point - sphere.center).normalize();
        double diffuse = std::max(0.0, normal.dot(Vec3(0, 1, 0)));
        return sphere.color * diffuse;
    }

    // Ground plane with checkered pattern
    Vec3 hit_point = ray.origin + ray.direction * t_ground;
    int check_x = static_cast<int>(std::floor(hit_point.x)) % 2;
    int check_z = static_cast<int>(std::floor(hit_point.z)) % 2;
    if ((check_x + check_z) % 2 == 0) {
        return Vec3(0.8, 0.8, 0.8);  // Light square
    } else {
        return Vec3(0.3, 0.3, 0.3);  // Dark square
    }
}

int main() {
    const int width = 800;
    const int height = 600;
    const double aspect_ratio = static_cast<double>(width) / height;

    std::vector<unsigned char> image(width * height * 3);

    Vec3 camera_pos(0, 2, -5);
    Vec3 camera_dir = Vec3(0, 0, 1).normalize();
    Vec3 camera_up = Vec3(0, 1, 0);
    Vec3 camera_right = camera_up.cross(camera_dir).normalize();

    Sphere sphere(Vec3(0, 1, 0), 1, Vec3(1, 0, 0));  // Red sphere at (0, 1, 0) with radius 1

    std::vector<int> rows(height);
    std::iota(rows.begin(), rows.end(), 0);

    std::for_each(std::execution::par, rows.begin(), rows.end(), [&](int y) {
        for (int x = 0; x < width; ++x) {
            double u = (x + 0.5) / width * 2 - 1;
            double v = -((y + 0.5) / height * 2 - 1);

            Vec3 ray_dir = camera_dir + camera_right * u * aspect_ratio + camera_up * v;
            Ray ray(camera_pos, ray_dir.normalize());

            Vec3 color = get_color(ray, sphere);

            int index = (y * width + x) * 3;
            image[index] = static_cast<unsigned char>(std::min(color.x * 255.0, 255.0));
            image[index + 1] = static_cast<unsigned char>(std::min(color.y * 255.0, 255.0));
            image[index + 2] = static_cast<unsigned char>(std::min(color.z * 255.0, 255.0));
        }
    });

    stbi_write_png("output.png", width, height, 3, image.data(), width * 3);
    std::cout << "Image saved as output.png" << std::endl;

    return 0;
}
