#pragma once

#include "vec3.h"
#include "ray.h"
#include "material.h"

struct Sphere {
    Vec3 center;
    float radius;
    Material material;

    __host__ __device__ Sphere() {}
    __host__ __device__ Sphere(const Vec3& c, float r, const Material& m) : center(c), radius(r), material(m) {}

    __device__ bool intersect(const Ray& ray, float& t) const {
        Vec3 oc = ray.origin - center;
        float b = oc.dot(ray.direction);
        float c = oc.dot(oc) - radius * radius;
        float discriminant = b * b - c;
        if (discriminant < 0) return false;
        float sqrtd = sqrtf(discriminant);
        float root = (-b - sqrtd);
        if (root < EPSILON) {
            root = (-b + sqrtd);
            if (root < EPSILON) return false;
        }
        t = root;
        return true;
    }
};

