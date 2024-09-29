#pragma once

#include "vec3.h"

struct Light {
    Vec3 position;
    Vec3 color;
    float intensity;
    float radius;

    __host__ __device__ Light() : position(Vec3()), color(Vec3(1, 1, 1)), intensity(1.0f), radius(0.5f) {}
    __host__ __device__ Light(const Vec3& pos, const Vec3& col, float inten, float rad)
        : position(pos), color(col), intensity(inten), radius(rad) {}
};

