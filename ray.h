#pragma once

#include "vec3.h"

struct Ray {
    Vec3 origin, direction;
    __host__ __device__ Ray(const Vec3& origin, const Vec3& direction) : origin(origin), direction(direction) {}
};
