#pragma once

#include "vec3.h"

struct Material {
    Vec3 color;
    float reflectivity;
    float specularity;

    __host__ __device__ Material() : color(Vec3()), reflectivity(0), specularity(0) {}
    __host__ __device__ Material(const Vec3& c, float r, float s) : color(c), reflectivity(r), specularity(s) {}
};

