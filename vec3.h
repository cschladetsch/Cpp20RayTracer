#pragma once

#include <cuda_runtime.h> // Add this line

// Vec3 struct for RGB and vector math
struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ Vec3 operator*(float t) const { return Vec3(x * t, y * t, z * t); }
    __host__ __device__ Vec3 operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
    __host__ __device__ Vec3 operator/(float t) const { return Vec3(x / t, y / t, z / t); }
    __host__ __device__ float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    __host__ __device__ Vec3 normalize() const {
        float length = sqrtf(x * x + y * y + z * z);
        return Vec3(x / length, y / length, z / length);
    }
    __host__ __device__ Vec3 reflect(const Vec3& normal) const {
        return *this - normal * (2.0f * this->dot(normal));
    }
};



