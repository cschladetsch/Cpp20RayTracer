#include <iostream>
#include <fstream>
#include "vec3.h"
#include "bmp_writer.h"

#pragma pack(push, 1)
struct BMPFileHeader {
    uint16_t fileType{0x4D42};  // File type always BM which is 0x4D42
    uint32_t fileSize{0};       // Size of the file (in bytes)
    uint16_t reserved1{0};      // Reserved, always 0
    uint16_t reserved2{0};      // Reserved, always 0
    uint32_t offsetData{54};    // Start position of pixel data (54 bytes)
};

// Bitmap info header
struct BMPInfoHeader {
    uint32_t size{40};          // Size of this header (in bytes)
    int32_t width{0};           // width of bitmap in pixels
    int32_t height{0};          // height of bitmap in pixels
    uint16_t planes{1};         // No. of planes for the target device, must be 1
    uint16_t bitCount{24};      // No. of bits per pixel
    uint32_t compression{0};    // 0 or 3 - uncompressed
    uint32_t sizeImage{0};      // 0 - for uncompressed images
    int32_t xPixelsPerMeter{0};
    int32_t yPixelsPerMeter{0};
    uint32_t colorsUsed{0};     // No. color indices in the color table. Use 0 for the max number of colors
    uint32_t colorsImportant{0}; // No. of important color indices. 0 means all are important
};
#pragma pack(pop)

// Write the BMP file
void writeBMP(const char* filename, int width, int height, Vec3* framebuffer) {
    int paddingAmount = (4 - (width * 3) % 4) % 4; // BMP rows must be padded to be a multiple of 4 bytes

    BMPFileHeader fileHeader;
    BMPInfoHeader infoHeader;

    fileHeader.fileSize = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + (width * height * 3) + (paddingAmount * height);
    infoHeader.width = width;
    infoHeader.height = height;

    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file) {
        std::cerr << "Could not write BMP file\n";
        return;
    }

    // Write the headers
    file.write((const char*)&fileHeader, sizeof(fileHeader));
    file.write((const char*)&infoHeader, sizeof(infoHeader));

    // Write the pixel data (bottom-up for BMP format)
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            Vec3 color = framebuffer[j * width + i];
            uint8_t r = static_cast<uint8_t>(255.99f * color.x);
            uint8_t g = static_cast<uint8_t>(255.99f * color.y);
            uint8_t b = static_cast<uint8_t>(255.99f * color.z);

            // BMP uses BGR format
            file.write((const char*)&b, 1);
            file.write((const char*)&g, 1);
            file.write((const char*)&r, 1);
        }
        // Add padding for BMP format
        file.write("\0\0\0", paddingAmount);
    }

    file.close();
}

