# C++ Raytracer with Progress Spinner

This project is a simple raytracer implemented in C++ that generates a 3D scene with spheres, reflections, and lighting. It includes a progress spinner to show rendering progress and estimated time remaining.

## Features

- Ray tracing with reflections and specular highlights
- Soft shadows for more realistic lighting
- Checkerboard ground plane
- Multiple spheres and light sources
- Progress spinner with completion percentage, elapsed time, and ETA
- Multithreaded rendering using OpenMP

## Prerequisites

To build and run this project, you need:

- A C++17 compliant compiler (e.g., GCC 7+ or Clang 5+)
- OpenMP for parallel processing
- [stb_image_write.h](https://github.com/nothings/stb/blob/master/stb_image_write.h) (single-file library for writing images)

## Building the Project

1. Make sure you have the `stb_image_write.h` file in the same directory as the `raytracer.cpp` file.

2. Compile the project using the following command:

   For GCC:
   ```
   g++ -std=c++17 -O3 -fopenmp raytracer.cpp -o raytracer
   ```

   For Clang:
   ```
   clang++ -std=c++17 -O3 -fopenmp raytracer.cpp -o raytracer
   ```

## Usage

Run the compiled program:

```
./raytracer [max_bounces]
```

- `max_bounces` (optional): Maximum number of light bounces for reflections (default: 10)

The program will start rendering the scene and display a progress spinner in the console. Once complete, it will save the rendered image as `output.png` in the same directory.

## Customization

You can customize the scene by modifying the `main` function in `raytracer.cpp`:

- Adjust the `width` and `height` variables to change the image resolution
- Modify the camera position and field of view
- Add, remove, or modify spheres in the `spheres` vector
- Adjust lighting by modifying the `lights` vector

## Contributing

Contributions to improve the raytracer or add new features are welcome. Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is open-source and available under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- [stb](https://github.com/nothings/stb) for the `stb_image_write.h` library
- The raytracing community for providing educational resources on the topic
