#!/bin/sh

# Default number of bounces
BOUNCES=10

# Check if a command line argument is provided
if [ $# -eq 1 ]; then
    BOUNCES=$1
fi

# Compile and run the raytracer with the specified number of bounces
g++ -std=c++20 -O3 -pthread raytracer.cpp -o raytracer && ./raytracer $BOUNCES && explorer.exe output.png
