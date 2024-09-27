#!/bin/sh

g++ -std=c++20 -O3 -pthread raytracer.cpp -o raytracer && ./raytracer && start ouput.png

