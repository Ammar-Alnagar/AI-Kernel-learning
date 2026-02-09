#!/bin/bash

# Build script for Module 1: Layouts and Tensors
# This script builds module 1 using CMake

set -e  # Exit on any error

echo "Building Module 1: Layouts and Tensors..."

# Navigate to module directory
cd "$(dirname "$0")"

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the module
echo "Building module..."
make -j$(nproc)

echo "Module 1 build completed successfully!"
echo ""
echo "To run the module, execute: ./module1"