#!/bin/bash

# Build script for Cutlass Learning Modules
# This script builds all modules using CMake

set -e  # Exit on any error

echo "Building all Cutlass Learning Modules..."

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build all modules
echo "Building all modules..."
make -j$(nproc) all_modules

echo "Build completed successfully!"
echo ""
echo "Built executables are located in:"
find . -name "module*" -type f -executable