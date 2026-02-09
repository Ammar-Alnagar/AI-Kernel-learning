#!/bin/bash

# Build script for Module 2: Tiled Copy
# This script builds module 2 using CMake

set -e  # Exit on any error

echo "Building Module 2: Tiled Copy..."

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

echo "Module 2 build completed successfully!"
echo ""
echo "To run the module, execute: ./module2"