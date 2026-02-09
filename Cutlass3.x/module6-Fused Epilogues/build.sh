#!/bin/bash

# Build script for Module 6: Fused Epilogues
# This script builds module 6 using CMake

set -e  # Exit on any error

echo "Building Module 6: Fused Epilogues..."

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

echo "Module 6 build completed successfully!"
echo ""
echo "To run the module, execute: ./module6"