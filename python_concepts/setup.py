#!/usr/bin/env python3
"""Validate Python environment for kernel engineering work."""

import sys
import subprocess

def main():
    print("=" * 50)
    print("Python Environment Validation")
    print("=" * 50)
    
    # Python version
    print(f"\nPython version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Try importing torch
    print("\n--- PyTorch ---")
    try:
        import torch
        print(f"torch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("torch not installed. Install with: pip install torch")
    
    # Try importing numpy
    print("\n--- NumPy ---")
    try:
        import numpy as np
        print(f"numpy version: {np.__version__}")
    except ImportError:
        print("numpy not installed. Install with: pip install numpy")
    
    print("\n" + "=" * 50)
    print("Validation complete")
    print("=" * 50)

if __name__ == "__main__":
    main()
