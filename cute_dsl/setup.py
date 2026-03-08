#!/usr/bin/env python3
"""
CuTe DSL Environment Validation Script

Validates:
  ✓ nvidia-cutlass-dsl installation
  ✓ GPU detection and compute capability
  ✓ CUDA toolkit availability
  ✓ Roofline peak TFLOPS for detected GPU
  ✓ Hello-world layout creation and print

Run: python setup.py
"""

import re
import subprocess
import sys
from typing import Optional, Tuple


def print_header(text: str):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def print_check(mark: str, text: str):
    print(f"  {mark} {text}")


def validate_cutlass_dsl() -> Optional[str]:
    """Check if nvidia-cutlass-dsl is installed and return version."""
    try:
        import cutlass

        version = getattr(cutlass, "__version__", "unknown")
        return version
    except ImportError:
        return None


def validate_cuda_toolkit() -> Optional[str]:
    """Check CUDA toolkit version via nvcc."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            # Parse: Cuda compilation tools, release 12.4, V12.4.131
            match = re.search(r"release (\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def detect_gpu() -> Optional[Tuple[str, int]]:
    """Detect GPU name and compute capability using nvidia-smi or torch."""
    # Try torch first (most reliable)
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            sm_version = cc[0] * 10  # e.g., (9, 0) -> 90
            return device_name, sm_version
    except ImportError:
        pass

    # Fallback to nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines:
                parts = lines[0].split(", ")
                device_name = parts[0]
                cc_str = parts[1].replace(".", "")  # "9.0" -> "90"
                return device_name, int(cc_str)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def get_roofline_peak(sm_version: int) -> dict:
    """
    Return theoretical peak TFLOPS for different precisions.

    Sources:
      - A100: https://www.nvidia.com/en-us/data-center/a100/
      - H100: https://www.nvidia.com/en-us/data-center/h100/
      - B200: https://www.nvidia.com/en-us/data-center/b200/

    Note: These are dense tensor core TFLOPS (sparse would be 2x).
    """
    roofline_data = {
        # Ampere (SM80)
        80: {
            "gpu_name": "A100 (or similar)",
            "fp16_tensor": 312,  # TFLOPS
            "fp32_tensor": 156,
            "tf32_tensor": 156,
            "fp64": 19.5,
            "memory_bw": 1555,  # GB/s (HBM2e)
        },
        # Hopper (SM90)
        90: {
            "gpu_name": "H100 (or similar)",
            "fp16_tensor": 989,  # TFLOPS (dense)
            "fp32_tensor": 494,
            "tf32_tensor": 494,
            "fp64": 34,
            "memory_bw": 3350,  # GB/s (HBM3)
        },
        # Blackwell (SM100) - estimated
        100: {
            "gpu_name": "B200 (or similar)",
            "fp16_tensor": 2250,  # TFLOPS (dense, estimated)
            "fp32_tensor": 1125,
            "tf32_tensor": 1125,
            "fp64": 50,
            "memory_bw": 8000,  # GB/s (HBM3e, estimated)
        },
        # Ada Lovelace (SM89) - e.g., RTX 4090
        89: {
            "gpu_name": "RTX 4090 (or similar)",
            "fp16_tensor": 330,  # TFLOPS (estimated)
            "fp32_tensor": 165,
            "tf32_tensor": 165,
            "fp64": 1.4,
            "memory_bw": 1008,  # GB/s (GDDR6X)
        },
    }

    return roofline_data.get(
        sm_version,
        {
            "gpu_name": f"Unknown (SM{sm_version})",
            "fp16_tensor": "N/A",
            "fp32_tensor": "N/A",
            "tf32_tensor": "N/A",
            "fp64": "N/A",
            "memory_bw": "N/A",
        },
    )


def test_hello_world_layout():
    """Create and print a simple layout to verify CuTe DSL works."""
    try:
        import cutlass
        import cutlass.cute as cute

        # Create a simple row-major layout: (128, 64) with stride (64, 1)
        layout = cute.make_layout((128, 64), stride=(64, 1))

        # Test cosize
        flat_layout = cute.cosize(layout)

        return True, str(layout), str(flat_layout)
    except Exception as e:
        return False, str(e), None


def main():
    print_header("CuTe DSL Environment Validation")

    all_passed = True

    # 1. Check nvidia-cutlass-dsl installation
    print("\n[1/5] Checking nvidia-cutlass-dsl installation...")
    version = validate_cutlass_dsl()
    if version:
        print_check("✓", f"nvidia-cutlass-dsl installed: v{version}")
    else:
        print_check("✗", "nvidia-cutlass-dsl NOT found")
        print("\n  Install with: pip install nvidia-cutlass-dsl")
        all_passed = False

    # 2. Check CUDA toolkit
    print("\n[2/5] Checking CUDA toolkit...")
    cuda_version = validate_cuda_toolkit()
    if cuda_version:
        print_check("✓", f"CUDA toolkit: {cuda_version}")
    else:
        print_check("⚠", "CUDA toolkit not detected (nvcc not in PATH)")
        print("  This may be OK if using CUDA via torch/cutlass")

    # 3. Detect GPU
    print("\n[3/5] Detecting GPU...")
    gpu_info = detect_gpu()
    if gpu_info:
        device_name, sm_version = gpu_info
        print_check("✓", f"GPU detected: {device_name} (SM{sm_version})")
    else:
        print_check("✗", "No CUDA GPU detected")
        all_passed = False
        sm_version = None

    # 4. Show roofline peak
    print("\n[4/5] Roofline Peak TFLOPS...")
    if sm_version:
        roofline = get_roofline_peak(sm_version)
        print_check("✓", f"GPU class: {roofline['gpu_name']}")
        print("\n  Theoretical Peak Performance:")
        print(f"    FP16 Tensor Core:  {roofline['fp16_tensor']} TFLOPS")
        print(f"    FP32 Tensor Core:  {roofline['fp32_tensor']} TFLOPS")
        print(f"    TF32 Tensor Core:  {roofline['tf32_tensor']} TFLOPS")
        print(f"    FP64:              {roofline['fp64']} TFLOPS")
        print(f"    Memory Bandwidth:  {roofline['memory_bw']} GB/s")

        print(
            f"\n  → Target for Project 01 (Tiled GEMM): >75% of {roofline['fp16_tensor']} TFLOPS"
        )
    else:
        print_check("⚠", "Cannot compute roofline without GPU detection")

    # 5. Hello-world layout test
    print("\n[5/5] Testing CuTe DSL hello-world layout...")
    success, layout_str, flat_str = test_hello_world_layout()
    if success:
        print_check("✓", f"Layout created: {layout_str}")
        print_check("✓", f"Cosized layout: {flat_str}")
    else:
        print_check("✗", f"Layout creation failed: {layout_str}")
        all_passed = False

    # Summary
    print_header("Validation Summary")

    if all_passed:
        print("\n  ✓ All checks passed! Your environment is ready.\n")
        print("  Next steps:")
        print("    cd module_01_layouts")
        print("    python ex01_make_layout_FILL_IN.py")
        print()
        return 0
    else:
        print("\n  ✗ Some checks failed. Please fix the issues above.\n")
        print("  Common fixes:")
        print("    pip install nvidia-cutlass-dsl")
        print(
            "    pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
