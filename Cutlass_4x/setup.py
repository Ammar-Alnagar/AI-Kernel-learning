#!/usr/bin/env python3
"""
CUTLASS 4.x Python Setup Verification Script

Checks:
  1. nvidia-cutlass-dsl package installation
  2. CUDA Toolkit version
  3. GPU detection and SM architecture
  4. Peak TFLOPS calculation for your GPU

Run this first before starting any exercises.
"""

import subprocess
import sys
import re
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GPUInfo:
    name: str
    sm: int
    sm_name: str
    multiprocessors: int
    clock_mhz: int
    peak_fp16_tflops: float
    peak_fp32_tflops: float
    peak_fp64_tflops: float
    memory_bandwidth_gbps: float
    memory_size_gb: int


# GPU specifications (SM, MPs, peak FP16 TFLOPS per MP per cycle, memory bandwidth)
# FP16 TFLOPS = MPs × clock(GHz) × 2 (FP16 FMA) × 64 (ops/cycle/MP for Tensor Cores)
GPU_SPECS = {
    # Ampere (SM80)
    "A100": {"sm": 80, "mps": 108, "fp16_factor": 64, "bandwidth": 1555, "memory": 40},
    "A100-80GB": {"sm": 80, "mps": 108, "fp16_factor": 64, "bandwidth": 2039, "memory": 80},
    "A800": {"sm": 80, "mps": 108, "fp16_factor": 64, "bandwidth": 1555, "memory": 80},
    "RTX 3090": {"sm": 86, "mps": 82, "fp16_factor": 64, "bandwidth": 936, "memory": 24},
    "RTX 3090 Ti": {"sm": 86, "mps": 84, "fp16_factor": 64, "bandwidth": 1008, "memory": 24},
    "RTX 4090": {"sm": 89, "mps": 128, "fp16_factor": 64, "bandwidth": 1008, "memory": 24},
    "RTX 4080": {"sm": 89, "mps": 76, "fp16_factor": 64, "bandwidth": 717, "memory": 16},
    
    # Hopper (SM90)
    "H100": {"sm": 90, "mps": 114, "fp16_factor": 64, "bandwidth": 2000, "memory": 80},
    "H100-80GB": {"sm": 90, "mps": 114, "fp16_factor": 64, "bandwidth": 2000, "memory": 80},
    "H200": {"sm": 90, "mps": 132, "fp16_factor": 64, "bandwidth": 4800, "memory": 141},
    "H800": {"sm": 90, "mps": 114, "fp16_factor": 64, "bandwidth": 2000, "memory": 80},
    "GH200": {"sm": 90, "mps": 132, "fp16_factor": 64, "bandwidth": 4200, "memory": 96},
    
    # Blackwell (SM100/SM103)
    "B100": {"sm": 100, "mps": 144, "fp16_factor": 128, "bandwidth": 8000, "memory": 80},
    "B200": {"sm": 100, "mps": 144, "fp16_factor": 128, "bandwidth": 8000, "memory": 180},
    "GB200": {"sm": 100, "mps": 144, "fp16_factor": 128, "bandwidth": 8000, "memory": 180},
    "RTX 5090": {"sm": 103, "mps": 128, "fp16_factor": 128, "bandwidth": 1792, "memory": 32},  # Thor-like
}

SM_NAMES = {
    80: "Ampere",
    86: "Ampere (AD102)",
    89: "Ada Lovelace",
    90: "Hopper",
    100: "Blackwell",
    103: "Blackwell (Thor)",
    110: "Blackwell (SM110 alias)",
}


def run_cmd(cmd: str) -> Tuple[bool, str]:
    """Run a shell command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def check_cutlass_install() -> Tuple[bool, str]:
    """Check if nvidia-cutlass-dsl is installed."""
    success, output = run_cmd(
        "python -c \"import cutlass; print(cutlass.__version__)\""
    )
    if success:
        return True, f"nvidia-cutlass-dsl installed (version {output})"
    return False, "nvidia-cutlass-dsl NOT found"


def check_cuda_version() -> Tuple[bool, str]:
    """Check CUDA Toolkit version."""
    success, output = run_cmd("nvcc --version | grep release")
    if success:
        # Extract version like "V12.1.105"
        match = re.search(r"V(\d+\.\d+)", output)
        if match:
            version = match.group(1)
            major = int(version.split(".")[0])
            if major >= 12:
                return True, f"CUDA Toolkit: {version} ✓"
            return False, f"CUDA Toolkit {version} (need >= 12.x for optimal Blackwell support)"
    return False, "CUDA Toolkit NOT found (nvcc not in PATH)"


def detect_gpu() -> Tuple[bool, Optional[GPUInfo]]:
    """Detect GPU using nvidia-smi and query compute capability."""
    # Get GPU name
    success, name_output = run_cmd(
        "nvidia-smi --query-gpu=name --format=csv,noheader"
    )
    if not success:
        return False, None
    
    gpu_name = name_output.split("\n")[0].strip()
    
    # Get compute capability
    success, cc_output = run_cmd(
        "nvidia-smi --query-gpu=compute_cap --format=csv,noheader"
    )
    if not success:
        return False, None
    
    cc = cc_output.split("\n")[0].strip().replace(".", "")
    sm = int(cc)
    
    # Get multiprocessor count
    success, mp_output = run_cmd(
        "nvidia-smi --query-gpu=multiprocessor_count --format=csv,noheader"
    )
    mps = int(mp_output.split("\n")[0].strip()) if success else 0
    
    # Get clock speed
    success, clock_output = run_cmd(
        "nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader"
    )
    clock_mhz = int(clock_output.split("\n")[0].strip()) if success else 1500
    
    # Get memory info
    success, mem_output = run_cmd(
        "nvidia-smi --query-gpu=memory.total --format=csv,noheader"
    )
    memory_gb = int(mem_output.split("\n")[0].strip().split()[0]) // 1024 if success else 0
    
    # Get bandwidth (from memory info)
    success, bw_output = run_cmd(
        "nvidia-smi --query-gpu=memory.clock --format=csv,noheader"
    )
    # Approximate bandwidth calculation if not available
    bandwidth_gbps = 0
    
    # Try to match with known GPU specs
    gpu_info = None
    for known_name, specs in GPU_SPECS.items():
        if known_name.lower() in gpu_name.lower() or gpu_name.lower() in known_name.lower():
            # Calculate peak TFLOPS
            # For Tensor Core FP16: MPs × clock(GHz) × fp16_factor
            clock_ghz = clock_mhz / 1000.0
            peak_fp16 = mps * clock_ghz * specs["fp16_factor"] / 1000.0  # TFLOPS
            peak_fp32 = peak_fp16 / 8.0  # FP32 CUDA cores typically 1/8 of Tensor Core throughput
            peak_fp64 = peak_fp32 / 2.0  # FP64 typically 1/2 of FP32
            
            gpu_info = GPUInfo(
                name=gpu_name,
                sm=sm,
                sm_name=SM_NAMES.get(sm, f"Unknown (SM{sm})"),
                multiprocessors=mps,
                clock_mhz=clock_mhz,
                peak_fp16_tflops=round(peak_fp16, 1),
                peak_fp32_tflops=round(peak_fp32, 1),
                peak_fp64_tflops=round(peak_fp64, 1),
                memory_bandwidth_gbps=specs["bandwidth"],
                memory_size_gb=memory_gb if memory_gb > 0 else specs["memory"],
            )
            break
    
    # If no exact match, create generic info
    if gpu_info is None:
        clock_ghz = clock_mhz / 1000.0
        # Estimate fp16_factor based on SM
        fp16_factor = 64 if sm < 100 else 128
        peak_fp16 = mps * clock_ghz * fp16_factor / 1000.0
        
        gpu_info = GPUInfo(
            name=gpu_name,
            sm=sm,
            sm_name=SM_NAMES.get(sm, f"Unknown (SM{sm})"),
            multiprocessors=mps,
            clock_mhz=clock_mhz,
            peak_fp16_tflops=round(peak_fp16, 1),
            peak_fp32_tflops=round(peak_fp16 / 8.0, 1),
            peak_fp64_tflops=round(peak_fp16 / 16.0, 1),
            memory_bandwidth_gbps=0,  # Unknown
            memory_size_gb=memory_gb,
        )
    
    return True, gpu_info


def check_cute_dsl() -> Tuple[bool, str]:
    """Check if CuTe DSL is available."""
    success, _ = run_cmd("python -c \"import cutlass.cute; print('OK')\"")
    if success:
        return True, "CuTe DSL available ✓"
    return False, "CuTe DSL NOT found"


def print_checkmark(success: bool, message: str):
    """Print a checkmark or X based on success."""
    symbol = "✓" if success else "✗"
    status = "\033[92m" if success else "\033[91m"  # Green or Red
    reset = "\033[0m"
    print(f"{status}{symbol}{reset} {message}")


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def main():
    print_header("CUTLASS 4.x Python Setup Verification")
    
    all_passed = True
    
    # Check 1: nvidia-cutlass-dsl
    print("\n[1/4] Checking nvidia-cutlass-dsl installation...")
    success, msg = check_cutlass_install()
    print_checkmark(success, msg)
    all_passed = all_passed and success
    
    # Check 2: CuTe DSL
    print("\n[2/4] Checking CuTe DSL availability...")
    success, msg = check_cute_dsl()
    print_checkmark(success, msg)
    all_passed = all_passed and success
    
    # Check 3: CUDA Toolkit
    print("\n[3/4] Checking CUDA Toolkit version...")
    success, msg = check_cuda_version()
    print_checkmark(success, msg)
    all_passed = all_passed and success
    
    # Check 4: GPU Detection
    print("\n[4/4] Detecting GPU...")
    success, gpu_info = detect_gpu()
    
    if success and gpu_info:
        print_checkmark(True, f"GPU detected: {gpu_info.name}")
        print(f"\n  Architecture:     {gpu_info.sm_name} (SM{gpu_info.sm})")
        print(f"  Multiprocessors:  {gpu_info.multiprocessors}")
        print(f"  SM Clock:         {gpu_info.clock_mhz} MHz")
        print(f"  Memory:           {gpu_info.memory_size_gb} GB")
        print(f"  Memory Bandwidth: {gpu_info.memory_bandwidth_gbps} GB/s")
        
        print(f"\n  Peak TFLOPS (Tensor Core):")
        print(f"    FP16:  {gpu_info.peak_fp16_tflops:7.1f} TFLOPS")
        print(f"    FP32:  {gpu_info.peak_fp32_tflops:7.1f} TFLOPS")
        print(f"    FP64:  {gpu_info.peak_fp64_tflops:7.1f} TFLOPS")
        
        # Recommendations based on GPU
        print(f"\n  Recommended Exercises:")
        if gpu_info.sm >= 100:
            print("    → Module 07: Blackwell Features (PDL, tcgen05, FP4)")
            print("    → Module 05: Blockscaled FP4 GEMM (SM103)")
        elif gpu_info.sm >= 90:
            print("    → Module 06: Attention Kernels (Hopper FMHA)")
            print("    → Module 05: FP8 GEMM (E4M3/E5M2)")
        else:
            print("    → Module 01-04: Core ops, autotuning, epilogue fusion")
            print("    → Module 05: INT8 GEMM (Ampere supports INT8 Tensor Core)")
    else:
        print_checkmark(False, "GPU detection failed")
        all_passed = False
    
    # Final summary
    print_header("Summary")
    
    if all_passed:
        print("\n\033[92m✓ All checks passed! You're ready to start.\033[0m")
        print("\nNext steps:")
        print("  1. Set environment variable for JIT cache (recommended):")
        print("     export CUTE_DSL_CACHE_DIR=$HOME/.cutlass_cache")
        print("\n  2. Start with Module 01:")
        print("     cd 01_high_level_ops")
        print("     python ex01_gemm_basic_FILL_IN.py")
    else:
        print("\n\033[91m✗ Some checks failed. Please install missing components.\033[0m")
        print("\nInstall nvidia-cutlass-dsl:")
        print("  pip install nvidia-cutlass-dsl")
        print("\nInstall CUDA Toolkit:")
        print("  https://developer.nvidia.com/cuda-toolkit")
        sys.exit(1)


if __name__ == "__main__":
    main()
