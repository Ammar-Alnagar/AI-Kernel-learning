# SPDX-License-Identifier: Apache-2.0
"""
cutedsl_utils.py — shared runtime helpers for the CuTe DSL curriculum.

Every exercise and project in this curriculum imports from here so that:

* Arch gating is uniform — a kernel written for SM90/SM100 *skips with a clear message*
  instead of crashing on an unsupported GPU.
* Benchmarking uses the official ``cutlass.testing.benchmark`` timing loop.
* Roofline numbers come from one place.

Nothing in here builds IR, so it is safe to import from plain Python.
"""

from __future__ import annotations

import functools
import os
import sys
from dataclasses import dataclass

import cutlass

__all__ = [
    "GpuInfo",
    "dsl_version",
    "gpu_info",
    "gpu_arch",
    "arch_supported",
    "requires_arch",
    "CUTE_DSL_SKIP",
    "benchmark_us",
    "benchmark_ms",
    "sm_count",
    "peak_tflops",
    "peak_bandwidth_gbs",
    "roofline_percent",
    "banner",
    "rule",
    "pass_fail",
    "check",
]

# ---------------------------------------------------------------------------
# Version / arch detection
# ---------------------------------------------------------------------------


def dsl_version() -> str:
    """Version string of the installed ``nvidia-cutlass-dsl`` package."""
    return getattr(cutlass, "__version__", "unknown")


@dataclass(frozen=True)
class GpuInfo:
    name: str
    sm: int  # e.g. 89, 90, 100, 120
    sm_count: int
    total_mem_gb: float

    @property
    def sm_arch(self) -> str:
        return f"sm_{self.sm}"

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.name} (sm_{self.sm}, {self.sm_count} SMs)"


_CACHE: dict[str, GpuInfo] = {}


def gpu_info() -> GpuInfo | None:
    """Probe the current CUDA device. Returns ``None`` when no GPU is visible."""
    if "info" in _CACHE:
        return _CACHE["info"]
    info: GpuInfo | None = None
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info = GpuInfo(
                name=props.name,
                sm=props.major * 10 + props.minor,
                sm_count=props.multi_processor_count,
                total_mem_gb=props.total_memory / (1024**3),
            )
    except Exception:
        info = None
    _CACHE["info"] = info
    return info


def gpu_arch() -> int | None:
    info = gpu_info()
    return None if info is None else info.sm


def sm_count() -> int:
    info = gpu_info()
    return 0 if info is None else info.sm_count


_ARCH_SUPPORTED: dict[int, tuple[int, ...]] = {
    # arch -> minimum nvidia-cutlass-dsl minor version that supports it
    70: (4, 0),
    75: (4, 0),
    80: (4, 0),
    86: (4, 0),
    89: (4, 0),
    90: (4, 0),
    100: (4, 1),
    103: (4, 3),
    110: (4, 5),
    120: (4, 2),
    121: (4, 5),
}


def _dsl_minor() -> tuple[int, int]:
    ver = dsl_version()
    try:
        parts = ver.split(".")
        return int(parts[0]), int(parts[1])
    except Exception:
        return (4, 0)


def arch_supported(*sm: int) -> bool:
    """True when every requested ``sm_XX`` is both present and DSL-supported.

    Pass ``sm=0`` (or nothing) to mean "any GPU".
    """
    targets = [s for s in sm if s]
    if not targets:
        return gpu_info() is not None
    info = gpu_info()
    if info is None:
        return False
    dsl = _dsl_minor()
    return all(t == info.sm and _ARCH_SUPPORTED.get(t, (4, 0)) <= dsl for t in targets)


class _Skip(Exception):
    """Raised by :func:`requires_arch` when the hardware does not match."""


CUTE_DSL_SKIP = "SKIP"


def requires_arch(*sm: int):
    """Skip an example when the running GPU is not one of ``sm``.

    ``@requires_arch(90)`` means SM90 only; ``@requires_arch(90, 100)`` means either.
    Use ``@requires_arch()`` for "any GPU".

    The wrapped function returns the string ``"SKIP"`` and prints an explanatory
    banner instead of raising, so calling scripts stay linear.
    """

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if arch_supported(*sm):
                return fn(*args, **kwargs)
            targets = ", ".join(f"sm_{s}" for s in sm) if sm else "any"
            have = str(gpu_info()) if gpu_info() else "no CUDA device"
            print(f"[{CUTE_DSL_SKIP}] {fn.__name__}: requires {targets}; running on {have}.")
            return CUTE_DSL_SKIP

        wrapper.__cute_dsl_required_arch__ = sm  # type: ignore[attr-defined]
        return wrapper

    return deco


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------


def benchmark_us(
    callable_,
    *,
    warmup_iterations: int = 10,
    iterations: int = 100,
    kernel_arguments=None,
    workspace_generator=None,
    workspace_count: int = 1,
) -> float:
    """Time ``callable_`` (a ``@cute.jit`` host function) and return microseconds."""
    from cutlass.testing import JitArguments, benchmark as _benchmark

    kwargs = dict(
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        workspace_count=workspace_count,
    )
    if workspace_generator is not None:
        kwargs["workspace_generator"] = workspace_generator
    if kernel_arguments is not None:
        kwargs["kernel_arguments"] = kernel_arguments
    return _benchmark(callable_, **kwargs)


def benchmark_ms(*args, **kwargs) -> float:
    """Same as :func:`benchmark_us` but returns milliseconds."""
    return benchmark_us(*args, **kwargs) / 1000.0


# ---------------------------------------------------------------------------
# Roofline
# ---------------------------------------------------------------------------

# (dtype, tensor-core peak TFLOP/s, tensor-core peak TFLOP/s per SM, note)
_PEAK_TABLE: dict[int, dict[str, float]] = {
    80: {"fp16": 312.0, "bf16": 312.0, "fp32": 19.5, "tf32": 156.0, "fp8": 624.0, "int8": 624.0},
    86: {"fp16": 181.0, "bf16": 181.0, "fp32": 11.4, "tf32": 90.5, "fp8": 362.0, "int8": 362.0},
    89: {"fp16": 181.0, "bf16": 181.0, "fp32": 11.4, "tf32": 90.5, "fp8": 362.0, "int8": 362.0},
    90: {"fp16": 989.0, "bf16": 989.0, "fp32": 67.0, "tf32": 495.0, "fp8": 1979.0, "int8": 1979.0},
    100: {"fp16": 1979.0, "bf16": 1979.0, "fp32": 67.0, "tf32": 989.0, "fp8": 3958.0, "int8": 3958.0},
    120: {"fp16": 419.0, "bf16": 419.0, "fp32": 19.6, "tf32": 210.0, "fp8": 838.0, "int8": 838.0},
}

# HBM/GDDR bandwidth in GB/s keyed by arch.
_BW_TABLE: dict[int, float] = {70: 900.0, 75: 616.0, 80: 2039.0, 86: 300.0, 89: 448.0, 90: 3352.0, 100: 8000.0, 120: 1792.0}


def peak_tflops(dtype: str = "fp16") -> float:
    """Dense tensor-core peak TFLOP/s for the running GPU (0.0 if unknown)."""
    sm = gpu_arch()
    if sm is None:
        return 0.0
    for key in (sm, 100 if sm >= 100 else 90, 80):
        if key in _PEAK_TABLE:
            return _PEAK_TABLE[key].get(dtype, 0.0)
    return 0.0


def peak_bandwidth_gbs() -> float:
    """Peak device memory bandwidth in GB/s for the running GPU (0.0 if unknown)."""
    sm = gpu_arch()
    if sm is None:
        return 0.0
    for key in (sm, 100 if sm >= 100 else 90, 80):
        if key in _BW_TABLE:
            return _BW_TABLE[key]
    return 0.0


def roofline_percent(achieved_tflops: float, dtype: str = "fp16") -> float:
    """Achieved fraction of tensor-core peak, in percent."""
    peak = peak_tflops(dtype)
    return 0.0 if peak <= 0 else 100.0 * achieved_tflops / peak


# ---------------------------------------------------------------------------
# Console formatting
# ---------------------------------------------------------------------------

_WIDTH = 68


def banner(title: str, width: int = _WIDTH) -> None:
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def rule(label: str = "", width: int = _WIDTH) -> None:
    if label:
        print(f"-- {label} " + "-" * max(0, width - len(label) - 4))
    else:
        print("-" * width)


def pass_fail(ok: bool, detail: str = "") -> bool:
    tag = "\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m"
    print(f"[{tag}] {detail}" if detail else f"[{tag}]")
    return ok


def check(name: str, got, want, *, tol: float = 0.0) -> bool:
    """Print a single named comparison and return True when it matches."""
    if tol:
        import math

        ok = math.isclose(float(got), float(want), rel_tol=tol, abs_tol=tol)
    else:
        ok = got == want
    return pass_fail(ok, f"{name}: got={got} want={want}")


def _selftest() -> None:  # pragma: no cover
    banner(f"cutedsl_utils self-test — CuTe DSL {dsl_version()}")
    print(f"  device      : {gpu_info()}")
    print(f"  peak fp16   : {peak_tflops('fp16'):.1f} TFLOP/s")
    print(f"  peak memory : {peak_bandwidth_gbs():.0f} GB/s")
    print(f"  sm_90 ok    : {arch_supported(90)}")


if __name__ == "__main__":
    _selftest()