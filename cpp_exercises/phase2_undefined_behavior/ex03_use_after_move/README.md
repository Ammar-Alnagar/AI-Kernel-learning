# ex03: Use-After-Move (UB)

Debug undefined behavior caused by accessing a moved-from object.

## What You Build

A fix for code that accesses a buffer after it has been moved from, leaving it empty.

## What You Observe

The buggy version crashes (null pointer dereference) or triggers sanitizer errors. The fixed version checks validity before access, or better, never accesses moved-from objects.

## CUTLASS/CUDA Mapping

After moving a `DeviceBuffer`, the old object's device pointer is null. Launching a kernel with the old pointer accesses invalid device memory. CUTLASS kernel launchers always use the new owner after move.

## Build Command

```bash
# Buggy version (sanitizer catches UB)
g++ -std=c++20 -O2 -fsanitize=address,undefined -o ex03_buggy exercise.cpp && ./ex03_buggy

# Fixed version
g++ -std=c++20 -O2 -o ex03_fixed solution.cpp && ./ex03_fixed
```
