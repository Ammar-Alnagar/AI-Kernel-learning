# ex03: unique_ptr Is Move-Only

Debug the misconception that unique_ptr cannot be moved.

## What You Build

A demonstration that unique_ptr transfers ownership via `std::move()`, leaving the source as nullptr.

## What You Observe

After `ptr2 = std::move(ptr1)`, ptr1 becomes nullptr and ptr2 owns the resource. Only ptr2's destructor deletes the memory. No copy occurs — ownership is transferred.

## CUTLASS/CUDA Mapping

CUTLASS uses `unique_ptr<DeviceMemory, Deleter>` for RAII device allocations. Moving the unique_ptr transfers device ownership from the host builder to the kernel launcher. This is zero-overhead — no reference counting like shared_ptr.

## Build Command

```bash
g++ -std=c++20 -O2 -Wall -Wextra -o ex03 exercise.cpp && ./ex03
```
