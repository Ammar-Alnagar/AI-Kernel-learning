# Project 09: Vectorized Copy Kernel

## Objective

Implement high-bandwidth memory copy using vectorized loads and stores. This project teaches:
- Vectorized memory operations (float4, uint4)
- 128-bit load/store instructions
- Memory coalescing optimization
- Bandwidth-bound kernel optimization

## Theory

### Vectorized Loads

GPU memory subsystem can transfer 128 bits per transaction:

```cpp
// Scalar loads (32-bit each)
float a = src[i];
float b = src[i+1];
float c = src[i+2];
float d = src[i+3];

// Vectorized load (128-bit)
float4 v = reinterpret_cast<float4*>(src)[i/4];
```

### Memory Coalescing

Consecutive threads should access consecutive memory:

```
Thread 0: loads [0:3]
Thread 1: loads [4:7]
Thread 2: loads [8:11]
...
```

## Your Task

Implement vectorized copy with:
- float4 loads/stores
- Proper alignment handling
- Remainder handling for non-multiple-of-4 sizes

---

**Ready to maximize bandwidth? Open `vectorized_copy.cu`!**
