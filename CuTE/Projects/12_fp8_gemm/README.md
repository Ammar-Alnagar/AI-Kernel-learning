# Project 12: FP8 GEMM

## Objective

Implement GEMM with FP8 (E4M3/E5M2) inputs. This project teaches:
- FP8 data formats (E4M3, E5M2)
- FP8 matrix multiplication
- Conversion between FP8 and FP32
- Mixed-precision accumulation

## Theory

### FP8 Formats

Two common FP8 formats:
- **E4M3**: 4 exponent, 3 mantissa (max ~448)
- **E5M2**: 5 exponent, 2 mantissa (max ~57344)

### FP8 GEMM

```
C_fp32 = A_fp8 × B_fp8  (accumulate in FP32)
```

## Your Task

Implement FP8 GEMM with:
- FP8 storage (uint8_t)
- FP8 to FP32 conversion
- FP32 accumulation

---

**Ready for FP8? Open `fp8_gemm.cu`!**
