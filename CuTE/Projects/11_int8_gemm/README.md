# Project 11: INT8 GEMM with Fused Dequantization

## Objective

Implement GEMM with INT8 inputs and fused dequantization to FP32 output. This project teaches:
- Quantized arithmetic
- Per-channel dequantization
- Fused kernel operations
- Mixed-precision computation

## Theory

### Quantized GEMM

```
C = dequant(A_int8, scale_A) × dequant(B_int8, scale_B)
  = (A_int8 × scale_A) × (B_int8 × scale_B)
  = (A_int8 × B_int8) × (scale_A × scale_B)
```

### Per-Channel Quantization

Each output channel has its own scale:
```cpp
C[i,j] = sum_k(A_int8[i,k] * B_int8[k,j]) * scale_A[i] * scale_B[j]
```

## Your Task

Implement INT8 GEMM with:
- INT8 input matrices
- Per-channel scale factors
- Fused dequantization in output

---

**Ready for quantization? Open `int8_gemm.cu`!**
