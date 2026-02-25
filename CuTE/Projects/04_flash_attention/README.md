# Project 04: FlashAttention - Tiled Attention Mechanism

## Objective

Implement the FlashAttention algorithm using CuTe. This project teaches:
- Attention mechanism fundamentals
- Tiled/Blocked attention computation
- Online softmax for numerical stability
- SRAM-efficient attention (IO-aware algorithm)

## Theory

### The Attention Problem

Standard attention computes:

```
Attention(Q, K, V) = softmax(QK^T / √d) × V
```

Where:
- `Q` (Query): `(batch, seq_len, d)`
- `K` (Key): `(batch, seq_len, d)`
- `V` (Value): `(batch, seq_len, d)`
- Output: `(batch, seq_len, d)`

The naive implementation has **O(n²)** memory complexity - it materializes the full `(seq_len × seq_len)` attention matrix.

### FlashAttention Insight

FlashAttention computes attention in **tiles** that fit in SRAM, avoiding HBM accesses for the attention matrix:

```
Algorithm:
1. Partition Q into tiles of size Bc
2. Partition K, V into tiles of size Br
3. For each Q tile:
   a. Initialize running max and sum
   b. For each K, V tile:
      - Compute Q_tile × K_tile^T (attention scores)
      - Update running max (online softmax)
      - Accumulate weighted values
   c. Write output tile
```

### Online Softmax for Attention

FlashAttention uses a clever online softmax that updates statistics incrementally:

```cpp
// For each tile, update running statistics
m_new = max(m_old, max(scores))           // New max
P' = exp(scores - m_new)                   // Rescaled probabilities
ℓ_new = exp(m_old - m_new) * ℓ_old + sum(P')  // Rescaled sum
O_new = (ℓ_old * O_old * exp(m_old - m_new) + P' × V) / ℓ_new
```

## Your Task

### Step 1: Understand the Data Layout

```cpp
// Input tensors (row-major, batch-major layout)
// Q: [batch, seq_len, d]
// K: [batch, seq_len, d]  
// V: [batch, seq_len, d]
// Output: [batch, seq_len, d]

// For simplicity, we'll implement single-batch first
// Q: [seq_len, d], K: [seq_len, d], V: [seq_len, d]
```

### Step 2: Implement Tiled Attention

```cpp
__global__ void flash_attention_cute(
    float* Q, float* K, float* V, float* O,
    int seq_len, int d) {
    
    // Tile configuration
    constexpr int Br = 64;  // Row tile (queries)
    constexpr int Bc = 64;  // Column tile (keys/values)
    
    // Each block computes one row tile of output
    int tile_idx = blockIdx.x;
    int q_start = tile_idx * Br;
    
    // TODO: Initialize running statistics
    // float m_i = -infinity;  // Running max
    // float l_i = 0.0f;        // Running sum
    // float o_i[Br][d] = {0};  // Accumulator
    
    // TODO: Loop over K, V tiles
    // for (int k_start = 0; k_start < seq_len; k_start += Bc) {
    //     // Load Q tile, K tile, V tile
    //     
    //     // Compute attention scores: S = Q_tile × K_tile^T
    //     
    //     // Scale by 1/sqrt(d)
    //     
    //     // Update running max and rescale
    //     
    //     // Compute probabilities: P = exp(S - m_new)
    //     
    //     // Update running sum
    //     
    //     // Accumulate: O = P × V
    // }
    
    // TODO: Normalize output by running sum
    // Write output tile
}
```

### Step 3: Key CuTe Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `make_layout(seq_len, d)` | 2D tensor layout | For Q, K, V matrices |
| `tiled_copy()` | Efficient tile loading | From GMEM to SMEM |
| `cute::gemm()` | Matrix multiply | For Q × K^T and P × V |

### Step 4: Implementation Hints

#### Loading Tiles

```cpp
// Load Q tile into shared memory
__shared__ float Q_tile[Br][d];
__shared__ float K_tile[Bc][d];
__shared__ float V_tile[Bc][d];

// Cooperative loading
for (int i = tid; i < Br * d; i += blockDim.x) {
    int r = i / d;
    int c = i % d;
    Q_tile[r][c] = Q[(q_start + r) * d + c];
}
```

#### Computing Attention Scores

```cpp
// For each query position i in tile and key position j:
float score = 0.0f;
for (int k = 0; k < d; k++) {
    score += Q_tile[i][k] * K_tile[j][k];
}
score *= rsqrtf((float)d);  // Scale by 1/sqrt(d)
```

## Exercises

### Exercise 1: Basic FlashAttention (Required)

Implement tiled attention with:
- Tile size: 64 queries per block
- Inner loop over 64 keys/values
- Online softmax for numerical stability
- Single batch for simplicity

### Exercise 2: Multi-Block Attention (Challenge)

Extend to handle:
- Multiple batches
- Variable sequence lengths
- Causal masking (for decoder attention)

### Exercise 3: Fused Kernels (Advanced)

Add fused operations:
- Fused bias: `QK^T + bias`
- Fused scaling: Pre-scale Q or K
- Fused output: `O × W_O` (output projection)

## Verification

Your implementation is correct if:

```
[PASS] FlashAttention: Output matches reference (max error: 0.001953)
Output shape: [16, 64]
Sample output (first row):
  0.123  -0.456   0.789  -0.012   0.345  -0.678   0.901  -0.234
```

## Memory Complexity Comparison

| Algorithm | HBM Reads | HBM Writes | SRAM Usage |
|-----------|-----------|------------|------------|
| Naive | O(n²) | O(n²) | O(1) |
| FlashAttention | O(n) | O(n) | O(Br × Bc) |

For `seq_len = 4096`:
- Naive: 16M attention scores in HBM
- FlashAttention: ~4K tiles in SRAM

## Common Pitfalls

1. **Forgetting to scale by 1/√d**: Causes softmax saturation
2. **Incorrect online softmax update**: Must rescale old accumulator
3. **Missing synchronization**: `__syncthreads()` after tile loads
4. **Wrong index calculation**: Careful with `q_start + i` vs just `i`

## Mathematical Background

### Attention Formula

```
Attention(Q, K, V) = softmax(QK^T / √d) × V
```

Expanded:

```
A[i,j] = exp(Q[i] · K[j] / √d) / Σ_k exp(Q[i] · K[k] / √d)
O[i] = Σ_j A[i,j] × V[j]
```

### Online Softmax Update

```
m' = max(m, x_new)
ℓ' = ℓ × exp(m - m') + exp(x_new - m')
```

## Performance Notes

| seq_len | Naive Memory | FlashAttention Memory | Speedup |
|---------|--------------|----------------------|---------|
| 512 | 1 MB | 16 KB | 2x |
| 4096 | 64 MB | 16 KB | 8x |
| 16384 | 1 GB | 16 KB | 20x |

## Next Steps

After completing this project:
1. Try [Project 05: FlashInfer](../05_flashinfer/) for variable sequences
2. Implement multi-head attention
3. Add causal masking for autoregressive models
4. Study the [FlashAttention paper](https://arxiv.org/abs/2205.14135)

---

**Ready to revolutionize attention? Open `flash_attention.cu` and start implementing!**
