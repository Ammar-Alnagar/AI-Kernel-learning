# Project 05: FlashInfer - Variable Sequence Length Attention

## Objective

Implement a FlashInfer-style kernel that handles **variable sequence lengths** efficiently using page tables. This project teaches:
- PagedAttention for memory-efficient KV caching
- Variable sequence length handling
- Page table lookups in CUDA kernels
- Multi-request batching with different lengths

## Theory

### The Problem: Variable Length Sequences

In real-world LLM inference:
- Different requests have different sequence lengths
- KV cache grows dynamically during generation
- Naive padding wastes computation and memory

```
Request 1: [Q Q Q Q Q Q Q Q]  (length 8)
Request 2: [Q Q Q Q . . . .]  (length 4, padded to 8)
Request 3: [Q Q Q Q Q . . .]  (length 5, padded to 8)
```

Wasted computation: 9/24 = 37.5%

### FlashInfer Solution: PagedAttention

FlashInfer uses **page tables** to map logical positions to physical memory blocks:

```
Logical KV Cache:           Physical Memory (Pages):
Position 0-3  →  Page 2     Page 0: [K0 V0] [K1 V1] [K2 V2] [K3 V3]
Position 4-7  →  Page 0     Page 1: [K4 V4] [K5 V5] [K6 V6] [K7 V7]
Position 8-11 →  Page 1     Page 2: [K0 V0] [K1 V1] [K2 V2] [K3 V3]
...
```

### Page Table Structure

```cpp
// Page table: maps (batch, logical_position) → physical_block_id
int page_table[batch][max_blocks_per_seq];

// For each request, store:
// - start_page: first page index in page_table
// - seq_len: actual sequence length (no padding)
struct RequestInfo {
    int start_page;
    int seq_len;
    int num_blocks;
};
```

## Your Task

### Step 1: Understand the Data Structures

```cpp
// Input structures
float* Q;           // Queries: [total_queries, d]
float* K_pages;     // Paged K cache: [num_pages, page_size, d]
float* V_pages;     // Paged V cache: [num_pages, page_size, d]
int* page_table;    // Page table: [batch, max_blocks]

// Per-request info
struct RequestInfo {
    int q_start;      // Query start index
    int seq_len;      // Actual sequence length
    int kv_page_start;// First KV page index
};
```

### Step 2: Implement PagedAttention Kernel

```cpp
__global__ void flashinfer_cute_kernel(
    float* Q, float* K_pages, float* V_pages, float* O,
    int* page_table, RequestInfo* req_info,
    int batch, int d, int page_size) {
    
    int req_idx = blockIdx.x;
    if (req_idx >= batch) return;
    
    // Get request info
    RequestInfo req = req_info[req_idx];
    int q_start = req.q_start;
    int seq_len = req.seq_len;
    
    // TODO: For each query in this request:
    // 1. Load Q vector
    // 2. For each KV page:
    //    a. Look up physical page_id from page_table
    //    b. Load K, V vectors from paged memory
    //    c. Compute attention scores
    // 3. Apply softmax and accumulate output
}
```

### Step 3: Key CuTe Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `make_layout(pages, page_size, d)` | 3D paged layout | For K/V pages |
| `gather()` | Indirect memory access | Page table lookup |
| `tiled_partition()` | Work distribution | Among threads |

### Step 4: Implementation Hints

#### Page Table Lookup

```cpp
// Get physical page ID for logical position
int get_physical_page(int page_table_row, int logical_pos, int page_size) {
    int block_idx = logical_pos / page_size;
    int page_id = page_table[page_table_row * max_blocks + block_idx];
    return page_id;
}
```

#### Loading from Paged Memory

```cpp
// Load K vector from paged cache
void load_k_from_page(float* K_out, float* K_pages, 
                      int page_id, int offset_in_page, int d) {
    int base_idx = page_id * page_size * d + offset_in_page * d;
    for (int k = 0; k < d; k++) {
        K_out[k] = K_pages[base_idx + k];
    }
}
```

## Exercises

### Exercise 1: Basic PagedAttention (Required)

Implement attention with:
- Single page size (e.g., 16 keys per page)
- Page table lookup for each KV position
- Variable sequence lengths per request
- No padding in computation

### Exercise 2: Multi-Page Loading (Challenge)

Optimize by:
- Loading multiple pages at once
- Using shared memory for page data
- Coalescing page table lookups

### Exercise 3: Prefix Caching (Advanced)

Implement common prefix optimization:
- Detect shared prefixes across requests
- Cache prefix attention results
- Reuse prefix computations

## Verification

Your implementation is correct if:

```
[PASS] FlashInfer: All requests match reference (max error: 0.001953)
Request 0 (len=12): PASS
Request 1 (len=8):  PASS
Request 2 (len=15): PASS
Request 3 (len=5):  PASS
```

## Memory Efficiency Comparison

| Approach | Memory Usage | Computation Waste |
|----------|--------------|-------------------|
| Padded batch | O(batch × max_len) | Up to 50% |
| PagedAttention | O(total_tokens) | 0% |

For a batch with lengths [100, 50, 75, 200]:
- Padded (max=200): 800 positions
- PagedAttention: 425 positions
- Savings: 47%

## Common Pitfalls

1. **Wrong page index calculation**: `block_idx = logical_pos / page_size`
2. **Missing bounds checks**: Verify `page_id < num_pages`
3. **Incorrect page table stride**: `page_table[req_idx * max_blocks + block_idx]`
4. **Off-by-one in page offset**: `offset = logical_pos % page_size`

## Mathematical Background

PagedAttention computes the same formula as standard attention:

```
Attention(Q, K, V) = softmax(QK^T / √d) × V
```

But K and V are accessed indirectly:

```
K_physical[pos] = K_pages[page_table[logical_block]][offset_in_block]
```

## Performance Notes

| Optimization | Speedup | Use Case |
|--------------|---------|----------|
| Basic PagedAttention | 1x | Variable lengths |
| Multi-page loading | 2-3x | Long sequences |
| Prefix caching | 5-10x | Shared prompts |

## Next Steps

After completing this project:
1. Study the [FlashInfer paper](https://arxiv.org/abs/2401.14238)
2. Implement page table management on CPU
3. Add support for multi-head attention
4. Explore continuous batching for LLM serving

---

**Ready to handle variable sequences? Open `flashinfer.cu` and start implementing!**
