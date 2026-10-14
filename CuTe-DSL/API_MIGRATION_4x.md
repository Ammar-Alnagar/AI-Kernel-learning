# CuTe DSL 4.x API Migration Guide

**Applies to:** `nvidia-cutlass-dsl >= 4.2` (curriculum pinned to **4.8.0**)
**Audience:** anyone holding a CuTe C++ 3.x mental model and needing the current Python DSL.

This guide is the **single source of truth** for the API in this curriculum. Every snippet
below was executed against `nvidia-cutlass-dsl==4.8.0` with CUDA 12/13 drivers on a real GPU.

---

## 0. The three rules that break everything if you ignore them

### Rule 1 — `@cute.jit` / `@cute.kernel`, never `@cutlass.jit`
`cutlass.jit` **no longer exists** at the top level in 4.8. The entry points are:

```python
import cutlass
import cutlass.cute as cute

@cute.kernel          # device kernel
def my_kernel(t: cute.Tensor): ...

@cute.jit             # host function that builds/launches IR
def launch(t: cute.Tensor): ...
```

`@cute.jit` and `@cute.kernel` are **not interchangeable**:

| Decorator | Runs on | Can call | Can launch |
|---|---|---|---|
| `@cute.kernel` | device (GPU) | `cute.arch.*`, tensors, math | no |
| `@cute.jit` | host (Python) | `make_layout`, `make_tiled_mma`, `from_dlpack` | **yes** |

### Rule 2 — layout/atom construction is JIT-only
Since 4.2, `cute.make_layout`, `cute.make_tiled_mma`, `cute.make_tensor`, ... build MLIR and
therefore **raise `CALL_OUTSIDE_JIT` if called from plain Python**. Wrap construction in a
`@cute.jit` host function (or do it inside the `@cute.kernel`).

```python
# WRONG — raises error[CALL_OUTSIDE_JIT]
layout = cute.make_layout((128, 64), stride=(64, 1))     # at module level

# RIGHT
@cute.jit
def host():
    layout = cute.make_layout((128, 64), stride=(64, 1))
```

### Rule 3 — a `@cute.jit` host function cannot *write* to a tensor
Storing to a device tensor from a `@cute.jit` host function **segfaults** in 4.8. Host
functions must launch a kernel; the kernel does the stores.

```python
# RIGHT — always launch a kernel, even for a one-element result
@cute.kernel
def k(res: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        res[0] = 42

@cute.jit
def launch(res: cute.Tensor):
    k(res).launch(grid=[1, 1, 1], block=[32, 1, 1])

launch(from_dlpack(res))     # plain Python calls the host function
```

---

## 1. Canonical file skeleton (4.8)

```python
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def my_kernel(mA: cute.Tensor, ...):
    ...

@cute.jit
def launch(mA: cute.Tensor, ...):
    my_kernel(mA, ...).launch(grid=[gx, gy, 1], block=[BLOCK, 1, 1])

def run():
    a = torch.randn(..., device="cuda")
    launch(from_dlpack(a))
    torch.cuda.synchronize()

if __name__ == "__main__":
    run()
```

**Launch signature:** `kernel(args...).launch(grid=[...], block=[...], stream=<stream or None>)`.
The old `kernel(args..., grid=..., block=...)` form was removed. `stream=` is optional;
`cutlass.LaunchConfig` exists but `.launch(grid=..., block=...)` is the idiomatic form.

**Torch interop:** `from cutlass.cute.runtime import from_dlpack` (this import path has been
stable across all 4.x). Keep `torch` available for reference results.

**Import order matters:** import `torch` **before** `cutlass` if your environment has a
`torch` name clash with the DSL's internal shim. (`import cutlass` first works on a clean
env, but `import torch; import cutlass` is universally safe.)

---

## 2. Old → new symbol map

### 2.1 Tensor creation

| Legacy (≤4.1) | 4.8 | Notes |
|---|---|---|
| `cute.make_gmem_tensor(ptr, layout)` | `cute.make_tensor(ptr, layout)` | one `make_tensor` for all spaces |
| `cute.make_smem_tensor(ptr, layout)` | `cute.struct.MemRange` + `get_tensor(...)` | see §3 |
| `cute.make_rmem_tensor(layout, dtype)` | `cute.make_rmem_tensor(layout, dtype)` | unchanged |
| `cute.make_rmem_tensor_like(t)` | `cute.make_rmem_tensor_like(t)` | unchanged |
| `cute.make_tensor(gmem_ptr, shape)` | `cute.make_tensor(ptr, shape)` | unchanged |

`cute.make_tensor(iterator, layout)` is the single constructor. The address space comes from
the pointer, not the constructor.

### 2.2 Pointer / alignment (replaces `cute.b128`)

```python
# legacy: cute.b128(value)
ptr = cute.make_ptr(cute.Float16, value, assumed_align=16)     # 16-byte aligned pointer
```

`cute.b128` / `cute.b64` / `cute.b32` are gone. Use `cute.make_ptr(dtype, value, assumed_align=N)`.

### 2.3 Copy atoms and TiledCopy

| Legacy | 4.8 |
|---|---|
| `cute.Copy_atom(cute.UniversalCopy, T)` | `cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), T)` |
| `cute.Copy_atom(cute.SmemCopy, T)` | `cute.nvgpu.CopyS2ROp()` / `cute.nvgpu.CopyR2SOp()` |
| `cute.Copy_atom(cute.TmaCopy, T)` | `cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()` |
| `cute.copy.async(atom, src, dst)` | `cute.copy(atom, src, dst)` |
| `cute.copy(atom, src, dst)` | `cute.copy(atom, src, dst)` |
| `cute.make_tiled_copy_tv(...)` | `cute.make_tiled_copy_tv(...)` (unchanged) |
| — | `cute.make_tiled_copy_A/B/C(atom, tiled_mma)` (new, matches MMA TV layout) |
| `cute.autovec_copy` / `cute.basic_copy` | unchanged |

```python
atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cute.Float16, num_bits_per_copy=128)
thr_layout = cute.make_layout((128,))                       # 128 threads
val_layout = cute.make_ordered_layout((BM, BK), order=(1, 0))
tiled_copy = cute.make_tiled_copy_tv(atom, thr_layout, val_layout)
```

Copy-op namespace in 4.8:

| Purpose | Op |
|---|---|
| generic sync copy (gmem↔smem↔rmem) | `cute.nvgpu.CopyUniversalOp()` |
| register → shared | `cute.nvgpu.CopyR2SOp()` |
| shared → register | `cute.nvgpu.CopyS2ROp()` |
| global → shared, async (SM80 `cp.async`) | `cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL)` |
| TMA bulk tensor G→S (SM90+) | `cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()` |
| TMA bulk S→G | `cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()` |
| `ldmatrix` (S2R, MMA operand) | `cute.nvgpu.warp.LdMatrix8x8x16bOp(is_major, ...)` |
| `stmatrix` (R2S) | `cute.nvgpu.warp.StMatrix8x8x16bOp(is_major)` |

### 2.4 MMA atoms and TiledMMA

| Legacy | 4.8 |
|---|---|
| `cute.MMA_atom(...)` | `cute.make_mma_atom(op)` |
| `cute.Mma_Sm80_F16F16F16F32_TN(...)` | `cute.nvgpu.warp.MmaF16BF16Op(cute.Float16, cute.Float32, (16, 8, 16))` |
| `cute.Mma_Sm90_64x8x16_F16F16F16F32_TN` | `cute.nvgpu.warpgroup.MmaF16BF16Op(...)` |
| `cute.Mma_Sm100_...` | `cute.nvgpu.tcgen05.MmaF16BF16Op(...)` |
| `cute.make_tiled_mma(atom, (M,N,K))` | `cute.make_tiled_mma(op, atoms_layout, permutation_mnk=(M,N,K))` |
| — | `tiled_mma.make_fragment_A/B/C(...)` (new, replaces `make_rmem_tensor_like`) |
| `thr_mma.partition_fragment_A(t)` | `thr_mma.partition_A(t)` + `tiled_mma.make_fragment_A(...)` |
| `cute.gemm(tiled_mma, D, A, B, C)` | `cute.gemm(tiled_mma, D, A, B, C)` (unchanged) |

```python
op  = cute.nvgpu.warp.MmaF16BF16Op(cute.Float16, cute.Float32, (16, 8, 16))
atoms_layout = cute.make_ordered_layout((2, 2, 1), order=(1, 0, 2))
tiled_mma = cute.make_tiled_mma(op, atoms_layout, permutation_mnk=(64, 32, 32))

thr_mma = tiled_mma.get_slice(tidx)
tCsA   = thr_mma.partition_A(sA)          # SMEM tile -> MMA view
tCrA   = tiled_mma.make_fragment_A(tCsA)  # MMA view -> register fragment
tCrC   = tiled_mma.make_fragment_C(tCgC)
tCrC.fill(0.0)
cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
```

`cute.gemm` verifies `A.M == D.M` and `B.N == D.N`; the fragment shapes must agree with
`permutation_mnk`. Getting `permutation_mnk` wrong yields
`error: 'cute.gemm' op expects A's M == D'M and B's N == D'N`.

### 2.5 Dtypes

| Legacy | 4.8 |
|---|---|
| `cutlass.float16` / `float32` | `cutlass.Float16` / `cutlass.Float32` |
| `cutlass.float8_e4m3` | `cutlass.Float8E4M3FN` |
| `cutlass.float8_e5m2` | `cutlass.Float8E5M2` |
| `cutlass.int8` / `int32` | `cutlass.Int8` / `cutlass.Int32` |
| `cutlass.boolean` | `cutlass.Boolean` |
| `cutlass.tfloat32` | `cutlass.TFloat32` |

### 2.6 Thread/block indices and barriers

| Legacy | 4.8 |
|---|---|
| `cute.thread_idx` | `cute.arch.thread_idx()` |
| `cute.block_idx` | `cute.arch.block_idx()` |
| `cute.grid_idx` / `grid_dim` | `cute.arch.grid_idx()` / `cute.arch.grid_dim()` |
| `cute.warp_idx` | `cute.arch.warp_idx()` |
| `cute.lane_idx` | `cute.arch.lane_idx()` |
| `cute.sync_threads()` | `cute.arch.sync_threads()` |
| `cute.async_wait(n)` | `cute.arch.cp_async_wait_group(n)` |
| `cute.async_commit()` | `cute.arch.cp_async_commit_group()` |
| `cute.fence()` | `cute.arch.fence_view_async_shared()` / `cute.arch.fence_proxy(...)` |

### 2.7 Misc

| Legacy | 4.8 |
|---|---|
| `cute.copy(atom, src, dst, pred=p)` | unchanged — `pred=` is still a keyword |
| `cute.make_identity_tensor(shape)` | unchanged |
| `cute.elem_less(a, b)` | unchanged |
| `cute.cosize(layout)` | unchanged |
| `cute.Swizzle(3, 3, 2)` | `cute.make_swizzle(3, 3, 2)` |
| `cute.composition(swz, layout)` | unchanged (swizzle first, layout second) |
| `cute.Barrier` / manual mbarrier | `cutlass.pipeline.MbarrierArray` |
| `cute.pipeline.PipelineAsync` | `cutlass.pipeline.PipelineAsync` |
| `cutlass.utils.SmemAllocator()` | `cutlass.memory.SmemAllocator()` (utils version is deprecated) |

---

## 3. Shared memory (the `SmemAllocator` pattern)

```python
sA_layout = cute.make_ordered_layout((BM, BK, STAGES), order=(1, 0, 2))
swizzle   = cute.make_swizzle(3, 3, 2)

@cute.struct
class Smem:
    a: cute.struct.Align[cute.struct.MemRange[cute.Float16, cute.cosize(sA_layout)], 16]

@cute.kernel
def k(...):
    smem    = cutlass.memory.SmemAllocator()
    storage = smem.allocate(Smem.size_in_bytes(), byte_alignment=16)
    sA = Smem(storage).a.get_tensor(sA_layout, swizzle=swizzle)
```

Layouts, atoms and the struct must be defined **inside** the kernel (or in a `@cute.jit`
host function) because they build IR.

---

## 4. Pipelines (4.8)

`cutlass.pipeline` ships the mbarrier-backed pipelines:

| Class | Use |
|---|---|
| `PipelineAsync` | `cp.async` (SM80+) double/multi buffering |
| `PipelineCpAsync` | `cp.async` variant |
| `PipelineTmaAsync` | TMA load + mbarrier (SM90+) |
| `PipelineTmaUmma` / `PipelineUmmaAsync` / `PipelineAsyncUmma` | Blackwell `tcgen05` UMMA |
| `PipelineTmaStore` | TMA epilogue store |
| `PipelineClcFetchAsync` | cluster-launch-control fetch (SM100+) |

```python
pipeline = cutlass.pipeline.PipelineAsync.create(
    num_stages=NUM_STAGES,
    producer_group=cutlass.pipeline.CooperativeGroup(1),
    consumer_group=cutlass.pipeline.CooperativeGroup(1),
    barrier_storage=storage,
    tidx=tidx,
)
pipeline.producer_acquire(src_stage)
pipeline.consumer_wait(src_stage)
# ...
pipeline.producer_commit(src_stage)
pipeline.consumer_release(src_stage)
```

For a hand-rolled double buffer that needs no mbarriers, a plain
`cp_async_commit_group()` / `cp_async_wait_group(0)` + `sync_threads()` loop is still the
clearest teaching form and is what the curriculum uses for Module 06.

---

## 5. Deprecated / removed in 4.8

- `cutlass.jit`, `cutlass.cute.jit` → use `cute.jit` (`cutlass.jit` is gone).
- `cute.make_gmem_tensor` / `cute.make_smem_tensor` / `cute.make_builtin_tensor` → `cute.make_tensor`.
- `cute.b128` / `b64` / `b32` → `cute.make_ptr(..., assumed_align=...)`.
- `cutlass.utils.SmemAllocator` / `cutlass.utils.TmemAllocator` → `cutlass.memory.*`.
- `cute.arch` shrank: use `cute.arch.*` for PTX, `cute.nvgpu.*` for copy/MMA ops.

---

## 6. Pinning and environment

```bash
pip install "nvidia-cutlass-dsl==4.8.0"
```

Architecture support in 4.8:

| Arch | Min DSL version |
|---|---|
| SM80 (A100) | 4.0 |
| SM90 (H100) | 4.0 |
| SM100 (B200) | 4.1 |
| SM120 (RTX 50xx / GB20x) | 4.2 |

A kernel compiled for `sm_90a` cannot run on `sm_89`. Gate arch-specific exercises so they
*compile-skip* rather than *fail* on unsupported hardware.