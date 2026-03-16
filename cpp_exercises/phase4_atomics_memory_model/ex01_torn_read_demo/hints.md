# Hints for ex01_torn_read_demo

## H1 — Concept Direction
A 64-bit value may be written as two 32-bit operations internally. Without atomicity, another thread can read between these operations, getting high bits from one write and low bits from another — a "torn" value that was never actually written.

## H2 — Names the Tool
Use `std::atomic<uint64_t>` for the shared counter. Atomic operations guarantee indivisible read-modify-write: `atomic_counter++` cannot be interleaved with other threads' operations.

## H3 — Minimal Usage (Unrelated Context)
```cpp
std::atomic<uint64_t> counter{0};

// Thread-safe 64-bit increment:
counter++;  // Atomic — no tearing

// Non-atomic (can tear):
uint64_t counter2 = 0;
counter2++;  // May be two 32-bit operations internally
```
