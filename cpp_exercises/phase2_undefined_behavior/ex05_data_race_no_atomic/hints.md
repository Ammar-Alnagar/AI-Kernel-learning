# Hints for ex05_data_race_no_atomic

## H1 — Concept Direction
A data race occurs when two threads access the same variable, at least one writes, and there's no synchronization. The `shared_counter++` operation is read-modify-write — not atomic. Two threads can read the same value, both increment, and write back — losing one update.

## H2 — Names the Tool
Use `std::atomic<int>` for the shared counter. Atomic increment `atomic_counter++` is indivisible — no lost updates. The type guarantees both atomicity (indivisible operations) and visibility (changes seen by other threads).

## H3 — Minimal Usage (Unrelated Context)
```cpp
std::atomic<int> counter{0};

// Thread-safe increment:
counter++;  // Atomic — no data race

// Compare with non-atomic (data race):
int counter2 = 0;
counter2++;  // NOT atomic — data race if multiple threads
```
