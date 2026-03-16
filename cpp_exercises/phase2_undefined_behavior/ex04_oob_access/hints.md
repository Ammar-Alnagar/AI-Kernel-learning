# Hints for ex04_oob_access

## H1 — Concept Direction
Array indices in C++ are 0-based. An array of size 5 has valid indices 0, 1, 2, 3, 4. Accessing index 5 or higher reads/writes memory beyond the array — undefined behavior that may corrupt adjacent variables.

## H2 — Names the Tool
Fix the loop condition: change `i <= size` to `i < size`. For safe access, use `std::array::at()` or `std::vector::at()` which throw `std::out_of_range` on OOB access instead of UB.

## H3 — Minimal Usage (Unrelated Context)
```cpp
std::array<int, 5> arr = {1, 2, 3, 4, 5};
arr.at(4);  // OK: returns 5
arr.at(5);  // Throws std::out_of_range

// Raw array: always check bounds
if (idx < arr.size()) {
    value = arr[idx];  // Safe
}
```
