# Hints for ex01_signed_overflow

## H1 — Concept Direction
Signed integer overflow is Undefined Behavior in C++. When `INT32_MAX + 1` occurs, the result is not guaranteed to be `INT32_MIN` — the program can do anything. The compiler optimizes assuming overflow never happens.

## H2 — Names the Tool
Fix by using `int64_t` for accumulation (larger range), or `uint32_t` (defined wrap behavior). Alternatively, check before adding: `if (sum > INT32_MAX - value) handle_overflow()`.

## H3 — Minimal Usage (Unrelated Context)
```cpp
// Safe accumulation pattern:
int64_t sum = 0;
for (int32_t x : values) {
    sum += x;  // int64_t rarely overflows
}
// Or use unsigned for defined wrap:
uint32_t hash = 0;
hash += value;  // Wraps modulo 2^32 — defined!
```
