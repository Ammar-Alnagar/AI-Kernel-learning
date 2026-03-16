// SOLUTION: ex01_signed_overflow
// Demonstrates signed integer overflow UB and how to fix it

#include <iostream>
#include <limits>
#include <cstdint>

// BUGGY: This function has signed integer overflow UB
int32_t sum_with_overflow(int32_t start, int32_t count) {
    int32_t sum = 0;
    for (int32_t i = 0; i < count; ++i) {
        sum += start + i;  // UB when sum exceeds INT32_MAX
    }
    return sum;
}

// FIXED: Use int64_t for accumulation to avoid overflow
int32_t sum_fixed(int32_t start, int32_t count) {
    int64_t sum = 0;  // Larger type avoids overflow
    for (int32_t i = 0; i < count; ++i) {
        sum += static_cast<int64_t>(start) + i;
    }
    // In production, check if result fits in int32_t
    if (sum > std::numeric_limits<int32_t>::max()) {
        std::cerr << "Warning: result exceeds int32_t range\n";
    }
    return static_cast<int32_t>(sum & 0xFFFFFFFF);  // Lower 32 bits
}

// ALTERNATIVE FIX: Use unsigned (defined wrap behavior)
uint32_t sum_unsigned(uint32_t start, uint32_t count) {
    uint32_t sum = 0;
    for (uint32_t i = 0; i < count; ++i) {
        sum += start + i;  // Defined: wraps modulo 2^32
    }
    return sum;
}

int main() {
    std::cout << "INT32_MAX = " << std::numeric_limits<int32_t>::max() << "\n";
    std::cout << "INT32_MIN = " << std::numeric_limits<int32_t>::min() << "\n\n";
    
    // Test case 1: Normal case (no overflow)
    std::cout << "=== Test 1: Normal case ===\n";
    int32_t result1 = sum_with_overflow(0, 100);
    std::cout << "sum(0, 100) = " << result1 << " (expected: 4950)\n";
    
    // Test case 2: Near overflow (triggers UB)
    std::cout << "\n=== Test 2: Near overflow (UB!) ===\n";
    int32_t near_max = std::numeric_limits<int32_t>::max() - 100;
    std::cout << "Starting near INT32_MAX: " << near_max << "\n";
    int32_t result2 = sum_with_overflow(near_max, 200);
    std::cout << "sum(" << near_max << ", 200) = " << result2 << "\n";
    std::cout << "(Result is garbage due to UB — may vary by compiler/build)\n";
    
    // Test case 3: Fixed version
    std::cout << "\n=== Test 3: Fixed version ===\n";
    int32_t result3 = sum_fixed(near_max, 200);
    std::cout << "sum_fixed(" << near_max << ", 200) = " << result3 << "\n";
    
    // Test case 4: Unsigned version (defined wrap)
    std::cout << "\n=== Test 4: Unsigned version (defined wrap) ===\n";
    uint32_t result4 = sum_unsigned(near_max, 200);
    std::cout << "sum_unsigned(" << near_max << ", 200) = " << result4 << "\n";
    std::cout << "(Wraps via modulo 2^32 — defined behavior)\n";
    
    std::cout << "\n=== KEY LEARNING ===\n";
    std::cout << "Signed integer overflow is UNDEFINED BEHAVIOR.\n";
    std::cout << "Compiler assumes it never happens — optimizations break code.\n";
    std::cout << "\nFix options:\n";
    std::cout << "1. Use unsigned (defined wrap modulo 2^n)\n";
    std::cout << "2. Use larger type (avoid overflow entirely)\n";
    std::cout << "3. Check before operation: if (a > MAX - b) overflow()\n";
    
    return 0;
}

// KEY_INSIGHT:
// Signed integer overflow is UB — the compiler can assume it never happens.
// This means: loop optimizations may remove overflow checks, infinite loops
// may appear, results may differ between Debug and Release builds.
//
// Unsigned overflow is DEFINED: wraps modulo 2^n (e.g., 2^32 for uint32_t).
// This is why size_t (unsigned) is used for array indices — wrap is safe.
//
// CUDA mapping: In device code, signed overflow is also UB. Kernel loop
// counters should use unsigned or check bounds explicitly. CUTLASS uses
// size_t and uint32_t for tile indices to avoid this class of bugs.
