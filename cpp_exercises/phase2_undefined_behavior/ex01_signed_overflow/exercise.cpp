// CONCEPT: Signed integer overflow is Undefined Behavior
// FORMAT: DEBUG
// TIME_TARGET: 10 min
// WHY_THIS_MATTERS: UB means the compiler can assume it never happens — optimizations break your code.
// CUDA_CONNECTION: Kernel loop counters — overflow causes infinite loops or skipped iterations.

#include <iostream>
#include <limits>
#include <cstdint>

// SYMPTOMS:
// 1. With -O2 optimization, loop may become infinite or skip entirely
// 2. Sanitizer reports: "signed integer overflow"
// 3. Output differs between Debug and Release builds

// BUG: This function has signed integer overflow UB
// When i reaches INT32_MAX, i + 1 overflows to INT32_MIN (or worse)
int32_t sum_with_overflow(int32_t start, int32_t count) {
    int32_t sum = 0;
    // BUG: When start + i exceeds INT32_MAX, signed overflow UB occurs
    // The compiler may optimize assuming this never happens
    for (int32_t i = 0; i < count; ++i) {
        sum += start + i;  // BUG LINE: potential signed overflow
    }
    return sum;
}

// TODO: Fix by using unsigned arithmetic or checking for overflow
// Option 1: Use uint32_t (wraps defined, not UB)
// Option 2: Use int64_t (larger range, overflow unlikely)
// Option 3: Check before adding: if (sum > INT32_MAX - value) handle_overflow()

int32_t sum_fixed(int32_t start, int32_t count) {
    // TODO: Implement fix here
    // Use int64_t for accumulation, then check if result fits in int32_t
    int64_t sum = 0;
    for (int32_t i = 0; i < count; ++i) {
        sum += static_cast<int64_t>(start) + i;
    }
    // For this exercise, just return the lower 32 bits
    // In real code, you'd check if sum exceeds INT32_MAX
    return static_cast<int32_t>(sum);
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
    std::cout << "(Still wraps, but via defined unsigned behavior)\n";
    
    std::cout << "\n=== KEY LEARNING ===\n";
    std::cout << "Signed integer overflow is UNDEFINED BEHAVIOR.\n";
    std::cout << "Compiler assumes it never happens — optimizations break code.\n";
    std::cout << "Fix: use unsigned (defined wrap) or larger type (avoid overflow).\n";
    
    return 0;
}

// VERIFY (with sanitizer):
// Running buggy version with -fsanitize=undefined:
// runtime error: signed integer overflow: 2147483647 + 1 cannot be represented in type 'int'
//
// Fixed version: no sanitizer error, consistent output across builds.

// BUILD COMMAND:
// g++ -std=c++20 -O2 -fsanitize=undefined -o ex01 exercise.cpp && ./ex01
