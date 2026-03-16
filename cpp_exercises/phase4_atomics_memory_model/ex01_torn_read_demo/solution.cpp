// SOLUTION: ex01_torn_read_demo
// Demonstrates torn reads/writes and atomic fix

#include <iostream>
#include <thread>
#include <vector>
#include <cstdint>

// BUGGY: 64-bit counter without atomic — can tear
uint64_t shared_counter = 0;

void increment_counter(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        // Non-atomic read-modify-write on 64-bit value
        // Can tear: read high 32 bits from one write, low 32 bits from another
        shared_counter++;
    }
}

void demonstrate_tearing() {
    const int num_threads = 4;
    const int iterations = 100000;
    const uint64_t expected = static_cast<uint64_t>(num_threads) * iterations;
    
    std::cout << "=== Torn Read/Write Demonstration (BUGGY) ===\n";
    std::cout << "Threads: " << num_threads << ", Iterations each: " << iterations << "\n";
    std::cout << "Expected final count: " << expected << "\n";
    
    shared_counter = 0;
    
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(increment_counter, iterations);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "Actual final count: " << shared_counter << "\n";
    std::cout << "Difference: " << static_cast<int64_t>(expected - shared_counter) << "\n";
    std::cout << "(Results vary — tearing + data race is non-deterministic)\n";
}

void demonstrate_fixed() {
    const int num_threads = 4;
    const int iterations = 100000;
    const uint64_t expected = static_cast<uint64_t>(num_threads) * iterations;
    
    std::cout << "\n=== Fixed: Atomic Counter ===\n";
    std::cout << "Threads: " << num_threads << ", Iterations each: " << iterations << "\n";
    std::cout << "Expected final count: " << expected << "\n";
    
    std::atomic<uint64_t> atomic_counter{0};
    
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&atomic_counter, iterations]() {
            for (int i = 0; i < iterations; ++i) {
                atomic_counter++;  // Atomic increment — no tearing
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "Actual final count: " << atomic_counter << "\n";
    std::cout << "Difference: " << static_cast<int64_t>(expected - atomic_counter) << "\n";
    std::cout << "(Should be 0 — atomic operations are indivisible)\n";
}

void demonstrate_tearing_concept() {
    std::cout << "\n=== Understanding Tearing ===\n";
    std::cout << "A 64-bit value is stored as two 32-bit halves:\n";
    std::cout << "  [high 32 bits][low 32 bits]\n\n";
    
    std::cout << "Without atomic, a write may happen in two steps:\n";
    std::cout << "  1. Write high 32 bits: 0x00000000\n";
    std::cout << "  2. Write low 32 bits:  0x00000001\n";
    std::cout << "  Result: 0x0000000000000001\n\n";
    
    std::cout << "If Thread B reads BETWEEN steps 1 and 2:\n";
    std::cout << "  - Reads high 32 bits: 0x00000000 (old value)\n";
    std::cout << "  - Reads low 32 bits:  0x00000001 (new value)\n";
    std::cout << "  - Combined: 0x0000000000000001 (correct by luck)\n\n";
    
    std::cout << "But if Thread A writes a DIFFERENT value:\n";
    std::cout << "  1. Thread A writes high: 0x00000001 (for value 0x0000000100000000)\n";
    std::cout << "  2. Thread B reads high:  0x00000001\n";
    std::cout << "  3. Thread A writes low:  0x00000000\n";
    std::cout << "  4. Thread B reads low:   0x00000000\n";
    std::cout << "  - B sees: 0x0000000100000000 (NEITHER the old nor new value!)\n";
    std::cout << "  This is a TORN READ — a value that was never written.\n";
}

int main() {
    // demonstrate_tearing();  // Uncomment to see UB
    demonstrate_fixed();
    demonstrate_tearing_concept();
    
    std::cout << "\n=== KEY LEARNING ===\n";
    std::cout << "Non-atomic 64-bit operations can TEAR:\n";
    std::cout << "  - Write: may be two 32-bit writes internally\n";
    std::cout << "  - Read: may read high/low from different writes\n";
    std::cout << "\nstd::atomic<uint64_t> guarantees:\n";
    std::cout << "  1. ATOMICITY: no tearing (indivisible read/write)\n";
    std::cout << "  2. VISIBILITY: changes seen by other threads\n";
    std::cout << "  3. ORDERING: with memory_order, controls when visible\n";
    std::cout << "\nNote: On x86-64, aligned 64-bit loads/stores are atomic at hardware level.\n";
    std::cout << "But the compiler may still generate non-atomic code without std::atomic!\n";
    std::cout << "Always use std::atomic for shared 64-bit values.\n";
    
    return 0;
}

// KEY_INSIGHT:
// Tearing: reading/writing a value in multiple parts (e.g., high/low 32 bits).
// Without atomicity, another thread can interleave, causing torn reads.
//
// Two layers of std::atomic:
// Layer 1 (ATOMICITY): Prevents tearing — read/write is indivisible
// Layer 2 (ORDERING): memory_order controls WHEN changes become visible
//
// This exercise focuses on Layer 1 (atomicity).
// Next exercises cover Layer 2 (memory ordering).
//
// CUDA mapping: Device-side 64-bit operations need atomics too.
// atomicAdd((unsigned long long*)ptr, 1) ensures atomic 64-bit increment.
// Without it, multiple threads updating a counter get torn results.
