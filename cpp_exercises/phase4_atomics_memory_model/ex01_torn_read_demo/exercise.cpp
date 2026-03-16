// CONCEPT: Torn reads/writes — 64-bit counter without atomic
// FORMAT: DEBUG
// TIME_TARGET: 15 min
// WHY_THIS_MATTERS: 64-bit values on 32-bit systems (or non-atomic on any system) can tear.
// CUDA_CONNECTION: Global memory counters without atomics produce incorrect results.

#include <iostream>
#include <thread>
#include <vector>
#include <cstdint>

// SYMPTOMS:
// 1. Counter shows values that were never written (torn reads)
// 2. Different results each run (non-deterministic)
// 3. ThreadSanitizer reports: data race
// 4. More likely on 32-bit systems, but can happen on 64-bit too

// BUG: 64-bit counter without atomic — can tear on read/write
// A 64-bit write may be two 32-bit writes internally
// Thread A reads high 32 bits, low 32 bits from different writes
uint64_t shared_counter = 0;  // BUG LINE: should be std::atomic<uint64_t>

void increment_counter(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        // BUG: Non-atomic read-modify-write on 64-bit value
        // This is TWO operations: read 64-bit, add 1, write 64-bit
        // Another thread can interleave between read and write
        shared_counter++;  // BUG LINE: torn read/write possible
    }
}

void demonstrate_tearing() {
    const int num_threads = 4;
    const int iterations = 100000;
    const uint64_t expected = static_cast<uint64_t>(num_threads) * iterations;
    
    std::cout << "=== Torn Read/Write Demonstration ===\n";
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
    std::cout << "(Negative difference means counter exceeded expected — very bad!)\n";
}

// TODO: Fix by using std::atomic<uint64_t>
// std::atomic guarantees:
// 1. No tearing: 64-bit read/write is indivisible
// 2. No lost updates: read-modify-write is atomic
// 3. Visibility: changes visible to other threads

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

int main() {
    // Uncomment to see tearing (with ThreadSanitizer):
    // demonstrate_tearing();
    
    demonstrate_fixed();
    
    std::cout << "\n=== KEY LEARNING ===\n";
    std::cout << "Non-atomic 64-bit operations can TEAR:\n";
    std::cout << "  - Write: may be two 32-bit writes internally\n";
    std::cout << "  - Read: may read high/low from different writes\n";
    std::cout << "\nstd::atomic<uint64_t> guarantees:\n";
    std::cout << "  1. ATOMICITY: no tearing (indivisible read/write)\n";
    std::cout << "  2. VISIBILITY: changes seen by other threads\n";
    std::cout << "  3. ORDERING: with memory_order, controls when visible\n";
    
    return 0;
}

// VERIFY:
// Buggy version: ThreadSanitizer reports data race, count varies wildly
// Fixed version: no sanitizer errors, count == expected (400000)

// BUILD COMMAND:
// g++ -std=c++20 -O2 -fsanitize=thread -o ex01 exercise.cpp -lpthread && ./ex01
