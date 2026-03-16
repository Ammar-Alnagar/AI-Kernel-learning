// CONCEPT: Data race on non-atomic shared variable — Undefined Behavior
// FORMAT: DEBUG
// TIME_TARGET: 15 min
// WHY_THIS_MATTERS: Data races are UB — the compiler assumes they don't happen.
// CUDA_CONNECTION: Multiple threads updating shared counter without synchronization.

#include <iostream>
#include <thread>
#include <vector>

// SYMPTOMS:
// 1. Final count is less than expected (lost updates)
// 2. Results vary between runs (non-deterministic)
// 3. With optimization, may see torn reads (partial updates)
// 4. ThreadSanitizer reports: data race

// BUG: Shared counter without synchronization
// Multiple threads read-modify-write simultaneously
// This is a DATA RACE — undefined behavior
int shared_counter = 0;  // BUG LINE: should be std::atomic<int>

void increment_counter(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        // BUG: Read-modify-write without synchronization
        // Thread 1: reads 0
        // Thread 2: reads 0 (before Thread 1 writes)
        // Thread 1: writes 1
        // Thread 2: writes 1 (lost Thread 1's update!)
        shared_counter++;  // BUG LINE: data race!
    }
}

void demonstrate_data_race() {
    const int num_threads = 4;
    const int iterations = 100000;
    const int expected = num_threads * iterations;
    
    std::cout << "=== Data race demonstration ===\n";
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
    std::cout << "Lost updates: " << (expected - shared_counter) << "\n";
    std::cout << "(Results vary — data race is non-deterministic)\n";
}

// TODO: Fix by using std::atomic<int>
// std::atomic provides:
// 1. Atomicity: read-modify-write is indivisible
// 2. Visibility: changes are visible to other threads
// 3. No data race: defined behavior

// Declare atomic counter (uncomment and use in fixed version)
// std::atomic<int> atomic_counter{0};

void increment_atomic_counter(int iterations) {
    // TODO: Use atomic_counter++ instead of shared_counter++
    // Atomic increment is indivisible — no lost updates
}

void demonstrate_fixed() {
    const int num_threads = 4;
    const int iterations = 100000;
    const int expected = num_threads * iterations;
    
    std::cout << "\n=== Fixed: atomic counter ===\n";
    std::cout << "Threads: " << num_threads << ", Iterations each: " << iterations << "\n";
    std::cout << "Expected final count: " << expected << "\n";
    
    std::atomic<int> atomic_counter{0};
    
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&atomic_counter, iterations]() {
            for (int i = 0; i < iterations; ++i) {
                atomic_counter++;  // Atomic increment — no data race
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "Actual final count: " << atomic_counter << "\n";
    std::cout << "Lost updates: " << (expected - atomic_counter) << "\n";
    std::cout << "(Should be 0 — atomic operations are thread-safe)\n";
}

int main() {
    // Uncomment to see data race (with ThreadSanitizer):
    // demonstrate_data_race();
    
    demonstrate_fixed();
    
    std::cout << "\n=== KEY LEARNING ===\n";
    std::cout << "Data race on non-atomic shared variable is UNDEFINED BEHAVIOR.\n";
    std::cout << "Symptoms: lost updates, torn reads, non-deterministic results.\n";
    std::cout << "\nFix: use std::atomic<T> for shared variables.\n";
    std::cout << "atomic<T> guarantees atomicity (indivisible operations)\n";
    std::cout << "and visibility (changes seen by other threads).\n";
    
    return 0;
}

// VERIFY:
// Buggy version: ThreadSanitizer reports data race, count < expected
// Fixed version: no sanitizer errors, count == expected (400000)

// BUILD COMMAND:
// g++ -std=c++20 -O2 -fsanitize=thread -o ex05 exercise.cpp -lpthread && ./ex05
