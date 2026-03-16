// SOLUTION: ex05_data_race_no_atomic
// Demonstrates data race UB and atomic fix

#include <iostream>
#include <thread>
#include <vector>
#include <atomic>

// BUGGY: Shared counter without synchronization
int shared_counter = 0;

void increment_counter(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        // DATA RACE: multiple threads read-modify-write simultaneously
        // This is UB — compiler assumes no data races
        shared_counter++;
    }
}

void demonstrate_data_race() {
    const int num_threads = 4;
    const int iterations = 100000;
    const int expected = num_threads * iterations;
    
    std::cout << "=== Data race demonstration (BUGGY) ===\n";
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

void demonstrate_memory_order() {
    std::cout << "\n=== Memory order demonstration ===\n";
    
    std::atomic<int> data{0};
    std::atomic<bool> ready{false};
    
    // Writer thread
    std::thread writer([&data, &ready]() {
        data.store(42, std::memory_order_relaxed);  // Write data
        ready.store(true, std::memory_order_release);  // Publish (release semantics)
    });
    
    // Reader thread
    std::thread reader([&data, &ready]() {
        while (!ready.load(std::memory_order_acquire)) {  // Wait for publish
            std::this_thread::yield();
        }
        std::cout << "Reader sees data = " << data.load(std::memory_order_relaxed) << "\n";
    });
    
    writer.join();
    reader.join();
    
    std::cout << "(Acquire/release ensures reader sees writer's data)\n";
}

int main() {
    // demonstrate_data_race();  // Uncomment to see UB
    demonstrate_fixed();
    demonstrate_memory_order();
    
    std::cout << "\n=== KEY LEARNING ===\n";
    std::cout << "Data race on non-atomic shared variable is UNDEFINED BEHAVIOR.\n";
    std::cout << "\nstd::atomic<T> provides two guarantees:\n";
    std::cout << "  1. ATOMICITY: read-modify-write is indivisible (no lost updates)\n";
    std::cout << "  2. VISIBILITY: changes are visible to other threads\n";
    std::cout << "\nMemory order (relaxed, acquire, release, seq_cst) controls\n";
    std::cout << "WHEN changes become visible — separate from atomicity.\n";
    
    return 0;
}

// KEY_INSIGHT:
// Data race = two threads access same variable, at least one writes, no sync.
// Data races are UB — the compiler assumes they don't happen and optimizes
// accordingly. This can cause torn reads, lost updates, or worse.
//
// std::atomic<T> fixes atomicity (indivisible operations).
// memory_order controls visibility (when other threads see changes).
//
// Two layers:
// Layer 1 (atomicity): atomic<T> prevents tearing — use for counters, flags
// Layer 2 (ordering): memory_order_acquire/release for data dependencies
//
// CUDA mapping: Atomic operations in device code (atomicAdd, atomicCAS) provide
// the same guarantees. Global memory counters need atomics. Shared memory
// counters within a warp can use __syncwarp() instead (cheaper).
