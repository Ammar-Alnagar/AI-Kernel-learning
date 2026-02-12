# Makefile for CUDA Fundamentals Tutorials

# Compiler
NVCC = nvcc

# Compiler flags
CFLAGS = -O2

# Target executables
TARGETS = thread_hierarchy_tutorial memory_hierarchy_tutorial hands_on_kernels_tutorial hands_on_kernels_solutions \
	vector_add_exercise matrix_mul_exercise reduction_exercise memory_coalescing_exercise \
	shared_memory_banking_exercise atomic_operations_exercise cuda_streams_exercise \
	warp_primitives_exercise master_hands_on_tutorial

# Default target
all: $(TARGETS)

# Thread Hierarchy Tutorial
thread_hierarchy_tutorial: thread_hierarchy_tutorial.cu
	$(NVCC) $(CFLAGS) $< -o $@

# Memory Hierarchy Tutorial
memory_hierarchy_tutorial: memory_hierarchy_tutorial.cu
	$(NVCC) $(CFLAGS) $< -o $@

# Hands-On Kernels Tutorial (with missing code)
hands_on_kernels_tutorial: hands_on_kernels_tutorial.cu
	$(NVCC) $(CFLAGS) $< -o $@

# Solutions for Hands-On Kernels
hands_on_kernels_solutions: hands_on_kernels_solutions.cu
	$(NVCC) $(CFLAGS) $< -o $@

# Individual Hands-On Exercises
vector_add_exercise: vector_add_exercise.cu
	$(NVCC) $(CFLAGS) $< -o $@

matrix_mul_exercise: matrix_mul_exercise.cu
	$(NVCC) $(CFLAGS) $< -o $@

reduction_exercise: reduction_exercise.cu
	$(NVCC) $(CFLAGS) $< -o $@

memory_coalescing_exercise: memory_coalescing_exercise.cu
	$(NVCC) $(CFLAGS) $< -o $@

shared_memory_banking_exercise: shared_memory_banking_exercise.cu
	$(NVCC) $(CFLAGS) $< -o $@

atomic_operations_exercise: atomic_operations_exercise.cu
	$(NVCC) $(CFLAGS) $< -o $@

cuda_streams_exercise: cuda_streams_exercise.cu
	$(NVCC) $(CFLAGS) $< -o $@

warp_primitives_exercise: warp_primitives_exercise.cu
	$(NVCC) $(CFLAGS) $< -o $@

master_hands_on_tutorial: master_hands_on_tutorial.cu
	$(NVCC) $(CFLAGS) $< -o $@

# Clean build artifacts
clean:
	rm -f $(TARGETS) *.o

# Run all tutorials
run-all: all
	@echo "Running Thread Hierarchy Tutorial..."
	./thread_hierarchy_tutorial
	@echo "\nRunning Memory Hierarchy Tutorial..."
	./memory_hierarchy_tutorial
	@echo "\nHands-On Kernels Tutorial compiled (requires code completion)"
	@echo "\nRunning Solutions for Hands-On Kernels..."
	./hands_on_kernels_solutions

# Run individual tutorials
run-thread-hierarchy: thread_hierarchy_tutorial
	./thread_hierarchy_tutorial

run-memory-hierarchy: memory_hierarchy_tutorial
	./memory_hierarchy_tutorial

run-hands-on: hands_on_kernels_tutorial
	./hands_on_kernels_tutorial

run-solutions: hands_on_kernels_solutions
	./hands_on_kernels_solutions

# Run individual hands-on exercises
run-vector-add: vector_add_exercise
	./vector_add_exercise

run-matrix-mul: matrix_mul_exercise
	./matrix_mul_exercise

run-reduction: reduction_exercise
	./reduction_exercise

run-coalescing: memory_coalescing_exercise
	./memory_coalescing_exercise

run-shared-memory: shared_memory_banking_exercise
	./shared_memory_banking_exercise

run-atomic: atomic_operations_exercise
	./atomic_operations_exercise

run-streams: cuda_streams_exercise
	./cuda_streams_exercise

run-warp: warp_primitives_exercise
	./warp_primitives_exercise

run-master: master_hands_on_tutorial
	./master_hands_on_tutorial

.PHONY: all clean run-all run-thread-hierarchy run-memory-hierarchy run-hands-on run-solutions \
	run-vector-add run-matrix-mul run-reduction run-coalescing run-shared-memory \
	run-atomic run-streams run-warp run-master