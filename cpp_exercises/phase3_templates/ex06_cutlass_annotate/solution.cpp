// SOLUTION: ex06_cutlass_annotate
// CUTLASS source annotation with answers explained

#include <iostream>

// This exercise teaches reading CUTLASS 3.x source code patterns.
// The code below is simplified but captures the essential structures.

// ==================== Section 1: Policy Struct ====================

// Policy struct with type alias exports
template<class ArchTag, class ElementA, class ElementB, class ElementC, class Layout>
struct CollectiveGemmPolicy {
  using ArchTag = ArchTag;                              // Export template parameter
  using ElementA = ElementA;                            // Queryable from outside
  using ElementB = ElementB;
  using ElementC = ElementC;
  using Layout = Layout;
  
  // Tile shapes at different levels of the hierarchy
  using ThreadBlockShape = cutlass::gemm::GemmShape<128, 128, 64>;  // CTA level
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;           // Warp level
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;     // Tensor core
};

// A1: 'using Name = Name;' exports the template parameter as a named type.
//     Outside code queries: typename Policy::ArchTag
//     This is the type alias export pattern (ex01).
//
// A2: ThreadBlockShape: tiles per thread block (CTA level).
//     WarpShape: tiles per warp (subset of thread block).
//     InstructionShape: tensor core instruction dimensions (hardware).
//     Example: ThreadBlock 128x128, Warp 64x64, Instruction 16x8.
//
// A3: Nested aliases group related config. One template parameter
//     (Policy) instead of many. Easier to swap policies.


// ==================== Section 2: Kernel with CRTP ====================

template<class ProblemShape, class Policy, class TileIterator>
class CollectiveGemm {
public:
  using ElementA = typename Policy::ElementA;    // typename required for dependent name
  using ElementB = typename Policy::ElementB;
  using ElementC = typename Policy::ElementC;
  
  using SharedStorage = typename Policy::SharedStorage;  // __shared__ memory wrapper
  
  __device__ void operator()(
      ProblemShape problem_shape,
      ElementA* ptr_A,
      ElementB* ptr_B,
      ElementC* ptr_C,
      SharedStorage& shared_storage)  // Reference to avoid copy
  {
    // Kernel implementation uses shared_storage for cooperative loading
  }
};

// A4: 'typename' is required for DEPENDENT NAMES — names that depend
//     on template parameters. Policy::ElementA depends on Policy.
//     Without 'typename', compiler assumes it's a value/function.
//
// A5: SharedStorage is __shared__ memory wrapper (RAII).
//     Passed by reference to avoid copying (it's large).
//     Reference ensures all threads see same shared memory.


// ==================== Section 3: Type Dispatch ====================

// Forward declarations
struct GemmFloat {};
struct GemmHalf {};
struct GemmBFloat16 {};

// Type trait: maps Element type to Gemm implementation type
template<class Element>
struct GemmTypeSelector;

template<>
struct GemmTypeSelector<float> {
  using Type = GemmFloat;    // Type trait pattern
};

template<>
struct GemmTypeSelector<__half> {
  using Type = GemmHalf;
};

template<>
struct GemmTypeSelector<__nv_bfloat16> {
  using Type = GemmBFloat16;
};

// Usage:
// using SelectedGemm = typename GemmTypeSelector<Element>::Type;

// A6: Type trait / type mapping pattern.
//     Maps input type (Element) to output type (GemmType).
//     Similar to std::remove_const<T>::type.
//
// A7: Separation of concerns. GemmTypeSelector handles dispatch.
//     Main Gemm class handles computation. Easier to maintain.


// ==================== Section 4: Parameter Pack Expansion ====================

// Dummy stage types for demonstration
struct Stage0 {};
struct Stage1 {};
struct Stage2 {};

template<class... Stages>
class Pipeline {
  static constexpr int NumStages = sizeof...(Stages);  // Count types in pack
  
  template<int Stage>
  __device__ void process() {
    // Process individual stage
  }
  
  __device__ void run() {
    // Fold expression: expands to process<Stage0>(), process<Stage1>(), process<Stage2>()
    (process<Stages>(), ...);  // Unary right fold with comma operator
  }
};

// Usage:
// Pipeline<Stage0, Stage1, Stage2> pipeline;
// pipeline.run();  // Calls process<0>(), process<1>(), process<2>()

// A8: sizeof...(Stages) returns the NUMBER of types in the pack.
//     Pipeline<Stage0, Stage1, Stage2> has sizeof...(Stages) = 3.
//
// A9: Fold expression expands to: process<Stage0>(), process<Stage1>(), process<Stage2>()
//     The comma operator sequences the calls. Unary right fold.


// ==================== Demonstration ====================

void print_answers() {
    std::cout << "\n=== ANSWERS ===\n\n";
    
    std::cout << "A1: 'using Name = Name;' exports the template parameter as a named type.\n";
    std::cout << "    Outside code queries: typename Policy::ArchTag\n";
    std::cout << "    This is the type alias export pattern (ex01).\n\n";
    
    std::cout << "A2: ThreadBlockShape: tiles per thread block (CTA level).\n";
    std::cout << "    WarpShape: tiles per warp (subset of thread block).\n";
    std::cout << "    InstructionShape: tensor core instruction dimensions (hardware).\n";
    std::cout << "    Example: ThreadBlock 128x128, Warp 64x64, Instruction 16x8.\n\n";
    
    std::cout << "A3: Nested aliases group related config. One template parameter\n";
    std::cout << "    (Policy) instead of many. Easier to swap policies.\n\n";
    
    std::cout << "A4: 'typename' is required for DEPENDENT NAMES — names that depend\n";
    std::cout << "    on template parameters. Policy::ElementA depends on Policy.\n";
    std::cout << "    Without 'typename', compiler assumes it's a value/function.\n\n";
    
    std::cout << "A5: SharedStorage is __shared__ memory wrapper (RAII).\n";
    std::cout << "    Passed by reference to avoid copying (it's large).\n";
    std::cout << "    Reference ensures all threads see same shared memory.\n\n";
    
    std::cout << "A6: Type trait / type mapping pattern.\n";
    std::cout << "    Maps input type (Element) to output type (GemmType).\n";
    std::cout << "    Similar to std::remove_const<T>::type.\n\n";
    
    std::cout << "A7: Separation of concerns. GemmTypeSelector handles dispatch.\n";
    std::cout << "    Main Gemm class handles computation. Easier to maintain.\n\n";
    
    std::cout << "A8: sizeof...(Stages) returns the NUMBER of types in the pack.\n";
    std::cout << "    Pipeline<Stage0, Stage1, Stage2> has sizeof...(Stages) = 3.\n\n";
    
    std::cout << "A9: Fold expression expands to: process<Stage0>(), process<Stage1>(), process<Stage2>()\n";
    std::cout << "    The comma operator sequences the calls. Unary right fold.\n\n";
}

int main() {
    std::cout << "=== CUTLASS Source Annotation Exercise ===\n";
    std::cout << "\nThis file contains the ANSWERS.\n";
    std::cout << "Try exercise.cpp first, then compare with this file.\n";
    
    print_answers();
    
    std::cout << "=== KEY TAKEAWAY ===\n";
    std::cout << "CUTLASS uses:\n";
    std::cout << "  1. Type alias exports (using X = X)\n";
    std::cout << "  2. Nested policy structs (grouping config)\n";
    std::cout << "  3. typename for dependent names\n";
    std::cout << "  4. Type traits for dispatch\n";
    std::cout << "  5. Variadic templates + fold expressions\n";
    std::cout << "\nAll patterns covered in this curriculum.\n";
    
    return 0;
}
