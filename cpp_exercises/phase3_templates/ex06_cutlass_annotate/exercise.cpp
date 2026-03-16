// CONCEPT: Reading CUTLASS source — annotate opaque sections
// FORMAT: ANNOTATE
// TIME_TARGET: 30 min
// WHY_THIS_MATTERS: CUTLASS 3.x source is dense with templates. You must decode it.
// CUDA_CONNECTION: Real CUTLASS SM90 collective kernel source.

#include <iostream>

// Below is a SIMPLIFIED excerpt from CUTLASS 3.x (cutlass/include/cutlass/gemm/collective/)
// Real CUTLASS is more complex, but this captures the essential patterns.

// ANNOTATE EXERCISE: Read the code and answer questions below each section

// ==================== Section 1: Policy Struct ====================

/*
template<class ArchTag, class ElementA, class ElementB, class ElementC, class Layout>
struct CollectiveGemmPolicy {
  using ArchTag = ArchTag;                              // ANNOTATE: What is this?
  using ElementA = ElementA;                            // ANNOTATE: Why export these?
  using ElementB = ElementB;
  using ElementC = ElementC;
  using Layout = Layout;
  
  using ThreadBlockShape = cutlass::gemm::GemmShape<128, 128, 64>;  // ANNOTATE: M, N, K?
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;           // ANNOTATE: Relation to ThreadBlockShape?
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;     // ANNOTATE: Hardware-specific?
};
*/

// Questions for Section 1:
// Q1: Why does the policy use `using Name = Name;` (e.g., `using ArchTag = ArchTag;`)?
//     Hint: Think about how outside code queries these types.
//
// Q2: What do ThreadBlockShape, WarpShape, InstructionShape represent?
//     Hint: Think about the CUDA thread hierarchy and tensor core dimensions.
//
// Q3: Why are these nested type aliases instead of template parameters?
//     Hint: Think about grouping related configuration.


// ==================== Section 2: Kernel with CRTP ====================

/*
template<class ProblemShape, class Policy, class TileIterator>
class CollectiveGemm {
public:
  using ElementA = typename Policy::ElementA;    // ANNOTATE: Why typename?
  using ElementB = typename Policy::ElementB;
  using ElementC = typename Policy::ElementC;
  
  using SharedStorage = typename Policy::SharedStorage;  // ANNOTATE: What is this for?
  
  __device__ void operator()(
      ProblemShape problem_shape,
      ElementA* ptr_A,
      ElementB* ptr_B,
      ElementC* ptr_C,
      SharedStorage& shared_storage)  // ANNOTATE: Why reference?
  {
    // Kernel implementation...
  }
};
*/

// Questions for Section 2:
// Q4: Why is `typename` required before `Policy::ElementA`?
//     Hint: Think about dependent names in templates.
//
// Q5: What is SharedStorage and why is it passed by reference?
//     Hint: Think about CUDA shared memory and avoiding copies.


// ==================== Section 3: Type Dispatch ====================

/*
template<class Element>
struct GemmTypeSelector;

template<>
struct GemmTypeSelector<float> {
  using Type = GemmFloat;    // ANNOTATE: What pattern is this?
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
*/

// Questions for Section 3:
// Q6: What design pattern is `GemmTypeSelector`?
//     Hint: Think about mapping types to types.
//
// Q7: Why use this instead of template specialization on the main Gemm class?
//     Hint: Think about separating concerns.


// ==================== Section 4: Parameter Pack Expansion ====================

/*
template<class... Stages>
class Pipeline {
  static constexpr int NumStages = sizeof...(Stages);  // ANNOTATE: What does this do?
  
  template<int Stage>
  __device__ void process() {
    // Process stage...
  }
  
  __device__ void run() {
    // ANNOTATE: What does this fold expression do?
    (process<Stages>(), ...);  
  }
};

// Usage:
// Pipeline<Stage0, Stage1, Stage2> pipeline;
// pipeline.run();  // Calls process<0>(), process<1>(), process<2>()
*/

// Questions for Section 4:
// Q8: What does `sizeof...(Stages)` return?
//     Hint: Think about parameter packs.
//
// Q9: What does the fold expression `(process<Stages>(), ...)` expand to?
//     Hint: Think about comma operator and pack expansion.


// ==================== Answers (DO NOT LOOK UNTIL YOU'VE TRIED) ====================

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
    std::cout << "\nRead the commented code sections above.\n";
    std::cout << "Answer questions Q1-Q9 in your own words.\n";
    std::cout << "Then check answers below.\n";
    
    std::cout << "\nPress Enter to reveal answers...";
    std::cin.get();
    
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
