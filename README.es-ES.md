

# Aprendizaje de Kernels de IA

Un repositorio integral dedicado a dominar la programación de kernels de GPU, primitivas de computación de alto rendimiento (HPC) y los fundamentos matemáticos de las arquitecturas modernas de IA. Este proyecto sirve como una ruta de aprendizaje estructurada desde los conceptos fundamentales de CUDA hasta optimizaciones avanzadas utilizando bibliotecas estándar de la industria como CUTLASS y Triton.

---

## Estructura del Repositorio y Análisis Profundo de Módulos

### 1. [Ruta de Aprendizaje de CUDA](./Cuda/) (Fundamentos a Dominio)
*Plan de estudios completo que cubre todo el ecosistema de CUDA, desde la jerarquía de hilos hasta optimizaciones específicas del hardware.*
- **[Fundamentos](./Cuda/fundamentals/):** Conceptos centrales que incluyen Grillas (Grids), Bloques (Blocks), Hilos (Threads) y la Jerarquía de Memoria (Global vs. Compartida vs. Registros).
- **[Optimización de Memoria](./Cuda/memory_optimization/):** Técnicas para maximizar el ancho de banda mediante Coalescencia, evitando Conflictos de Bancos de Memoria Compartida e implementando Swizzling.
- **[Optimización de Ejecución](./Cuda/execution_optimization/):** Equilibrar recursos con análisis de Ocupación y utilizar Primitivas a Nivel de Warp (`__shfl_sync`) para comunicación intra-warp.
- **[Conceptos Avanzados](./Cuda/advanced_concepts/):** Copias asíncronas (`cp.async`), programación de Tensor Cores y Pipelining de Software para ocultar la latencia de memoria.
- **[Cudalings](./Cuda/cudalings/):** Una serie de ejercicios interactivos a pequeña escala para reforzar la sintaxis y lógica de CUDA.

### 2. [Metaprogramación de Plantillas](./Template_Metaprogramming/) (C++ Avanzado)
*Un prerrequisito para el desarrollo avanzado de bibliotecas, este módulo cubre los patrones de plantillas de C++ necesarios para comprender y contribuir a bibliotecas como CUTLASS y CuTE.*

### 3. [Ruta de Aprendizaje de CUTLASS 3.x](./Cutlass_3x/) (Biblioteca de Plantillas C++ para GEMM)
*Ejemplos en profundidad para CUTLASS 3.x, la biblioteca fundamental de NVIDIA para construir kernels de alto rendimiento específicos para cada arquitectura.*
- **[GEMM Básico](./Cutlass_3x/00_basic_gemm/):** Comprender la estructura fundamental del kernel de Multiplicación General Matriz-Matriz (GEMM).
- **[Especialización de Arquitectura](./Cutlass_3x/):** Explorando ejemplos adaptados para las arquitecturas Volta, Turing, Ampere, Hopper y Blackwell (p. ej., `14_ampere_tf32_tensorop_gemm`, `48_hopper_warp_specialized_gemm`, `70_blackwell_gemm`).
- **[Características Avanzadas](./Cutlass_3x/):** Implementación de Operaciones Colectivas, Visitantes de Epílogo (Epilogue Visitors), Stream-K, GEMM Disperso (Sparse), Operaciones Fusionadas (p. ej., GEMM+Softmax) y GEMM Distribuido en múltiples GPUs.
- **[Integración con Python](./Cutlass_3x/40_cutlass_py/):** Técnicas para interactuar con kernels de CUTLASS desde Python.

### 4. [Ruta de Aprendizaje de CuTE & CuTe-DSL](./CuTE/) (C++20 Moderno para Kernels de LLM)
*Un plan de estudios práctico para construir kernels de LLM de grado de producción utilizando CuTE, el sucesor moderno en C++20 de los conceptos de layout y tensores de CUTLASS.*
- **[Módulo 01: Layouts](./CuTE/Module_01_Layouts/):** Dominar el álgebra de layouts (`make_layout`), teselado jerárquico y patrones para GQA (Grouped Query Attention).
- **[Módulo 02: Tensores](./CuTE/Module_02_Tensors/):** Trabajo con vistas de tensores, segmentación por cabeza (per-head slicing) e iteración de bloques con `local_tile`.
- **[Módulo 03-04: Operaciones Teseladas](./CuTE/Module_03_TiledCopy/):** Uso de cargas vectorizadas de 128 bits, pipelinas `cp.async` y Átomos MMA para aceleración con Tensor Cores.
- **[Módulo 05-06: Enchufado y Swizzling](./CuTE/Module_05_Swizzle/):** Diseño de swizzling libre de conflictos de banco y pipelinas asíncronas multietapa.
- **[CuTe-DSL](./CuTe-DSL/):** Un Lenguaje Específico de Dominio (DSL) basado en Python para expresar conceptos de CuTE con mayor facilidad y productividad.
- **[CuTile](./CuTile/):** Una biblioteca complementaria y tutorial centrado en estrategias avanzadas de teselado.
- **[Proyectos](./CuTE/Projects/):** Implementaciones finales de un **Tiled GEMM** de alto rendimiento y el kernel de **FlashAttention-2 Prefill**.

### 5. [CUTLASS 4.x y Python](./Cutlass_4x/) (Generación de Kernels a Alto Nivel con Python)
*Explorando la próxima generación de CUTLASS, que enfatiza flujos de trabajo primero en Python y ajuste automático de rendimiento.*
- **[Operaciones a Alto Nivel](./Cutlass_4x/01_high_level_ops/):** Definición y personalización de kernels utilizando constructores intuitivos y de alto nivel en Python.
- **[Autotuning](./Cutlass_4x/02_autotuning/):** Aprovechando marcos integrados para descubrir automáticamente configuraciones de kernel óptimas para el hardware objetivo.

### 6. [Ruta de Aprendizaje de Triton](./Triton/) (Desarrollo de Kernels Primero en Python)
*Módulos estructurados para escribir kernels de GPU de alto rendimiento desde cero utilizando el DSL de Triton en Python.*
- **[Módulo 01-04: Fundamentos y Teselado](./Triton/Module-01-Basics/):** Suma vectorial, operaciones de memoria, verificaciones de límites y la filosofía central de la programación basada en bloques.
- **[Módulo 05: Multiplicación de Matrices](./Triton/Module-05-Matrix-Multiplication/):** Implementación paso a paso de Tiled GEMM y estrategias de optimización.
- **[Módulo 06-08: Técnicas Avanzadas](./Triton/Module-06-Advanced-Memory/):** Optimización de jerarquía de caché, reducciones paralelas y garantía de estabilidad numérica.
- **[Optimización y Dominio](./Triton/optimization/):** Análisis en profundidad de enchufado (pipelining), memoria compartida, especialización de warp y construcción de kernels de atención personalizados.

### 7. [Matemáticas del Transformer](./transformer_math/) (Fundamentos Teóricos)
*El "por qué" matemático detrás de los kernels modernos de IA, con enfoque en la inferencia de Modelos de Lenguaje Grande (LLM).*
- **[Atención y KV Cache](./transformer_math/01_attention/):** Derivaciones de producto punto escalado, enmascarado causal y el problema de "Pared de Memoria" (Memory Wall) de la caché KV.
- **[FlashAttention](./transformer_math/05_flash_attention/):** Ideas sobre teselado, el problema de E/S y la formulación matemática de Online Softmax.
- **[Variantes Arquitectónicas](./transformer_math/03_attention_variants/):** Análisis de MHA, MQA, GQA y sus implicaciones en el rendimiento del hardware.
- **[Análisis de Sistemas](./transformer_math/10_arithmetic_intensity/):** Uso de modelos Roofline para analizar el rendimiento de decodificación vs. prellenado (prefill) y el impacto del tamaño de lote (batch size).

### 8. [Estructuras de Datos y Algoritmos](./DSA/) (Implementaciones Paralelas)
*Algoritmos clásicos y modernos rediseñados para la ejecución masivamente paralela en GPU.*
- **[Fundamentos en CPU](./DSA/):** Conceptos centrales de estructuras de datos y algoritmos (que sirven como línea base).
- **[Primitivas de GPU](./GPU-DSA/):** Reducciones paralelas de alto rendimiento, sumas prefijas (scans) e indexación ordenada por Morton (curvas Z).
- **[Ordenamiento en GPU](./GPU-DSA/):** Implementaciones de kernels de Radix Sort y Bitonic Sort.
- **[Kernels Avanzados](./GPU-DSA/):** LayerNorm fusionado (Fused LayerNorm), Online Softmax, Paged Attention (estilo vLLM) y patrones de doble almacenamiento intermedio (Double Buffering).

### 9. [Ruta de Aprendizaje de Candle](./Candle/) (ML en Rust + Optimización de Inferencia)
*Un plan de estudios completo de completar código para Hugging Face Candle, que avanza desde los fundamentos de tensores hasta la optimización consciente de CUDA/Tensor Core y el diseño de inferencia de mini-LLM.*
- **[Cuaderno de Trabajo de Candle](./Candle/README.md):** Ejercicios guiados que cubren tensores, autograd, construcción de modelos, bloques transformer, caché KV, cuantización y optimización basada en perfiles.
- **[Ruta de Rendimiento Avanzado](./Candle/README.md):** Análisis comparativo de GEMM consciente de Tensor Cores, estrategia de tipo de datos (`F16`/`BF16`), consideraciones de layout de memoria y análisis de cuellos de botella basado en Nsight.

### 10. [Lenguajes y Ecosistemas de Soporte](./)
- **[Programación del Lado del Anfitrión](./Concurrency/):** Hilos múltiples en C++, primitivas de sincronización y conducción de cargas de trabajo asíncronas de CUDA.
- **[Habilidades en Python](./python_crashcourse/):** Un curso intensivo y análisis conceptuales en profundidad de Python para HPC.
- **[Lenguaje Mojo](./MojoLang/):** Proyectos exploratorios utilizando Mojo, un nuevo lenguaje diseñado para programación de IA de alto rendimiento.
- **[Ecosistema de Construcción, Depuración y Perfilado](./)**
    - **[Sistema de Construcción CMake](./Cmake-guide/):** Una guía completa para construir proyectos complejos de CUDA/C++ con CMake moderno.
    - **[Ensamblador PTX](./PTX/):** Programación a bajo nivel de ISA virtual para depuración y optimización de grano fino.
    - **[Herramientas de Perfilado](./Profiling/):** Módulos sobre el uso de Nsight Compute (NCU) y Nsight Systems (NSYS) para identificar y resolver cuellos de botellas de rendimiento.
- **[Sistemas Multi-GPU](./NCCL/):**
    - **[NCCL](./NCCL/):** Módulos que cubren primitivas de comunicación multi-GPU y multi-nodo (AllReduce, AllGather) y los algoritmos subyacentes de anillo/árbol.

---

## Herramientas y Entorno
*   **Compiladores:** `nvcc` (CUDA Toolkit 12.x+), `clang++` (C++17/20).
*   **Bibliotecas:** CUTLASS (3.x, 4.x), CuTE, Triton, NCCL.
*   **Perfilado:** NVIDIA Nsight Compute y Nsight Systems.
*   **Objetivo de Hardware:** Ejemplos y optimizaciones dirigidas a NVIDIA Ampere (SM80), Ada Lovelace (SM89), Hopper (SM90) y Blackwell (SM100-series).

## Cómo Empezar
1.  Para desarrollo enfocado en C++, comience con **[CUDA](./Cuda/)**, luego **[Metaprogramación de Plantillas](./Template_Metaprogramming/)**, seguido por **[CUTLASS 3.x](./Cutlass_3x/)** y **[CuTE](./CuTE/)**.
2.  Para un enfoque primero en Python, comience con **[Habilidades en Python](./python_crashcourse/)**, luego sumérjase en la **[Ruta de Aprendizaje de Triton](./Triton/)**.
3.  Consulte los módulos de **[Matemáticas del Transformer](./transformer_math/)** para comprender la teoría detrás de los kernels que está construyendo.
4.  Utilice las herramientas de **[Perfilado](./Profiling/)** de manera continua para medir su progreso y validar el impacto de las optimizaciones.
5.  Para desarrollo e inferencia de modelos nativos en Rust, siga la **[Ruta de Aprendizaje de Candle](./Candle/README.md)**.
