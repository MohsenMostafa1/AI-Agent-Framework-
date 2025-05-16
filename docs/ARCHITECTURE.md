# AI Agent Framework Architecture

## Core Components
### Quantization Layer
- Supports 4/8-bit via `bitsandbytes` (LLM.int8()) and GPTQ/AWQ
- Automatic precision selection based on CUDA capability (see `cuda_compatibility.yaml`)
- Fallback to CPU inference with INT8 quantization

### Memory Subsystem
```mermaid
graph TD
    A[Episodic Memory] --> C[Working Buffer]
    B[Long-term Memory] --> C
    C --> D[ColBERT RAG Retriever]
    D --> E[LLM Context Window]
