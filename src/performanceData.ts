import {
  GroupedQueryAttention,
  ModelArchitectureGraph,
  MLPUpProj,
  MLPDownProj,
  RMSNorm,
  SiLU,
  TransformBlock,
  HardwareMapping,
  PerfettoSpan,
  Embedding,
  Linear
} from './contracts';

// Generate dynamic mock architecture
export const generateMockArchitecture = (config: any): ModelArchitectureGraph => {
  const NUM_LAYERS = config.num_hidden_layers || 32;
  const blocks: TransformBlock[] = [];
  
  for (let i = 0; i < NUM_LAYERS; i++) {
    blocks.push({
      attention_norm: new RMSNorm(config.hidden_size, 1e-5),
      attention: new GroupedQueryAttention(config.num_attention_heads, config.num_key_value_heads || config.num_attention_heads),
      mlp_norm: new RMSNorm(config.hidden_size, 1e-5), // Some have mlp norm separately
      mlp_gate: new MLPUpProj(config.hidden_size, config.intermediate_size),
      mlp_up: new MLPUpProj(config.hidden_size, config.intermediate_size),
      activation: new SiLU(),
      mlp_down: new MLPDownProj(config.intermediate_size, config.hidden_size),
    });
  }

  return {
    model_name: config.model_type,
    model_type: config.model_type,
    hidden_size: config.hidden_size,
    num_layers: NUM_LAYERS,
    vocab_search: new Embedding(config.vocab_size, config.hidden_size),
    blocks,
    final_norm: new RMSNorm(config.hidden_size, 1e-5),
    lm_head: new Linear(config.hidden_size, config.vocab_size)
  };
};

/** Legacy flat list — kept for backward compatibility */
export const getHardwareMappings = (): HardwareMapping[] =>
  getHardwareMappingsByDevice().CPU;

/**
 * Hardware kernel mappings grouped by device target.
 *
 * CPU  → vllm/csrc/cpu  (torch ops registered via torch_bindings.cpp)
 *        GEMM via onednn_mm / weight_packed_linear (AVX-512BF16+VNNI)
 *        Attention via cpu_attention_with_kv_cache (AMX / AVX-512 dispatch)
 *        MLA via mla_decode_kvcache
 *        GDN via chunk_gated_delta_rule_cpu / fused_gdn_gating_cpu
 *
 * XPU  → vllm-xpu-kernels/csrc  (SYCL/DPC++, registered via torch_bindings.cpp)
 *        Attention via flash_attn (csrc/flash_attn) or GDN attn (csrc/attention)
 *        GEMM via grouped_gemm + oneDNN (csrc/xpu)
 *        Norm / activation via layernorm.cpp / activation.cpp
 *
 * GPU  → vLLM CUDA kernels (FlashAttn v3, CUTLASS, Triton, cuBLAS)
 */
export const getHardwareMappingsByDevice = (): Record<'CPU' | 'XPU' | 'GPU', HardwareMapping[]> => ({
  CPU: [
    // csrc/cpu/torch_bindings.cpp — no op registered for plain embedding; falls back to PyTorch ATen gather
    { component_type: 'Embedding',              hardware_kernel: 'aten::embedding (ATen scatter)', execution_engine: 'PyTorch ATen',  instruction_set: 'AVX2',              device: 'CPU' },
    // rms_norm / fused_add_rms_norm  (layernorm.cpp)
    { component_type: 'RMSNorm',                hardware_kernel: 'rms_norm / fused_add_rms_norm',  execution_engine: 'vLLM CPU',      instruction_set: 'AVX-512',           device: 'CPU' },
    // cpu_attention_with_kv_cache  (cpu_attn.cpp + cpu_attn_amx.hpp)
    { component_type: 'GroupedQueryAttention',  hardware_kernel: 'cpu_attention_with_kv_cache',    execution_engine: 'vLLM CPU',      instruction_set: 'AMX-BF16 / AVX-512',device: 'CPU' },
    { component_type: 'MultiHeadAttention',     hardware_kernel: 'cpu_attention_with_kv_cache',    execution_engine: 'vLLM CPU',      instruction_set: 'AVX-512',           device: 'CPU' },
    // mla_decode_kvcache  (mla_decode.cpp)
    { component_type: 'MultiLatentAttention',   hardware_kernel: 'mla_decode_kvcache',             execution_engine: 'vLLM CPU',      instruction_set: 'AMX-BF16',          device: 'CPU' },
    // cpu_attention_with_kv_cache with sliding_window_left/right args
    { component_type: 'SlidingWindowAttention', hardware_kernel: 'cpu_attention_with_kv_cache (SWA)',execution_engine: 'vLLM CPU',    instruction_set: 'AVX-512',           device: 'CPU' },
    // chunk_gated_delta_rule_cpu + fused_gdn_gating_cpu  (sgl-kernels GDN)
    { component_type: 'GatedDeltaNet',          hardware_kernel: 'chunk_gated_delta_rule_cpu',     execution_engine: 'vLLM CPU (sgl)',instruction_set: 'AVX-512BF16+VNNI', device: 'CPU' },
    // weight_packed_linear (AVX-512BF16+VNNI) or onednn_mm fallback  (micro_gemm / dnnl_kernels.cpp)
    { component_type: 'MLPUpProj',              hardware_kernel: 'weight_packed_linear / onednn_mm',execution_engine: 'vLLM CPU / oneDNN',instruction_set: 'AMX-BF16 / AVX-512BF16',device: 'CPU' },
    { component_type: 'MLPDownProj',            hardware_kernel: 'weight_packed_linear / onednn_mm',execution_engine: 'vLLM CPU / oneDNN',instruction_set: 'AMX-BF16 / AVX-512BF16',device: 'CPU' },
    { component_type: 'MLP',                    hardware_kernel: 'weight_packed_linear / onednn_mm',execution_engine: 'vLLM CPU / oneDNN',instruction_set: 'AMX-BF16 / AVX-512BF16',device: 'CPU' },
    { component_type: 'Linear',                 hardware_kernel: 'onednn_mm',                      execution_engine: 'oneDNN',        instruction_set: 'AMX-BF16',          device: 'CPU' },
    // silu_and_mul  (activation.cpp)
    { component_type: 'SiLU',                   hardware_kernel: 'silu_and_mul',                   execution_engine: 'vLLM CPU',      instruction_set: 'AVX-512',           device: 'CPU' },
    // gelu_tanh_and_mul / gelu_and_mul  (activation.cpp)
    { component_type: 'GELU',                   hardware_kernel: 'gelu_tanh_and_mul / gelu_fast',  execution_engine: 'vLLM CPU',      instruction_set: 'AVX-512',           device: 'CPU' },
    // gelu_tanh_and_mul  (activation.cpp)
    { component_type: 'GeGLU',                  hardware_kernel: 'gelu_tanh_and_mul',              execution_engine: 'vLLM CPU',      instruction_set: 'AVX-512',           device: 'CPU' },
    { component_type: 'ReLU',                   hardware_kernel: 'aten::relu (ATen)',               execution_engine: 'PyTorch ATen',  instruction_set: 'AVX2',              device: 'CPU' },
  ],
  XPU: [
    // vllm-xpu-kernels: no custom embedding kernel; ATen fallback
    { component_type: 'Embedding',              hardware_kernel: 'aten::embedding (ATen scatter)', execution_engine: 'PyTorch ATen',  instruction_set: 'EU Vec (BF16)',     device: 'XPU' },
    // rms_norm / fused_add_rms_norm  (csrc/layernorm.cpp)
    { component_type: 'RMSNorm',                hardware_kernel: 'rms_norm / fused_add_rms_norm',  execution_engine: 'vllm-xpu-kernels',instruction_set: 'SYCL EU',         device: 'XPU' },
    // Flash attention varlen  (csrc/flash_attn/)
    { component_type: 'GroupedQueryAttention',  hardware_kernel: 'flash_attn_varlen_func (XPU)',   execution_engine: 'vllm-xpu-kernels',instruction_set: 'XMX (BF16)',       device: 'XPU' },
    { component_type: 'MultiHeadAttention',     hardware_kernel: 'flash_attn_varlen_func (XPU)',   execution_engine: 'vllm-xpu-kernels',instruction_set: 'XMX (BF16)',       device: 'XPU' },
    // MLA: concat_and_cache_mla + merge_attn_states  (csrc/attention/ + torch_bindings.cpp)
    { component_type: 'MultiLatentAttention',   hardware_kernel: 'concat_and_cache_mla + merge_attn_states',execution_engine: 'vllm-xpu-kernels',instruction_set: 'XMX (BF16)', device: 'XPU' },
    // GDN attention  (csrc/attention/ — GDN attn kernel)
    { component_type: 'SlidingWindowAttention', hardware_kernel: 'flash_attn_varlen_func (SWA)',   execution_engine: 'vllm-xpu-kernels',instruction_set: 'XMX (BF16)',       device: 'XPU' },
    { component_type: 'GatedDeltaNet',          hardware_kernel: 'gdn_attn (SYCL chunk scan)',     execution_engine: 'vllm-xpu-kernels',instruction_set: 'SYCL EU Vec',      device: 'XPU' },
    // Grouped GEMM  (csrc/xpu/ + oneDNN)
    { component_type: 'MLPUpProj',              hardware_kernel: 'grouped_gemm / oneDNN GEMM (XPU)',execution_engine: 'vllm-xpu-kernels / oneDNN',instruction_set: 'XMX (BF16)',device: 'XPU' },
    { component_type: 'MLPDownProj',            hardware_kernel: 'grouped_gemm / oneDNN GEMM (XPU)',execution_engine: 'vllm-xpu-kernels / oneDNN',instruction_set: 'XMX (BF16)',device: 'XPU' },
    { component_type: 'MLP',                    hardware_kernel: 'grouped_gemm / oneDNN GEMM (XPU)',execution_engine: 'vllm-xpu-kernels / oneDNN',instruction_set: 'XMX (BF16)',device: 'XPU' },
    { component_type: 'Linear',                 hardware_kernel: 'oneDNN GEMM (XPU)',              execution_engine: 'oneDNN (XPU)',  instruction_set: 'XMX (BF16)',        device: 'XPU' },
    // silu_and_mul / silu_and_mul_quant  (csrc/activation.cpp)
    { component_type: 'SiLU',                   hardware_kernel: 'silu_and_mul',                   execution_engine: 'vllm-xpu-kernels',instruction_set: 'SYCL EU Vec',     device: 'XPU' },
    // gelu_and_mul / gelu_tanh_and_mul  (csrc/activation.cpp)
    { component_type: 'GELU',                   hardware_kernel: 'gelu_tanh_and_mul / gelu_fast',  execution_engine: 'vllm-xpu-kernels',instruction_set: 'SYCL EU Vec',     device: 'XPU' },
    { component_type: 'GeGLU',                  hardware_kernel: 'gelu_tanh_and_mul',              execution_engine: 'vllm-xpu-kernels',instruction_set: 'SYCL EU Vec',     device: 'XPU' },
    { component_type: 'ReLU',                   hardware_kernel: 'aten::relu (ATen)',               execution_engine: 'PyTorch ATen',  instruction_set: 'SYCL EU Vec',      device: 'XPU' },
  ],
  GPU: [
    { component_type: 'Embedding',              hardware_kernel: 'CUDA_Embedding_FwdKernel',       execution_engine: 'PyTorch',        instruction_set: 'CUDA Cores',        device: 'GPU' },
    { component_type: 'RMSNorm',                hardware_kernel: 'vLLM rms_norm (CUDA)',           execution_engine: 'vLLM CUDA',      instruction_set: 'CUDA FP16',         device: 'GPU' },
    { component_type: 'GroupedQueryAttention',  hardware_kernel: 'FlashAttention-3 GQA varlen',    execution_engine: 'FlashAttn v3',   instruction_set: 'Tensor Core BF16',  device: 'GPU' },
    { component_type: 'MultiHeadAttention',     hardware_kernel: 'FlashAttention-3 MHA varlen',    execution_engine: 'FlashAttn v3',   instruction_set: 'Tensor Core FP16',  device: 'GPU' },
    { component_type: 'MultiLatentAttention',   hardware_kernel: 'DeepGEMM MLA FP8 GEMM',         execution_engine: 'DeepGEMM',       instruction_set: 'FP8 Tensor Core',   device: 'GPU' },
    { component_type: 'SlidingWindowAttention', hardware_kernel: 'FlashAttn-3 SWA varlen',         execution_engine: 'FlashAttn v3',   instruction_set: 'Tensor Core BF16',  device: 'GPU' },
    { component_type: 'GatedDeltaNet',          hardware_kernel: 'Triton chunk_gated_delta_rule',  execution_engine: 'Triton',         instruction_set: 'Tensor Core BF16',  device: 'GPU' },
    { component_type: 'MLPUpProj',              hardware_kernel: 'CUTLASS grouped_gemm FP16',      execution_engine: 'CUTLASS 3.x',    instruction_set: 'Tensor Core FP16',  device: 'GPU' },
    { component_type: 'MLPDownProj',            hardware_kernel: 'CUTLASS grouped_gemm FP16',      execution_engine: 'CUTLASS 3.x',    instruction_set: 'Tensor Core FP16',  device: 'GPU' },
    { component_type: 'MLP',                    hardware_kernel: 'CUTLASS grouped_gemm FP16',      execution_engine: 'CUTLASS 3.x',    instruction_set: 'Tensor Core FP16',  device: 'GPU' },
    { component_type: 'Linear',                 hardware_kernel: 'cuBLAS HGEMM',                   execution_engine: 'cuBLAS',         instruction_set: 'Tensor Core FP16',  device: 'GPU' },
    { component_type: 'SiLU',                   hardware_kernel: 'Triton silu_and_mul fused',      execution_engine: 'Triton',         instruction_set: 'CUDA FP16',         device: 'GPU' },
    { component_type: 'GELU',                   hardware_kernel: 'Triton gelu_tanh_and_mul',       execution_engine: 'Triton',         instruction_set: 'CUDA FP16',         device: 'GPU' },
    { component_type: 'GeGLU',                  hardware_kernel: 'Triton gelu_tanh_and_mul',       execution_engine: 'Triton',         instruction_set: 'CUDA FP16',         device: 'GPU' },
    { component_type: 'ReLU',                   hardware_kernel: 'cuDNN ReLU inplace',             execution_engine: 'cuDNN',          instruction_set: 'CUDA Cores',        device: 'GPU' },
  ],
});

export const generateMockSpans = (arch: ModelArchitectureGraph): PerfettoSpan[] => {
  const spans: PerfettoSpan[] = [];
  let currentMs = 0;
  
  // Embedding
  const embDuration = 1.0;
  spans.push({
    name: 'Embedding Search',
    category: `Initialization`,
    start_ms: currentMs,
    duration_ms: embDuration,
    pid: 1,
    tid: 1,
    component_ref: arch.vocab_search,
  });
  currentMs += embDuration;

  arch.blocks.forEach((block, layerIdx) => {
    // Norm
    const normDuration = 0.5 + Math.random() * 0.2;
    spans.push({
      name: 'RMSNorm',
      category: `Layer_${layerIdx}`,
      start_ms: currentMs,
      duration_ms: normDuration,
      pid: 1,
      tid: 1,
      component_ref: block.attention_norm,
    });
    currentMs += normDuration;

    // Attention
    const attnDuration = 2.0 + Math.random() * 1.5;
    spans.push({
      name: 'GQA',
      category: `Layer_${layerIdx}`,
      start_ms: currentMs,
      duration_ms: attnDuration,
      pid: 1,
      tid: 1,
      component_ref: block.attention,
    });
    currentMs += attnDuration;

    // MLP
    const mlpDuration = 3.0 + Math.random() * 2.0;
    spans.push({
      name: 'MLP Gate+Up (GEMM)',
      category: `Layer_${layerIdx}`,
      start_ms: currentMs,
      duration_ms: mlpDuration,
      pid: 1,
      tid: 1,
      component_ref: block.mlp_up,
    });
    
    // Activation concurrently
    const actStart = currentMs + 1.0; 
    spans.push({
      name: 'SiLU',
      category: `Layer_${layerIdx}`,
      start_ms: actStart,
      duration_ms: 0.5,
      pid: 1,
      tid: 2, 
      component_ref: block.activation,
    });

    currentMs += mlpDuration; 
    
    const mlpDownDuration = 2.5 + Math.random() * 1.5;
    spans.push({
      name: 'MLP Down (GEMM)',
      category: `Layer_${layerIdx}`,
      start_ms: currentMs,
      duration_ms: mlpDownDuration,
      pid: 1,
      tid: 1,
      component_ref: block.mlp_down,
    });
    currentMs += mlpDownDuration;
  });
  
  // Final Norm
  spans.push({
    name: 'RMSNorm',
    category: `Final`,
    start_ms: currentMs,
    duration_ms: 0.5,
    pid: 1,
    tid: 1,
    component_ref: arch.final_norm,
  });
  currentMs += 0.5;
  
  // LM Head
  spans.push({
    name: 'LM Head (Linear)',
    category: `Final`,
    start_ms: currentMs,
    duration_ms: 2.0,
    pid: 1,
    tid: 1,
    component_ref: arch.lm_head,
  });
  
  return spans;
};
