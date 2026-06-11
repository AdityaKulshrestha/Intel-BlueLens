import {
  GatedDeltaNet,
  GroupedQueryAttention,
  MLPUpProj,
  MLPDownProj,
  RMSNorm,
  SiLU,
  TransformBlock,
  ModelArchitectureGraph,
  Embedding,
  Linear,
} from '../contracts';

// Qwen3-Next: 80B total / 3B active MoE
// Hybrid 3:1 ratio — every 4th layer is Global Gated Attention, rest are Gated DeltaNet
export function buildQwen3Architecture(config: any): ModelArchitectureGraph {
  const numLayers = config.num_hidden_layers || 64;
  const blocks: TransformBlock[] = [];

  for (let i = 0; i < numLayers; i++) {
    // 3:1 ratio: layers 3, 7, 11, ... are global softmax attention
    const isGlobalAttention = (i + 1) % 4 === 0;

    blocks.push({
      attention_norm: new RMSNorm(config.hidden_size, 1e-6),
      attention: isGlobalAttention
        ? new GroupedQueryAttention(
            config.num_attention_heads || 64,
            config.num_key_value_heads || 4
          )
        : new GatedDeltaNet({ hidden_size: config.hidden_size, layer_idx: i }),
      mlp_gate: new MLPUpProj(config.hidden_size, config.intermediate_size || 24576),
      mlp_up: new MLPUpProj(config.hidden_size, config.intermediate_size || 24576),
      activation: new SiLU(),
      mlp_down: new MLPDownProj(config.intermediate_size || 24576, config.hidden_size),
    });
  }

  return {
    model_name: 'Qwen3-Next',
    model_type: 'qwen3_next',
    hidden_size: config.hidden_size || 8192,
    num_layers: numLayers,
    vocab_search: new Embedding(config.vocab_size || 152064, config.hidden_size || 8192),
    blocks,
    final_norm: new RMSNorm(config.hidden_size || 8192, 1e-6),
    lm_head: new Linear(config.hidden_size || 8192, config.vocab_size || 152064),
  };
}
