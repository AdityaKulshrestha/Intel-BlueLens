import {
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

export function buildLlama3Architecture(config: any): ModelArchitectureGraph {
  const numLayers = config.num_hidden_layers || 32;
  const blocks: TransformBlock[] = [];

  for (let i = 0; i < numLayers; i++) {
    blocks.push({
      attention_norm: new RMSNorm(config.hidden_size, config.rms_norm_eps || 1e-5),
      attention: new GroupedQueryAttention(
        config.num_attention_heads || 32,
        config.num_key_value_heads ?? config.num_attention_heads ?? 32
      ),
      mlp_gate: new MLPUpProj(config.hidden_size, config.intermediate_size),
      mlp_up: new MLPUpProj(config.hidden_size, config.intermediate_size),
      activation: new SiLU(),
      mlp_down: new MLPDownProj(config.intermediate_size, config.hidden_size),
    });
  }

  return {
    model_name: 'LLaMA 3',
    model_type: 'llama',
    hidden_size: config.hidden_size,
    num_layers: numLayers,
    vocab_search: new Embedding(config.vocab_size, config.hidden_size),
    blocks,
    final_norm: new RMSNorm(config.hidden_size, config.rms_norm_eps || 1e-5),
    lm_head: new Linear(config.hidden_size, config.vocab_size),
  };
}
