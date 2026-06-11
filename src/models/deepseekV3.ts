import {
  MultiLatentAttention,
  MLPUpProj,
  MLPDownProj,
  RMSNorm,
  SiLU,
  TransformBlock,
  ModelArchitectureGraph,
  Embedding,
  Linear,
} from '../contracts';

// DeepSeek-V3 / R1: MLA attention + ultra-sparse MoE
// 671B total, 37B active params per token, 61 transformer layers
export function buildDeepSeekV3Architecture(config: any): ModelArchitectureGraph {
  const numLayers = config.num_hidden_layers || 61;
  const blocks: TransformBlock[] = [];

  for (let i = 0; i < numLayers; i++) {
    blocks.push({
      attention_norm: new RMSNorm(config.hidden_size, 1e-6),
      // Multi-Head Latent Attention: compresses Q/K/V into shared latent space
      // kv_lora_rank=512, q_lora_rank=1536 (standard DeepSeek-V3 config)
      attention: new MultiLatentAttention(
        config.num_attention_heads || 128,
        512,   // kv_lora_rank
        1536   // q_lora_rank
      ),
      mlp_gate: new MLPUpProj(config.hidden_size, config.intermediate_size || 18432),
      mlp_up: new MLPUpProj(config.hidden_size, config.intermediate_size || 18432),
      activation: new SiLU(),
      mlp_down: new MLPDownProj(config.intermediate_size || 18432, config.hidden_size),
    });
  }

  return {
    model_name: 'DeepSeek-V3',
    model_type: 'deepseek_v3',
    hidden_size: config.hidden_size || 7168,
    num_layers: numLayers,
    vocab_search: new Embedding(config.vocab_size || 128256, config.hidden_size || 7168),
    blocks,
    final_norm: new RMSNorm(config.hidden_size || 7168, 1e-6),
    lm_head: new Linear(config.hidden_size || 7168, config.vocab_size || 128256),
  };
}
