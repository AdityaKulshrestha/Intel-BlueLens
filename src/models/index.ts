/**
 * Model Implementations Registry
 *
 * Each model parser maps a Hugging Face config to the normalized
 * ModelArchitectureGraph contract defined in contracts.ts.
 * To add a new model: implement a buildXxx() function and add it to the registry.
 */

import { ModelArchitectureGraph } from '../contracts';
import { generateMockArchitecture } from '../performanceData';

export { buildLlama3Architecture } from './llama3';
export { buildDeepSeekV3Architecture } from './deepseekV3';
export { buildQwen3Architecture } from './qwen3';
export { buildPlamo3Architecture } from './plamo3';

import { buildLlama3Architecture } from './llama3';
import { buildDeepSeekV3Architecture } from './deepseekV3';
import { buildQwen3Architecture } from './qwen3';
import { buildPlamo3Architecture } from './plamo3';

type ArchitectureBuilder = (config: any) => ModelArchitectureGraph;

const REGISTRY: Record<string, ArchitectureBuilder> = {
  llama3: buildLlama3Architecture,
  deepseek_v3: buildDeepSeekV3Architecture,
  qwen3_next: buildQwen3Architecture,
  plamo3: buildPlamo3Architecture,
};

/**
 * Returns the normalized architecture graph for a given model ID.
 * Falls back to `generateMockArchitecture` for unregistered models.
 */
export function getModelArchitecture(modelId: string, config: any): ModelArchitectureGraph {
  const builder = REGISTRY[modelId];
  if (builder) return builder(config);
  return generateMockArchitecture(config);
}
