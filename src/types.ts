export interface SubItem {
  name: string;
  description: string;
}

export interface Category {
  name: string;
  description?: string;
  items: SubItem[];
}

export interface LayerData {
  id: number;
  title: string;
  shortTitle: string;
  description: string;
  categories: Category[];
  color: string;
}

export interface HFConfig {
  architectures?: string[];
  hidden_act?: string;
  hidden_size?: number;
  intermediate_size?: number;
  max_position_embeddings?: number;
  num_attention_heads?: number;
  num_hidden_layers?: number;
  num_key_value_heads?: number;
  rms_norm_eps?: number;
  vocab_size?: number;
  moe_experts?: number;
  moe_shared_experts?: number;
  [key: string]: any;
}

