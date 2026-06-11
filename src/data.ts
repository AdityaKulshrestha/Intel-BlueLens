import { LayerData } from './types';

export const layersData: LayerData[] = [
  {
    id: 6,
    shortTitle: "Layer 6: Models",
    title: "Layer 6: Target Model Implementations",
    description: "At the pinnacle, these specific models synthesize the stack below them, mapping directly to highly optimized execution capabilities.",
    color: "from-intel-primary to-intel-bright",
    categories: [
      {
        name: "Supported Model Architectures",
        items: [
          {
            name: "Plamo 3",
            description: "Dense Causal Decoder utilizing standard MHA/GQA, SwiGLU, and interleaved sliding window patterns, mapped seamlessly to optimized execution paths."
          },
          {
            name: "DeepSeek-V3 / R1",
            description: "MoE paradigm, utilizing MLA for attention, Fine-Grained Experts with Shared Experts, mapped to specialized sparse routing SYCL kernels."
          },
          {
            name: "Kimi K2.6",
            description: "1 Trillion parameter MoE (32B active), utilizing MLA, SwiGLU, MoonViT for multimodal input, heavily leveraging memory bandwidth optimization for 256k long-context scaling."
          },
          {
            name: "Mistral / Mixtral",
            description: "Employs Sliding Window Attention (SWA), Rolling Buffer Cache, and a classic 8-expert MoE design, well supported by standard sparse compute paths."
          },
          {
            name: "Qwen3-Next",
            description: "80B total / 3B active MoE. Hybridizes Gated DeltaNet with Gated Attention (3:1), utilizes Partial RoPE, taking advantage of custom C++ linear attention algorithms."
          }
        ]
      }
    ]
  },
  {
    id: 5,
    shortTitle: "Layer 5: Architectural Paradigms",
    title: "Layer 5: Broad Architectural Paradigms",
    description: "The top-level orchestration dictates how the sub-modules are glued together.",
    color: "from-emerald-400 to-teal-500",
    categories: [
      {
        name: "Macro Architectures",
        items: [
          {
            name: "Dense Transformers",
            description: "(e.g., LLaMA 3). Rely on GQA, SwiGLU, and extreme data scaling to brute-force high performance."
          },
          {
            name: "Ultra-Sparse MoE Transformers",
            description: "(e.g., DeepSeek-V3). Scale up to 1 Trillion parameters but only activate ~30-37 Billion parameters per token, maximizing scale with minimal inference latency."
          },
          {
            name: "Hybrid Linear-Softmax Architectures",
            description: "To solve the quadratic bottleneck of global attention and the poor retrieval of linear attention, models like Kimi Linear and Qwen3-Next interleave linear layers (KDA / GDN) and global softmax layers (MLA / Gated Attention) in a 3:1 ratio."
          }
        ]
      }
    ]
  },
  {
    id: 4,
    shortTitle: "Layer 4: Feed-Forward",
    title: "Layer 4: Feed-Forward & Routing Layer",
    description: "Following token mixing, representations pass through a channel-mixing step.",
    color: "from-amber-400 to-orange-500",
    categories: [
      {
        name: "Channel Mixing Strategies",
        items: [
          {
            name: "Dense FFN",
            description: "The standard approach where every parameter processes every token."
          },
          {
            name: "Standard MoE (Mixture of Experts)",
            description: "Routes tokens to a Top-K subset of experts (e.g., Mixtral 8x7B routes to 2 out of 8 experts), decoupling parameter scale from compute cost."
          },
          {
            name: "Fine-Grained MoE with Shared Experts",
            description: "Introduced by DeepSeek. It divides experts into much smaller, specialized blocks (e.g., 384 experts) and designates a Shared Expert that routes to all tokens to process universal syntactic/common knowledge."
          }
        ]
      }
    ]
  },
  {
    id: 3,
    shortTitle: "Layer 3: Attention Mechanism",
    title: "Layer 3: Attention Mechanism & Token Mixing",
    description: "This is where tokens interact with each other. It is heavily fragmented based on the need for speed, long-context retrieval, or memory efficiency.",
    color: "from-rose-400 to-red-500",
    categories: [
      {
        name: "Standard / Global Attention",
        items: [
          {
            name: "MHA (Multi-Head Attention)",
            description: "The original quadratic O(L²) attention. High capacity but massive KV cache footprint."
          },
          {
            name: "GQA (Grouped-Query Attention)",
            description: "Shares a single Key/Value head across multiple Query heads, drastically reducing KV cache size and memory bandwidth. The default for LLaMA 3 and Qwen2."
          }
        ]
      },
      {
        name: "Latent / Compressed Attention",
        items: [
          {
            name: "MLA (Multi-Head Latent Attention)",
            description: "Introduced in DeepSeek-V2/V3. Compresses the Key and Value matrices into a single low-rank latent vector space before caching. It reconstructs the vectors on the fly via up-projection, reducing KV cache by >90% while retaining high performance."
          }
        ]
      },
      {
        name: "Chunked / Sparse Attention",
        items: [
          {
            name: "SWA (Sliding Window Attention)",
            description: "Uses a rolling buffer so tokens only attend to a localized recent window (e.g., 4096 tokens). Used in Mistral and Gemma."
          },
          {
            name: "DCA (Dual Chunk Attention)",
            description: "Divides sequences into chunks, computing Intra-chunk, Inter-chunk, and Successive-chunk attention. Effectively handles 100k+ token sequences."
          },
          {
            name: "MoBA (Mixture of Block Attention)",
            description: "Partitions the KV cache into blocks and uses an MoE-style gating network (router) to dynamically assign a query to the Top-K most relevant historical blocks, achieving sub-quadratic complexity."
          }
        ]
      },
      {
        name: "Linear / Recurrent Attention",
        items: [
          {
            name: "GDN (Gated DeltaNet)",
            description: "Views attention as an online gradient descent problem on a fast-weight memory state, adding a scalar 'forget gate' to decay old memories."
          },
          {
            name: "KDA (Kimi Delta Attention)",
            description: "Refines GDN by replacing the scalar gate with a channel-wise gate, allowing each feature dimension to maintain its own independent forgetting rate."
          }
        ]
      }
    ]
  },
  {
    id: 2,
    shortTitle: "Layer 2: Core Sub-Layers",
    title: "Layer 2: Core Sub-Layers",
    description: "This layer manages the basic mathematical transformations applied to token embeddings as they travel through the network.",
    color: "from-fuchsia-500 to-purple-600",
    categories: [
      {
        name: "Normalization",
        description: "Normalization placement heavily dictates model depth and training stability.",
        items: [
          {
            name: "Pre-LN (Pre-LayerNorm)",
            description: "The modern standard (used in LLaMA, GPT-3). It normalizes inputs before attention/FFN layers, stabilizing early training but leading to representation collapse at extreme depths."
          },
          {
            name: "Post-LN (Post-LayerNorm)",
            description: "Used in the original Transformer. It preserves representational capacity for deep networks but struggles with gradient vanishing/exploding."
          },
          {
            name: "Mix-LN",
            description: "A hybrid approach applying Post-LN to earlier layers (preventing gradient vanishing) and Pre-LN to deeper layers (ensuring uniform gradient norms)."
          },
          {
            name: "KEEL",
            description: "Replaces the standard residual path with a Highway-style connection to stabilize Post-LN, enabling models to scale to extreme depths (1000+ layers)."
          },
          {
            name: "Functions (RMSNorm & QK-Norm)",
            description: "RMSNorm is widely preferred over standard LayerNorm because it drops the mean-centering operation, saving memory and compute. QK-Norm is also increasingly applied to queries and keys to prevent logit drift and stabilize attention."
          }
        ]
      },
      {
        name: "Activation Functions",
        items: [
          {
            name: "SwiGLU",
            description: "A Gated Linear Unit utilizing the Swish (SiLU) function. It splits the input into two paths, multiplying them to capture complex, second-order polynomials. Used by LLaMA, DeepSeek, and Mistral."
          },
          {
            name: "GeGLU / GELU",
            description: "Gaussian Error Linear Units used in architectures like Gemma and BERT."
          }
        ]
      },
      {
        name: "Positional Embeddings",
        items: [
          {
            name: "RoPE (Rotary Position Embedding)",
            description: "Rotates Query and Key vectors in 2D pairs based on their sequence position. It captures relative positioning directly via the dot product."
          },
          {
            name: "Fractional/Partial RoPE",
            description: "Applies rotation to only a small fraction (e.g., 10%-25%) of the hidden dimensions, massively reducing the KV cache footprint for long contexts without hurting convergence."
          },
          {
            name: "YaRN & NTK-Aware",
            description: "Frequency-scaling techniques applied to RoPE to extrapolate the context window (e.g., from 4k to 128k+) without complete retraining."
          },
          {
            name: "NoPE (No Position Encoding)",
            description: "Relies purely on causal masking and recurrent/linear layers to implicitly track positions. Used in Kimi Linear to allow global attention layers to convert into pure Multi-Query Attention at inference."
          }
        ]
      }
    ]
  },
  {
    id: 1,
    shortTitle: "Layer 1: Low-Level Kernels",
    title: "Layer 1: Kernels & Primitives",
    description: "Highly optimized execution primitives mapped directly to specific model sub-layers using domain specific language and low-level implementations.",
    color: "from-blue-500 to-indigo-600",
    categories: [
      {
        name: "Low Level Implementations",
        items: [
          {
            name: "Shared Compute Primitives",
            description: "GEMM operations, FusedRMSNorm, FusedSiLU, FlashAttention."
          }
        ]
      },
      {
        name: "Domain Specific Language",
        items: [
          {
            name: "Hardware Independent",
            description: "Triton, Cutlass - used to author custom execution kernels for specialized layer types."
          }
        ]
      }
    ]
  },
  {
    id: 0,
    shortTitle: "Layer 0: Hardware",
    title: "Layer 0: Hardware Execution",
    description: "The base physical layer orchestrating the actual compute operations.",
    color: "from-intel-primary to-intel-bright",
    categories: [
      {
        name: "Hardware Groups",
        items: [
          {
            name: "Accelerators (BattlemageXPU, CRI)",
            description: "Executed via SYCL for massive data-parallel compute."
          },
          {
            name: "Processors (CPU)",
            description: "Executed via native C++/C, utilizing AVX/AMX vectors for zero-overhead, ultra-low latency inference."
          }
        ]
      }
    ]
  }
];
