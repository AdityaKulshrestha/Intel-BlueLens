# Intel BlueLens

LLM architecture and hardware execution visualizer for Intel platforms.

Visualizes the gap between abstract neural network architectures and their actual hardware execution paths — mapping model components to Intel CPU (IPEX/oneDNN/AMX), XPU (SYCL/XeTile), and GPU (CUTLASS/FlashAttn) kernels.

## Quick Start

**Prerequisites:** Node.js 18+

```bash
npm install
npm run dev
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for full deployment instructions.
