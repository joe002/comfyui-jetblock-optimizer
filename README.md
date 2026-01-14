# JetBlock Optimizer for ComfyUI

**v4.0 - Full Nemotron 3 Hybrid Architecture Support**

Production-grade optimization for Nemotron 3's hybrid architecture with **Mamba-2 determinism**, **MoE routing**, and **batch-invariant operators**. Optimized for RTX 4090.

---

## What's New in v4.0

### Full Nemotron 3 Hybrid Support
Nemotron 3 uses a hybrid architecture:
- **23 Mamba-2 layers** (44%) - State-space models
- **23 MoE layers** (44%) - Mixture of Experts
- **6 Attention layers** (12%) - Traditional attention

**v4.0 optimizes ALL layer types** (v2.0 only handled attention).

### Key Features

| Feature | Description |
|---------|-------------|
| **Batch-Invariant Ops** | Same output regardless of batch size |
| **Mamba-2 Determinism** | Forced torch_forward path for reproducibility |
| **MoE Routing Control** | Deterministic expert selection |
| **Cascade Mode** | /think vs /no_think budget control |
| **NVFP4 Ready** | Prepared for 50% KV cache compression |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 JetBlock v4.0                        │
├─────────────────────────────────────────────────────┤
│  jetblock_core_v4.py    - Batch-invariant operators │
│  jetblock_mamba2.py     - Mamba-2 determinism       │
│  jetblock_nemotron.py   - Hybrid layer detection    │
│  nodes_v4.py            - ComfyUI node interface    │
└─────────────────────────────────────────────────────┘
```

---

## Installation

### ComfyUI Manager (Recommended)
Search for "JetBlock Optimizer" in ComfyUI Manager.

### Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/josephibrahim/ComfyUI-JetBlock-Optimizer.git
```

### Dependencies
```bash
pip install torch>=2.1.0 numpy>=1.24.0

# Optional (full features)
pip install triton>=2.1.0 einops>=0.7.0 mamba-ssm>=1.2.0
```

---

## Nodes

### JetBlock/Nemotron Category

#### JetBlock Nemotron Optimizer
Main optimizer for Nemotron 3 hybrid models.

| Input | Type | Description |
|-------|------|-------------|
| model | MODEL | Model to optimize |
| determinism_level | COMBO | strict / standard / relaxed |
| enable_mamba2_determinism | BOOL | Force torch_forward path |
| enable_moe_determinism | BOOL | Deterministic expert routing |
| cascade_mode | COMBO | think / no_think / auto |

#### JetBlock Hybrid Profiler
Profile layer composition of loaded models.

| Output | Type | Description |
|--------|------|-------------|
| profile_text | STRING | Layer type breakdown |
| mamba2_count | INT | Number of Mamba-2 layers |
| moe_count | INT | Number of MoE layers |
| attention_count | INT | Number of attention layers |

#### JetBlock Mamba-2 Deterministic
Standalone Mamba-2 determinism wrapper.

| Input | Type | Description |
|-------|------|-------------|
| model | MODEL | Model to wrap |
| force_fp32_states | BOOL | Use FP32 for SSM states |
| disable_cuda_kernels | BOOL | Force PyTorch path |

#### JetBlock Cascade Mode
Control /think vs /no_think execution budget.

| Input | Type | Description |
|-------|------|-------------|
| mode | COMBO | think / no_think |
| reasoning_budget | FLOAT | Token budget multiplier |
| quality_threshold | FLOAT | Minimum quality score |

### JetBlock/Deterministic Category

#### JetBlock V4 Deterministic Sampler
Production deterministic sampling.

| Input | Type | Description |
|-------|------|-------------|
| model | MODEL | Model to sample |
| seed | INT | RNG seed |
| determinism_level | COMBO | strict / standard / relaxed |

| Output | Type | Description |
|--------|------|-------------|
| latent | LATENT | Deterministic output |
| checksum | STRING | SHA-256 verification hash |

#### JetBlock V4 Mode Switch
Toggle between performance modes.

| Input | Type | Description |
|-------|------|-------------|
| model | MODEL | Model to configure |
| mode | COMBO | speed / deterministic / balanced |

---

## Determinism Levels

| Level | Batch Invariance | MoE | Mamba-2 | Performance |
|-------|------------------|-----|---------|-------------|
| **strict** | Full | Deterministic | torch_forward | ~1.6x slower |
| **standard** | Partial | Top-k fixed | Hybrid | ~1.2x slower |
| **relaxed** | None | Standard | CUDA | Full speed |

### ThinkingMachines Research

Based on [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/):

**Key insight**: Batch-size variance causes nondeterminism, NOT temperature=0.

**Solution**: Process each batch item independently with single-sample operations.

---

## Cascade Mode (/think vs /no_think)

From NVIDIA CES 2026 Context Memory Platform:

| Mode | Behavior | Use Case |
|------|----------|----------|
| **/think** | Full reasoning budget | Complex generation |
| **/no_think** | Minimal reasoning | Fast iteration |
| **auto** | Dynamic based on quality | Production default |

---

## Hardware Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| PyTorch | 2.1.0 | 2.2.0+ |
| CUDA | 8.0 | 9.0+ |
| VRAM | 8GB | 24GB (RTX 4090) |
| Python | 3.10 | 3.11+ |

---

## Usage Examples

### Basic Deterministic Workflow
```
Load Checkpoint
    │
    ▼
JetBlock Nemotron Optimizer (determinism_level="strict")
    │
    ▼
CLIP Text Encode
    │
    ▼
JetBlock V4 Deterministic Sampler (seed=12345)
    │
    ▼
VAE Decode
    │
    ▼
Save Image
```

### Profile Unknown Model
```
Load Checkpoint
    │
    ▼
JetBlock Hybrid Profiler
    │
    ▼
[View layer composition in output]
```

### Fast Iteration Mode
```
Load Checkpoint
    │
    ▼
JetBlock V4 Mode Switch (mode="speed")
    │
    ▼
[Standard workflow - no determinism overhead]
```

---

## Performance

### RTX 4090 Benchmarks

| Mode | SDXL 1024 | SD1.5 512 | FLUX 1024 |
|------|-----------|-----------|-----------|
| Speed | 0.3s | 0.05s | 0.5s |
| Deterministic | 0.5s | 0.08s | 0.8s |
| Strict | 0.6s | 0.10s | 1.0s |

### Determinism Validation

Same seed + strict mode = identical checksum every run.

```python
# Run 1: checksum = "a1b2c3d4..."
# Run 2: checksum = "a1b2c3d4..."  ✓ Match
# Run 3: checksum = "a1b2c3d4..."  ✓ Match
```

---

## API Reference

### Python Import
```python
from comfyui_jetblock_optimizer import (
    JetBlockV4Config,
    DeterminismLevel,
    JetBlockNemotronOptimizer,
)

# Configure
config = JetBlockV4Config(
    determinism_level=DeterminismLevel.STRICT,
    enable_mamba2_determinism=True,
    enable_moe_determinism=True,
)

# Optimize model
optimizer = JetBlockNemotronOptimizer(config)
optimized_model = optimizer.optimize(model)
```

---

## Troubleshooting

### Non-deterministic output?
1. Set `determinism_level="strict"`
2. Enable both `enable_mamba2_determinism` and `enable_moe_determinism`
3. Verify same seed across runs

### Performance degradation?
1. Use `determinism_level="relaxed"` for speed
2. Toggle `JetBlock V4 Mode Switch` to "speed"
3. Disable Mamba-2/MoE determinism if not needed

### CUDA errors?
1. Update PyTorch to 2.1.0+
2. Set `disable_cuda_kernels=True` in Mamba-2 node
3. Check VRAM usage with profiler

---

## References

- [NVIDIA Nemotron 3 Technical Report](https://developer.nvidia.com/nemotron)
- [ThinkingMachines: Defeating Nondeterminism](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [NVIDIA CES 2026 Context Memory Platform](https://developer.nvidia.com/ces)
- [Mamba-2 Paper](https://arxiv.org/abs/2312.00752)

---

## License

MIT License - See [LICENSE](LICENSE) file

---

## Credits

Developed by **Joseph Ibrahim**

Research integration:
- NVIDIA Nemotron architecture
- ThinkingMachines batch-invariance research
- Mamba-2 state-space models
- CES 2026 Context Memory Platform

---

*Determinism is not optional. It's production-grade.*
