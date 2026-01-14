# JetBlock v3.0 - Nemotron 3 Hybrid Architecture Support

## Executive Summary

JetBlock v2.0 addressed ThinkingMachines batch-invariant determinism for standard transformer attention. **Nemotron 3 changes everything** - it's a hybrid Mamba-Transformer MoE architecture where attention is only 12% of layers.

**Critical insight**: Our current JetBlock optimizer only optimizes 6 out of 52 layers in Nemotron 3 Nano.

---

## Research Findings

### Nemotron 3 Architecture (NVIDIA, Dec 2025)

| Component | Count | Purpose |
|-----------|-------|---------|
| Mamba-2 layers | 23 | Long-range dependencies, O(N) complexity |
| MoE layers | 23 | Expert routing, 128 experts + 1 shared |
| Attention layers | 6 | GQA with 2 groups, structural reasoning |
| **Total** | **52** | Hybrid architecture |

**Layer Pattern**:
```
[Mamba-2 + MoE] x5 → Attention →
[Mamba-2 + MoE] x3 → Attention →
[Mamba-2 + MoE] x4 → Mamba-2 (final)
```

**Model Variants**:
- **Nano 30B-A3B**: 31.6B total, 3.2B active per token
- **Super 49B**: 100B total, 10B active (coming H1 2026)
- **Ultra 253B**: 500B total, 50B active (coming H1 2026)

### ThinkingMachines Determinism Requirements

Source: [Defeating Non-Determinism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)

**Root cause**: Batch-size variance in GPU kernels, NOT floating-point non-associativity.

**Three operations requiring batch-invariant implementations**:

1. **RMSNorm**
   - Split-reduction strategies break invariance
   - Solution: Accept modest slowdown for fixed parallelization

2. **MatMul**
   - Split-K operations break invariance
   - Solution: Consistent tile sizes (~20% performance loss)

3. **Attention**
   - Dual reductions (feature + sequence)
   - KV cache boundary conditions cause variance
   - Solution: Pre-update KV cache, fixed split-SIZE (not count)

**Results**:
- Without fix: 80 unique outputs from 1000 identical requests
- With fix: 1000/1000 identical
- Performance: ~1.6x slowdown (acceptable for determinism)

### Mamba-2 Determinism Issues

Source: [HuggingFace Transformers Mamba2](https://huggingface.co/docs/transformers/en/model_doc/mamba2)

**Critical finding**: "Slight discrepancy between batched and cached generation due to reimplemented Mamba 2 kernels."

**Two forward paths**:
1. `cuda_kernels_forward`: Fast but non-deterministic
2. `torch_forward`: 3-4x slower but consistent

**Root cause**: SSM tensor contractions have matmul equivalents but operation order differs.

**Sensitivity**: SSMs are sensitive to recurrent dynamics - FP32 may be required for stability.

---

## Gap Analysis: JetBlock v2.0 vs Requirements

| Requirement | v2.0 Status | v3.0 Needed |
|-------------|-------------|-------------|
| Batch-invariant attention | Partial (batch_size=1) | Fixed split-size, KV pre-update |
| Batch-invariant RMSNorm | Not implemented | Custom kernel |
| Batch-invariant MatMul | Not implemented | No Split-K wrapper |
| Mamba-2 support | Not implemented | Deterministic forward path |
| MoE routing determinism | Not implemented | Fixed expert selection |
| GQA support | Not implemented | 2-group attention |
| NVFP4 quantization | Not implemented | Future (Super/Ultra) |
| Hybrid layer detection | Not implemented | Auto-detect layer types |

---

## JetBlock v3.0 Architecture

### Core Design Principles

1. **Hybrid-Aware**: Auto-detect Mamba-2/Transformer/MoE layers
2. **Dual-Mode**: Speed vs Deterministic toggle (inherited from v2.0)
3. **Layer-Specific**: Apply appropriate optimization per layer type
4. **Nemotron-Optimized**: BF16 attention, fixed precision for SSM

### New Components

#### 1. BatchInvariantRMSNorm

```python
class BatchInvariantRMSNorm(nn.Module):
    """
    RMSNorm with fixed parallelization strategy.
    Eliminates batch-variance from reduction order.
    """
    def forward(self, x):
        # Force single-threaded reduction for invariance
        # Accept ~15% slowdown
```

#### 2. BatchInvariantMatMul

```python
class BatchInvariantMatMul:
    """
    MatMul wrapper that avoids Split-K operations.
    Uses consistent tile sizes regardless of batch size.
    ~20% slower but batch-invariant.
    """
    @staticmethod
    def matmul(a, b):
        # Disable cuBLAS auto-tuning
        # Force fixed tile strategy
```

#### 3. DeterministicMamba2Mixer

```python
class DeterministicMamba2Mixer(nn.Module):
    """
    Mamba-2 layer with guaranteed determinism.
    Forces torch_forward path, disables CUDA kernels.
    """
    def forward(self, x, cache=None):
        # CRITICAL: Use torch_forward, not cuda_kernels_forward
        # Fixed tensor contraction order
        # FP32 for SSM state calculations
```

#### 4. DeterministicMoERouter

```python
class DeterministicMoERouter(nn.Module):
    """
    MoE expert routing with deterministic selection.
    Fixed top-k ordering regardless of GPU state.
    """
    def route(self, x):
        # Deterministic argsort
        # Fixed expert activation order
```

#### 5. FixedSplitAttention

```python
class FixedSplitAttention(nn.Module):
    """
    Attention with fixed split-SIZE (not split count).
    Pre-updates KV cache before kernel execution.
    """
    def forward(self, q, k, v, kv_cache=None):
        # 1. Update KV cache FIRST (ThinkingMachines fix)
        # 2. Fixed split size = 64 tokens
        # 3. Consistent regardless of query count
```

#### 6. HybridLayerDetector

```python
class HybridLayerDetector:
    """
    Auto-detect layer types in hybrid architectures.
    Returns optimization strategy per layer.
    """
    def analyze(self, model):
        # Detect: Mamba2Mixer, MoE, Attention
        # Return layer_map with optimization config
```

### New ComfyUI Nodes

1. **JetBlockNemotronOptimizer**: Full Nemotron 3 optimization
2. **JetBlockMamba2Deterministic**: Mamba-2 specific determinism
3. **JetBlockHybridProfiler**: Analyze hybrid model architecture
4. **JetBlockV3ModeSwitch**: Speed/Deterministic toggle with hybrid support

### Configuration Updates

```python
@dataclass
class JetBlockV3Config:
    # Inherited from v2.0
    deterministic_mode: bool = False
    batch_size: int = 1  # Forced in deterministic mode
    cudnn_benchmark: bool = False  # Disabled in deterministic mode

    # NEW: Nemotron 3 specific
    enable_mamba2_determinism: bool = True
    enable_moe_determinism: bool = True
    enable_fixed_split_attention: bool = True

    # Precision control
    attention_precision: str = "bf16"  # Per Nemotron spec
    ssm_state_precision: str = "fp32"  # Required for stability

    # Performance tradeoffs
    accept_matmul_slowdown: bool = True  # ~20% for determinism
    accept_rmsnorm_slowdown: bool = True  # ~15% for determinism
    force_torch_forward: bool = True  # Disable CUDA kernels
```

---

## Implementation Plan

### Phase 1: Core Deterministic Operators (Week 1)
- [ ] BatchInvariantRMSNorm
- [ ] BatchInvariantMatMul
- [ ] FixedSplitAttention with KV pre-update
- [ ] Unit tests for each operator

### Phase 2: Mamba-2 Support (Week 2)
- [ ] DeterministicMamba2Mixer
- [ ] SSM state precision handling
- [ ] Integration with HuggingFace Mamba2
- [ ] Determinism validation tests

### Phase 3: MoE & Hybrid Support (Week 3)
- [ ] DeterministicMoERouter
- [ ] HybridLayerDetector
- [ ] Auto-optimization for mixed architectures
- [ ] End-to-end Nemotron 3 tests

### Phase 4: ComfyUI Integration (Week 4)
- [ ] New nodes for v3.0 features
- [ ] Migration path from v2.0
- [ ] Documentation and examples
- [ ] Performance benchmarks

---

## Performance Expectations

### Deterministic Mode (v3.0)

| Operation | Slowdown | Justification |
|-----------|----------|---------------|
| RMSNorm | ~15% | Fixed parallelization |
| MatMul | ~20% | No Split-K |
| Attention | ~10% | Fixed split-size |
| Mamba-2 | ~3x | torch_forward vs CUDA |
| **Overall** | ~1.8x | Acceptable for reproducibility |

### Speed Mode (unchanged)
- Full performance, no determinism guarantees
- Same as v2.0 behavior

---

## Validation Strategy

### Checksum Testing

```python
def validate_determinism(model, input, runs=100):
    checksums = set()
    for _ in range(runs):
        output = model(input)
        checksum = hashlib.sha256(output.cpu().numpy().tobytes()).hexdigest()[:16]
        checksums.add(checksum)

    assert len(checksums) == 1, f"Non-deterministic: {len(checksums)} unique outputs"
    return checksums.pop()
```

### Targeted Tests

1. **Batch-variance test**: Same input, different batch positions
2. **KV cache test**: Cached vs non-cached generation
3. **Mamba-2 test**: torch_forward consistency
4. **MoE test**: Expert routing stability
5. **End-to-end**: Full Nemotron 3 inference

---

## Migration from v2.0

### Breaking Changes
- None (v3.0 is additive)

### New Defaults
- Mamba-2 determinism: enabled by default
- MoE determinism: enabled by default
- Fixed split attention: enabled by default

### Deprecated
- Nothing deprecated

---

## References

1. [NVIDIA Nemotron 3 Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Technical-Report.pdf)
2. [ThinkingMachines: Defeating Non-Determinism](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
3. [Mamba-2: Algorithms and Systems](https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems)
4. [HuggingFace Mamba2 Documentation](https://huggingface.co/docs/transformers/en/model_doc/mamba2)
5. [State-Spaces Mamba GitHub](https://github.com/state-spaces/mamba)

---

## Approval Request

This plan addresses:
1. Nemotron 3 hybrid architecture (Mamba-2 + MoE + Attention)
2. ThinkingMachines batch-invariant determinism
3. Production-ready implementation for ComfyUI

**Estimated effort**: 4 phases
**Performance tradeoff**: ~1.8x slowdown for guaranteed reproducibility

Proceed with implementation?
