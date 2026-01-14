# JETBLOCK v4.0 — PRODUCTION PLAN

## ADHD-Optimized Document

**Reading time**: 5 min scan, 15 min deep read

**Sections**:
1. [TL;DR](#tldr) (30 seconds)
2. [The Big Picture](#big-picture) (2 min)
3. [What Already Exists](#what-exists) (1 min)
4. [What JetBlock v4 Adds](#what-v4-adds) (3 min)
5. [Implementation Plan](#implementation) (5 min)
6. [Ship Checklist](#ship-checklist) (2 min)

---

<a name="tldr"></a>
## TL;DR

```
┌─────────────────────────────────────────────────────────────────┐
│                         THE BOTTOM LINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FRAMEWORKS_PRODUCT has:  High-level determinism nodes          │
│  JetBlock v4 adds:        LOW-LEVEL GPU KERNELS + NEMOTRON      │
│                                                                 │
│  NO DUPLICATION — they're complementary layers                  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  FRAMEWORKS_PRODUCT    │    JetBlock v4                 │   │
│  │  ────────────────────  │    ──────────────              │   │
│  │  DeterministicSampler  │    LinearAttentionKernel       │   │
│  │  ChecksumValidator     │    Mamba2DeterministicMixer    │   │
│  │  MoERouterNode         │    BatchInvariantMatMul        │   │
│  │  ECHOContextNode       │    FixedSplitAttention         │   │
│  │  CascadeRefiner        │    NVFP4Quantizer              │   │
│  │                        │    NemotronHybridDetector      │   │
│  │  (orchestration)       │    (kernels)                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

<a name="big-picture"></a>
## THE BIG PICTURE

### CES 2026 Key Announcements (VFX Relevant)

| Announcement | What It Means | JetBlock Connection |
|--------------|---------------|---------------------|
| **NVFP4 KV Cache** | 50% memory = 2x context | Quantization support |
| **Context Memory Platform** | hot/warm/cold/archive tiers | ECHO integration |
| **Nemotron-Cascade** | Sequential domain RL | `/think` mode toggle |
| **Multi-model "trivial"** | Agent orchestration validated | MoE routing |
| **Object Permanence** | Temporal coherence | TemporalSkipper |

### Nemotron 3 Architecture (CRITICAL)

```
                    NEMOTRON 3 NANO — 52 LAYERS
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │   [MAMBA-2] ─► [MoE] ─► [MAMBA-2] ─► [MoE] ─►      │
    │       ↓                                             │
    │   [MAMBA-2] ─► [MoE] ─► [MAMBA-2] ─► [MoE] ─►      │
    │       ↓                                             │
    │   [MAMBA-2] ─► [MoE] ─► [ATTENTION] ────────►      │  ← 6 total
    │       ↓                                             │
    │   (repeat pattern...)                               │
    │                                                     │
    └─────────────────────────────────────────────────────┘

    LAYER BREAKDOWN:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Mamba-2:     23 layers (44%)  ← State-space, O(N)
    MoE:         23 layers (44%)  ← 128 experts
    Attention:    6 layers (12%)  ← GQA, 2 groups
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    JetBlock v2.0 only optimizes 6/52 layers (12%)
    JetBlock v4.0 optimizes ALL 52 layers
```

### ThinkingMachines Determinism (The Real Fix)

```
    ┌─────────────────────────────────────────────────────┐
    │                  COMMON MISCONCEPTION               │
    │                                                     │
    │   "temperature=0 makes it deterministic"            │
    │                                                     │
    │                       WRONG                         │
    │                                                     │
    └─────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │                    THE TRUTH                        │
    │                                                     │
    │   BATCH-SIZE VARIANCE is the culprit               │
    │                                                     │
    │   Same prompt, same seed, different batch sizes:    │
    │     Batch=1:  "The answer is 42"                   │
    │     Batch=4:  "The answer is 41"  ← DIFFERENT      │
    │     Batch=8:  "The answer is 43"  ← DIFFERENT      │
    │                                                     │
    │   FIX: batch_size=1 + batch-invariant kernels      │
    │                                                     │
    └─────────────────────────────────────────────────────┘
```

---

<a name="what-exists"></a>
## WHAT ALREADY EXISTS

### In FRAMEWORKS_PRODUCT (Don't Duplicate)

| File | What It Does | Status |
|------|--------------|--------|
| `comfyui_deterministic_nodes.py` | DeterministicSampler, ChecksumValidator, MoERouter | DONE |
| `comfyui_framework_nodes.py` | ECHO, PRISM, CSQMF integration nodes | DONE |
| `framework_orchestrator.py` | 7-agent async orchestration | DONE |
| `async_conductor.py` | Ralph v3 file-centric conductor | DONE |

### In JetBlock v2.0 (Current)

| Component | What It Does | Status |
|-----------|--------------|--------|
| `LinearAttentionKernel` | O(N) attention via ELU feature map | DONE |
| `DynamicConvolutionKernel` | Content-adaptive convolution | DONE |
| `JetBlockAttention` | Dual-path + gating | DONE |
| `TemporalCoherenceSkipper` | Timestep interpolation | DONE |
| `BatchInvariantConfig` | Basic determinism settings | PARTIAL |

---

<a name="what-v4-adds"></a>
## WHAT JETBLOCK v4 ADDS

### New Components (No Overlap with FRAMEWORKS_PRODUCT)

```
┌─────────────────────────────────────────────────────────────────┐
│                      JETBLOCK v4.0 SCOPE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. MAMBA-2 DETERMINISTIC MIXER                                │
│     ─────────────────────────                                   │
│     • Force torch_forward (not CUDA kernels)                   │
│     • Fixed tensor contraction order                           │
│     • FP32 for SSM state stability                             │
│                                                                 │
│  2. BATCH-INVARIANT OPERATORS                                  │
│     ─────────────────────────                                   │
│     • BatchInvariantRMSNorm (fixed parallelization)            │
│     • BatchInvariantMatMul (no Split-K)                        │
│     • FixedSplitAttention (KV pre-update)                      │
│                                                                 │
│  3. MOE ROUTING DETERMINISM                                    │
│     ────────────────────────                                    │
│     • Fixed expert selection order                             │
│     • Deterministic top-k                                      │
│     • Hash-based tie-breaking                                  │
│                                                                 │
│  4. NVFP4 QUANTIZATION                                         │
│     ─────────────────────                                       │
│     • KV cache compression (50% memory)                        │
│     • Attention in BF16 (Nemotron spec)                        │
│     • SSM states in FP32 (stability)                           │
│                                                                 │
│  5. NEMOTRON HYBRID DETECTOR                                   │
│     ───────────────────────                                     │
│     • Auto-detect Mamba-2/MoE/Attention layers                 │
│     • Apply per-layer optimization strategy                    │
│     • Report layer distribution                                │
│                                                                 │
│  6. NEMOTRON-CASCADE MODE                                      │
│     ─────────────────────                                       │
│     • /think vs /no_think toggle                               │
│     • Configurable thinking budget                             │
│     • Nemotron 3 specific optimizations                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Architecture Diagram

```
                         USER WORKFLOW
                              │
                              ▼
    ┌─────────────────────────────────────────────────────┐
    │              FRAMEWORKS_PRODUCT LAYER               │
    │  (DeterministicSampler, ECHO, MoE, Orchestration)  │
    └─────────────────────────┬───────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────┐
    │                  JETBLOCK v4 LAYER                  │
    │  (Kernels, Mamba-2, NVFP4, Hybrid Detection)       │
    └─────────────────────────┬───────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────┐
    │                    PyTorch / CUDA                   │
    └─────────────────────────────────────────────────────┘
```

### New ComfyUI Nodes (v4)

| Node | Category | Purpose |
|------|----------|---------|
| `JetBlockNemotronOptimizer` | jetblock/nemotron | Full Nemotron 3 optimization |
| `JetBlockMamba2Deterministic` | jetblock/mamba | Mamba-2 layer determinism |
| `JetBlockHybridProfiler` | jetblock/analysis | Layer type detection |
| `JetBlockNVFP4Quantizer` | jetblock/quantization | NVFP4 KV cache |
| `JetBlockCascadeMode` | jetblock/nemotron | /think mode toggle |

---

<a name="implementation"></a>
## IMPLEMENTATION PLAN

### Phase 1: Core Kernels (Ship First)

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 1: WEEK 1-2                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [ ] BatchInvariantRMSNorm                                      │
│      └── Fixed reduction order, ~15% slower                    │
│                                                                 │
│  [ ] BatchInvariantMatMul                                       │
│      └── No Split-K ops, ~20% slower                           │
│                                                                 │
│  [ ] FixedSplitAttention                                        │
│      └── KV pre-update, fixed split-size=64                    │
│                                                                 │
│  [ ] Unit tests for each operator                               │
│      └── Verify determinism across batch sizes                 │
│                                                                 │
│  DELIVERABLE: jetblock_core_v4.py with batch-invariant ops     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 2: Mamba-2 Support

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 2: WEEK 2-3                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [ ] DeterministicMamba2Mixer                                   │
│      └── Force torch_forward path                              │
│      └── FP32 SSM states                                       │
│      └── Fixed tensor contraction order                        │
│                                                                 │
│  [ ] Mamba2LayerWrapper                                         │
│      └── Drop-in replacement for HF Mamba2Mixer                │
│      └── Preserves weights, changes forward()                  │
│                                                                 │
│  [ ] Integration tests                                          │
│      └── Test with nvidia/Nemotron-3-Nano                      │
│      └── Verify 1000/1000 identical outputs                    │
│                                                                 │
│  DELIVERABLE: jetblock_mamba2.py                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 3: Nemotron Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 3: WEEK 3-4                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [ ] NemotronHybridDetector                                     │
│      └── Auto-detect layer types (Mamba/MoE/Attn)              │
│      └── Return optimization strategy per layer                │
│                                                                 │
│  [ ] DeterministicMoERouter                                     │
│      └── Fixed expert selection order                          │
│      └── Hash-based tie-breaking                               │
│                                                                 │
│  [ ] NVFP4Quantizer                                             │
│      └── KV cache 50% compression                              │
│      └── BF16 attention, FP32 SSM                              │
│                                                                 │
│  [ ] CascadeModeController                                      │
│      └── /think vs /no_think toggle                            │
│      └── Budget-aware inference                                │
│                                                                 │
│  DELIVERABLE: jetblock_nemotron.py                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 4: Production Polish

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 4: WEEK 4                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [ ] ComfyUI node registration                                  │
│      └── All 5 new nodes with proper INPUT_TYPES               │
│                                                                 │
│  [ ] pyproject.toml                                             │
│      └── Proper packaging for pip install                      │
│                                                                 │
│  [ ] README.md                                                  │
│      └── Installation, usage, examples                         │
│                                                                 │
│  [ ] Example workflows                                          │
│      └── workflows/nemotron_deterministic.json                 │
│      └── workflows/mamba2_benchmark.json                       │
│                                                                 │
│  [ ] Benchmark suite                                            │
│      └── Speed vs determinism tradeoffs                        │
│      └── Memory usage comparison                               │
│                                                                 │
│  DELIVERABLE: Shippable ComfyUI custom node package            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

<a name="ship-checklist"></a>
## SHIP CHECKLIST

### Production Requirements

```
┌─────────────────────────────────────────────────────────────────┐
│                      SHIP CHECKLIST                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STRUCTURE                                                      │
│  ─────────                                                      │
│  [ ] __init__.py with NODE_CLASS_MAPPINGS                      │
│  [ ] nodes.py with all node classes                            │
│  [ ] jetblock_core_v4.py (batch-invariant ops)                 │
│  [ ] jetblock_mamba2.py (Mamba-2 determinism)                  │
│  [ ] jetblock_nemotron.py (Nemotron integration)               │
│  [ ] requirements.txt                                          │
│  [ ] pyproject.toml                                            │
│                                                                 │
│  DOCUMENTATION                                                  │
│  ─────────────                                                  │
│  [ ] README.md (installation, quick start)                     │
│  [ ] CHANGELOG.md (v2→v4 changes)                              │
│  [ ] examples/ folder with workflow JSONs                      │
│                                                                 │
│  TESTING                                                        │
│  ───────                                                        │
│  [ ] tests/test_determinism.py                                 │
│  [ ] tests/test_mamba2.py                                      │
│  [ ] tests/test_nemotron.py                                    │
│  [ ] CI via GitHub Actions                                     │
│                                                                 │
│  VALIDATION                                                     │
│  ──────────                                                     │
│  [ ] 1000/1000 identical outputs test                          │
│  [ ] Memory usage benchmarks                                   │
│  [ ] Speed vs determinism comparison                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### File Structure

```
ComfyUI-JetBlock-Optimizer/
├── __init__.py              # Node registration
├── nodes.py                 # ComfyUI node classes (v4)
├── jetblock_core.py         # v2 core (keep for compatibility)
├── jetblock_core_v4.py      # v4 batch-invariant ops
├── jetblock_mamba2.py       # Mamba-2 determinism
├── jetblock_nemotron.py     # Nemotron integration
├── nodes_compatibility.py   # v2 backward compat
├── nodes_cosmos.py          # Cosmos features
├── compatibility_checker.py # Version checking
├── pyproject.toml          # Package config
├── requirements.txt        # Dependencies
├── README.md               # User docs
├── CHANGELOG.md            # Version history
├── JETBLOCK_V4_PRODUCTION_PLAN.md  # This file
├── tests/
│   ├── test_determinism.py
│   ├── test_mamba2.py
│   └── test_nemotron.py
└── examples/
    ├── nemotron_deterministic.json
    ├── mamba2_benchmark.json
    └── speed_vs_determinism.json
```

---

## DECISION MATRIX

### What Goes Where

| Feature | JetBlock v4 | FRAMEWORKS_PRODUCT | Both |
|---------|-------------|-------------------|------|
| GPU kernel optimization | X | | |
| Mamba-2 determinism | X | | |
| NVFP4 quantization | X | | |
| Batch-invariant operators | X | | |
| Nemotron hybrid detection | X | | |
| High-level determinism nodes | | X | |
| ECHO context management | | X | |
| MoE routing logic | | X | |
| 7-agent orchestration | | X | |
| Checksum validation | | | X |

### Integration Points

```
FRAMEWORKS_PRODUCT                    JETBLOCK v4
────────────────────                  ───────────────────
DeterministicSampler ─────────────────► JetBlockNemotronOptimizer
        │                                       │
        │                                       │
ECHOContextNode ──────────────────────► NVFP4Quantizer
        │                              (context compression)
        │                                       │
MoERouterNode ────────────────────────► DeterministicMoERouter
        │                              (low-level routing)
        │                                       │
ChecksumValidator ◄───────────────────► JetBlockDeterministicSampler
                                       (output verification)
```

---

## PERFORMANCE EXPECTATIONS

### Deterministic Mode Overhead

| Operation | Speed Impact | Memory Impact | Justification |
|-----------|-------------|---------------|---------------|
| RMSNorm | -15% | Same | Fixed parallelization |
| MatMul | -20% | Same | No Split-K |
| Attention | -10% | Same | Fixed split-size |
| Mamba-2 | -200% (3x slower) | Same | torch_forward |
| **Overall** | **-40% to -60%** | **Same** | Determinism guarantee |

### Speed Mode (Default)

- Full performance, no determinism guarantee
- Same as v2.0 behavior
- Use for iteration, switch to deterministic for final renders

---

## NEXT STEPS

**Immediate action**: Approve this plan and begin Phase 1

**Questions to resolve**:
1. Target PyTorch version? (2.0+, 2.1+, 2.2+?)
2. CUDA compute capability minimum? (7.5+, 8.0+?)
3. HuggingFace transformers version? (4.40+?)

---

*Document generated by deep analysis of:*
- *NVIDIA CES 2026 announcements*
- *ThinkingMachines batch-invariance research*
- *Nemotron 3 architecture technical reports*
- *FRAMEWORKS_PRODUCT ecosystem (65 files)*
- *JetBlock v2.0 current implementation*

*Formatted for ADHD readability: scan → understand → act*
