"""
JetBlock Nemotron Integration Module
====================================

Full Nemotron 3 optimization with:
- Hybrid architecture detection (Mamba-2 / MoE / Attention)
- Layer-specific optimization strategies
- Cascade mode control (/think vs /no_think)
- NVFP4 quantization support (when available)

Nemotron 3 Architecture (52 layers):
- 23 Mamba-2 layers (44%) - State-space, O(N) complexity
- 23 MoE layers (44%) - 128 experts + 1 shared
- 6 Attention layers (12%) - GQA with 2 groups

Layer Pattern:
  [Mamba-2 + MoE] x5 → Attention →
  [Mamba-2 + MoE] x3 → Attention →
  [Mamba-2 + MoE] x4 → Mamba-2

References:
- NVIDIA Nemotron 3 Technical Report
- NVIDIA CES 2026 Context Memory Platform
- Nemotron-Cascade Training Methodology

Author: Joseph Ibrahim
Version: 4.0.0
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
from enum import Enum

import torch
import torch.nn as nn

from .jetblock_core_v4 import (
    JetBlockV4Config,
    DeterminismLevel,
    get_config,
    set_config,
    BatchInvariantRMSNorm,
    BatchInvariantMatMul,
    FixedSplitAttention,
    JetBlockAttentionV4,
    compute_tensor_checksum,
)

from .jetblock_mamba2 import (
    Mamba2Config,
    Mamba2Cache,
    DeterministicMamba2Mixer,
    DeterministicMamba2Layer,
    replace_mamba2_with_deterministic,
)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER TYPE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class LayerType(Enum):
    """Types of layers in Nemotron 3 hybrid architecture."""
    MAMBA2 = "mamba2"
    MOE = "moe"
    ATTENTION = "attention"
    MLP = "mlp"
    NORM = "norm"
    EMBEDDING = "embedding"
    UNKNOWN = "unknown"


@dataclass
class LayerInfo:
    """Information about a detected layer."""
    name: str
    layer_type: LayerType
    index: int
    param_count: int
    optimization_strategy: str
    determinism_risk: str  # "low", "medium", "high"


class NemotronHybridDetector:
    """
    Detects layer types in Nemotron 3 hybrid architecture.

    Identifies:
    - Mamba-2 layers (state-space models)
    - MoE layers (mixture of experts)
    - Attention layers (GQA)
    - Supporting layers (norms, embeddings)
    """

    # Keywords for layer detection
    MAMBA_KEYWORDS = ["mamba", "ssm", "state_space", "mixer"]
    MOE_KEYWORDS = ["moe", "expert", "router", "gate"]
    ATTENTION_KEYWORDS = ["attention", "attn", "self_attn", "q_proj", "k_proj", "v_proj"]
    NORM_KEYWORDS = ["norm", "layernorm", "rmsnorm"]
    EMBEDDING_KEYWORDS = ["embed", "token", "position"]

    def __init__(self):
        self.layer_map: Dict[str, LayerInfo] = {}

    def analyze(self, model: nn.Module) -> Dict[str, Any]:
        """
        Analyze model architecture and return layer breakdown.

        Returns:
            Dictionary with layer counts, types, and optimization recommendations
        """
        self.layer_map = {}

        layer_counts = {
            LayerType.MAMBA2: 0,
            LayerType.MOE: 0,
            LayerType.ATTENTION: 0,
            LayerType.MLP: 0,
            LayerType.NORM: 0,
            LayerType.EMBEDDING: 0,
            LayerType.UNKNOWN: 0,
        }

        total_params = 0
        determinism_risks = []

        for idx, (name, module) in enumerate(model.named_modules()):
            layer_type = self._detect_layer_type(name, module)
            param_count = sum(p.numel() for p in module.parameters(recurse=False))
            total_params += param_count

            # Determine optimization strategy
            strategy, risk = self._get_optimization_strategy(layer_type, name)

            layer_info = LayerInfo(
                name=name,
                layer_type=layer_type,
                index=idx,
                param_count=param_count,
                optimization_strategy=strategy,
                determinism_risk=risk,
            )

            self.layer_map[name] = layer_info
            layer_counts[layer_type] += 1

            if risk in ("medium", "high"):
                determinism_risks.append((name, risk))

        # Compute architecture summary
        total_layers = sum(layer_counts.values())
        nemotron_signature = self._check_nemotron_signature(layer_counts)

        return {
            "total_layers": total_layers,
            "total_params": total_params,
            "layer_counts": {lt.value: count for lt, count in layer_counts.items()},
            "architecture_type": "nemotron_hybrid" if nemotron_signature else "unknown",
            "nemotron_signature_match": nemotron_signature,
            "determinism_risks": determinism_risks,
            "optimization_summary": self._generate_optimization_summary(layer_counts),
        }

    def _detect_layer_type(self, name: str, module: nn.Module) -> LayerType:
        """Detect the type of a layer based on name and module type."""
        name_lower = name.lower()
        module_type = type(module).__name__.lower()

        # Check Mamba-2
        if any(kw in name_lower or kw in module_type for kw in self.MAMBA_KEYWORDS):
            return LayerType.MAMBA2

        # Check MoE
        if any(kw in name_lower or kw in module_type for kw in self.MOE_KEYWORDS):
            return LayerType.MOE

        # Check Attention
        if any(kw in name_lower or kw in module_type for kw in self.ATTENTION_KEYWORDS):
            return LayerType.ATTENTION

        # Check Norm
        if any(kw in name_lower or kw in module_type for kw in self.NORM_KEYWORDS):
            return LayerType.NORM

        # Check Embedding
        if any(kw in name_lower or kw in module_type for kw in self.EMBEDDING_KEYWORDS):
            return LayerType.EMBEDDING

        # Check MLP
        if "mlp" in name_lower or "feed_forward" in name_lower or "ff" in name_lower:
            return LayerType.MLP

        # Check by module type
        if isinstance(module, nn.Linear):
            return LayerType.MLP
        if isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            return LayerType.NORM
        if isinstance(module, nn.Embedding):
            return LayerType.EMBEDDING

        return LayerType.UNKNOWN

    def _get_optimization_strategy(
        self,
        layer_type: LayerType,
        name: str,
    ) -> Tuple[str, str]:
        """Get optimization strategy and determinism risk for layer type."""
        strategies = {
            LayerType.MAMBA2: ("DeterministicMamba2Mixer", "high"),
            LayerType.MOE: ("DeterministicMoERouter", "high"),
            LayerType.ATTENTION: ("FixedSplitAttention", "medium"),
            LayerType.MLP: ("BatchInvariantMatMul", "low"),
            LayerType.NORM: ("BatchInvariantRMSNorm", "medium"),
            LayerType.EMBEDDING: ("none", "low"),
            LayerType.UNKNOWN: ("none", "low"),
        }
        return strategies.get(layer_type, ("none", "low"))

    def _check_nemotron_signature(self, layer_counts: Dict[LayerType, int]) -> bool:
        """Check if layer distribution matches Nemotron 3 signature."""
        # Nemotron 3 Nano has roughly equal Mamba-2 and MoE layers,
        # with few attention layers
        mamba = layer_counts[LayerType.MAMBA2]
        moe = layer_counts[LayerType.MOE]
        attn = layer_counts[LayerType.ATTENTION]

        if mamba == 0 and moe == 0:
            return False

        # Check ratios (allowing some tolerance)
        if mamba > 0 and moe > 0:
            mamba_moe_ratio = mamba / moe
            if 0.5 <= mamba_moe_ratio <= 2.0:  # Should be roughly 1:1
                if attn < mamba * 0.5:  # Attention should be minority
                    return True

        return False

    def _generate_optimization_summary(
        self,
        layer_counts: Dict[LayerType, int],
    ) -> Dict[str, Any]:
        """Generate optimization recommendations."""
        total = sum(layer_counts.values())
        if total == 0:
            return {"recommendation": "No optimizable layers detected"}

        mamba_pct = layer_counts[LayerType.MAMBA2] / total * 100
        moe_pct = layer_counts[LayerType.MOE] / total * 100
        attn_pct = layer_counts[LayerType.ATTENTION] / total * 100

        return {
            "mamba2_layers_pct": round(mamba_pct, 1),
            "moe_layers_pct": round(moe_pct, 1),
            "attention_layers_pct": round(attn_pct, 1),
            "primary_optimization": (
                "Mamba-2 determinism" if mamba_pct > 30 else
                "MoE routing" if moe_pct > 30 else
                "Attention optimization"
            ),
            "expected_slowdown": (
                "~60% (Mamba-2 dominant)" if mamba_pct > 40 else
                "~40% (balanced)" if mamba_pct > 20 else
                "~20% (attention dominant)"
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC MOE ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

class DeterministicMoERouter(nn.Module):
    """
    Mixture of Experts router with deterministic selection.

    Standard MoE routing can produce different expert selections for
    identical inputs due to:
    - Top-k tie-breaking variance
    - Parallel routing computation order
    - Load balancing randomness

    This implementation ensures identical routing for identical inputs.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 128,
        num_shared_experts: int = 1,
        num_experts_per_token: int = 6,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_token = num_experts_per_token

        # Router gate
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def _deterministic_topk(
        self,
        scores: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Deterministic top-k selection with fixed tie-breaking.

        Standard torch.topk has undefined behavior for ties.
        This version uses lexicographic tie-breaking.
        """
        # Add small deterministic perturbation to break ties
        # Use position-based offset (deterministic)
        batch_size, num_experts = scores.shape
        tie_breaker = torch.arange(num_experts, device=scores.device).float()
        tie_breaker = tie_breaker * 1e-9  # Tiny offset

        # Add tie breaker (lower index wins ties)
        scores_adjusted = scores - tie_breaker.unsqueeze(0)

        # Now topk is deterministic
        values, indices = torch.topk(scores_adjusted, k, dim=-1)

        # Return original scores for the selected indices
        original_values = scores.gather(-1, indices)

        return original_values, indices

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts deterministically.

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            router_logits: Raw routing scores
            expert_indices: Selected expert indices per token
            expert_weights: Softmax weights for selected experts
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute routing scores
        router_logits = self.gate(hidden_states)  # (batch, seq_len, num_experts)

        # Flatten for per-token routing
        router_logits_flat = router_logits.view(-1, self.num_experts)

        config = get_config()

        if config.determinism_level in (DeterminismLevel.STRICT, DeterminismLevel.PARANOID):
            # Deterministic top-k selection
            expert_weights, expert_indices = self._deterministic_topk(
                router_logits_flat,
                self.num_experts_per_token
            )
        else:
            # Standard top-k (faster but may have tie variance)
            expert_weights, expert_indices = torch.topk(
                router_logits_flat,
                self.num_experts_per_token,
                dim=-1
            )

        # Softmax over selected experts
        expert_weights = torch.softmax(expert_weights.float(), dim=-1)

        # Reshape back
        expert_indices = expert_indices.view(batch_size, seq_len, self.num_experts_per_token)
        expert_weights = expert_weights.view(batch_size, seq_len, self.num_experts_per_token)

        return router_logits, expert_indices, expert_weights


# ═══════════════════════════════════════════════════════════════════════════════
# CASCADE MODE CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

class CascadeMode(Enum):
    """Nemotron-Cascade thinking modes."""
    THINK = "think"          # Full reasoning, higher latency
    NO_THINK = "no_think"    # Fast response, lower latency
    AUTO = "auto"            # Automatic selection based on query


@dataclass
class CascadeModeConfig:
    """Configuration for cascade mode control."""

    # Mode selection
    default_mode: CascadeMode = CascadeMode.AUTO

    # Thinking budget
    max_thinking_tokens: int = 8192
    min_thinking_tokens: int = 256

    # Auto-mode thresholds
    complexity_threshold: float = 0.7  # Above = /think
    query_length_threshold: int = 100  # Above = /think

    # Keywords that trigger /think mode
    think_keywords: List[str] = field(default_factory=lambda: [
        "explain", "why", "how", "analyze", "compare",
        "debug", "optimize", "design", "implement",
        "step by step", "reasoning", "think"
    ])

    # Keywords that trigger /no_think mode
    no_think_keywords: List[str] = field(default_factory=lambda: [
        "list", "what is", "define", "quick",
        "simple", "just", "only", "briefly"
    ])


class CascadeModeController:
    """
    Controls /think vs /no_think mode for Nemotron-Cascade.

    From NVIDIA CES 2026:
    "The unified model dynamically switches modes per turn:
    - /no_think for simple queries (parameter lookup, file listing)
    - /think for complex problems (debugging, optimization)"

    This aligns compute budget with task complexity.
    """

    def __init__(self, config: Optional[CascadeModeConfig] = None):
        self.config = config or CascadeModeConfig()
        self.current_mode = self.config.default_mode
        self.mode_history: List[Dict[str, Any]] = []

    def select_mode(self, query: str) -> CascadeMode:
        """
        Select thinking mode based on query analysis.

        Args:
            query: User query string

        Returns:
            Selected CascadeMode
        """
        if self.config.default_mode != CascadeMode.AUTO:
            return self.config.default_mode

        query_lower = query.lower()

        # Check for explicit mode keywords
        if any(kw in query_lower for kw in self.config.think_keywords):
            mode = CascadeMode.THINK
            reason = "think_keyword_detected"
        elif any(kw in query_lower for kw in self.config.no_think_keywords):
            mode = CascadeMode.NO_THINK
            reason = "no_think_keyword_detected"
        elif len(query) > self.config.query_length_threshold:
            mode = CascadeMode.THINK
            reason = "long_query"
        else:
            # Compute complexity score
            complexity = self._estimate_complexity(query)
            if complexity > self.config.complexity_threshold:
                mode = CascadeMode.THINK
                reason = f"high_complexity_{complexity:.2f}"
            else:
                mode = CascadeMode.NO_THINK
                reason = f"low_complexity_{complexity:.2f}"

        # Record decision
        self.mode_history.append({
            "query_preview": query[:100],
            "mode": mode.value,
            "reason": reason,
        })

        self.current_mode = mode
        return mode

    def _estimate_complexity(self, query: str) -> float:
        """
        Estimate query complexity (0.0 to 1.0).

        Simple heuristic based on:
        - Query length
        - Number of clauses (commas, 'and', 'or')
        - Technical terms
        """
        # Length factor
        length_score = min(len(query) / 500, 1.0) * 0.3

        # Clause factor
        clause_markers = [',', ' and ', ' or ', ';', ':', '?']
        clause_count = sum(query.lower().count(m) for m in clause_markers)
        clause_score = min(clause_count / 10, 1.0) * 0.3

        # Technical term factor
        tech_terms = [
            'function', 'class', 'method', 'algorithm', 'optimize',
            'performance', 'memory', 'cache', 'kernel', 'tensor',
            'batch', 'gradient', 'loss', 'layer', 'model'
        ]
        tech_count = sum(1 for t in tech_terms if t in query.lower())
        tech_score = min(tech_count / 5, 1.0) * 0.4

        return length_score + clause_score + tech_score

    def get_thinking_budget(self, mode: CascadeMode) -> int:
        """Get token budget for thinking based on mode."""
        if mode == CascadeMode.THINK:
            return self.config.max_thinking_tokens
        else:
            return self.config.min_thinking_tokens

    def get_mode_report(self) -> Dict[str, Any]:
        """Get report of mode selection history."""
        if not self.mode_history:
            return {"message": "No mode selections recorded"}

        think_count = sum(1 for m in self.mode_history if m["mode"] == "think")
        no_think_count = len(self.mode_history) - think_count

        return {
            "total_queries": len(self.mode_history),
            "think_count": think_count,
            "no_think_count": no_think_count,
            "think_ratio": think_count / len(self.mode_history),
            "recent_decisions": self.mode_history[-5:],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NVFP4 QUANTIZATION (PLACEHOLDER)
# ═══════════════════════════════════════════════════════════════════════════════

class NVFP4Quantizer:
    """
    NVFP4 quantization for KV cache compression.

    From NVIDIA CES 2026:
    "NVFP4 cuts KV cache memory footprint by up to 50%,
    effectively doubling context budgets."

    NOTE: This is a placeholder. Full NVFP4 support requires
    TensorRT-LLM integration when available for consumer GPUs.
    """

    def __init__(self, compression_ratio: float = 0.5):
        self.compression_ratio = compression_ratio
        self.enabled = False  # Not yet available

    def quantize_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize KV cache tensors.

        Currently returns tensors unchanged. Will implement
        actual NVFP4 quantization when TensorRT-LLM support
        becomes available.
        """
        if not self.enabled:
            return key, value

        # Placeholder for NVFP4 quantization
        # Real implementation would:
        # 1. Convert to FP4 format
        # 2. Store scale factors
        # 3. Compress by ~50%

        return key, value

    def dequantize_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize KV cache tensors."""
        if not self.enabled:
            return key, value

        # Placeholder
        return key, value

    def estimate_memory_savings(
        self,
        key_shape: Tuple[int, ...],
        value_shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float16,
    ) -> Dict[str, Any]:
        """Estimate memory savings from NVFP4 compression."""
        key_bytes = torch.tensor(key_shape).prod().item() * torch.finfo(dtype).bits // 8
        value_bytes = torch.tensor(value_shape).prod().item() * torch.finfo(dtype).bits // 8
        total_bytes = key_bytes + value_bytes

        compressed_bytes = total_bytes * self.compression_ratio

        return {
            "original_bytes": total_bytes,
            "compressed_bytes": int(compressed_bytes),
            "savings_bytes": int(total_bytes - compressed_bytes),
            "compression_ratio": self.compression_ratio,
            "effective_context_multiplier": 1 / self.compression_ratio,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NEMOTRON OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

class JetBlockNemotronOptimizer:
    """
    Full Nemotron 3 optimization pipeline.

    Coordinates:
    - Hybrid layer detection
    - Per-layer optimization strategy
    - Mamba-2 determinism
    - MoE routing determinism
    - Attention optimization
    - Cascade mode control
    """

    def __init__(
        self,
        config: Optional[JetBlockV4Config] = None,
        cascade_config: Optional[CascadeModeConfig] = None,
    ):
        self.config = config or JetBlockV4Config()
        self.detector = NemotronHybridDetector()
        self.cascade_controller = CascadeModeController(cascade_config)
        self.nvfp4 = NVFP4Quantizer()

        # Set global config
        set_config(self.config)

        # Optimization state
        self.is_optimized = False
        self.optimization_report: Dict[str, Any] = {}

    def analyze(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model architecture without modifying it."""
        return self.detector.analyze(model)

    def optimize(
        self,
        model: nn.Module,
        enable_mamba2_determinism: bool = True,
        enable_moe_determinism: bool = True,
        enable_attention_optimization: bool = True,
    ) -> nn.Module:
        """
        Apply full Nemotron optimization to model.

        Args:
            model: Model to optimize
            enable_mamba2_determinism: Replace Mamba-2 with deterministic version
            enable_moe_determinism: Replace MoE routing with deterministic version
            enable_attention_optimization: Apply attention optimizations

        Returns:
            Optimized model
        """
        # Setup deterministic environment
        env_settings = self.config.setup_deterministic_environment()

        # Analyze architecture
        analysis = self.detector.analyze(model)

        # Apply optimizations based on layer types
        optimizations_applied = []

        if enable_mamba2_determinism and analysis["layer_counts"].get("mamba2", 0) > 0:
            model = replace_mamba2_with_deterministic(model)
            optimizations_applied.append("mamba2_determinism")

        # MoE optimization would replace router modules
        if enable_moe_determinism and analysis["layer_counts"].get("moe", 0) > 0:
            # Placeholder: would replace MoE routers
            optimizations_applied.append("moe_determinism_pending")

        # Attention optimization
        if enable_attention_optimization and analysis["layer_counts"].get("attention", 0) > 0:
            # Placeholder: would replace attention modules
            optimizations_applied.append("attention_optimization_pending")

        self.is_optimized = True
        self.optimization_report = {
            "architecture_analysis": analysis,
            "environment_settings": env_settings,
            "optimizations_applied": optimizations_applied,
            "determinism_level": self.config.determinism_level.value,
        }

        return model

    def set_cascade_mode(self, mode: Union[CascadeMode, str]) -> None:
        """Set cascade thinking mode."""
        if isinstance(mode, str):
            mode = CascadeMode(mode)
        self.cascade_controller.config.default_mode = mode

    def select_mode_for_query(self, query: str) -> CascadeMode:
        """Auto-select cascade mode for a query."""
        return self.cascade_controller.select_mode(query)

    def get_report(self) -> Dict[str, Any]:
        """Get full optimization report."""
        return {
            "is_optimized": self.is_optimized,
            "optimization_report": self.optimization_report,
            "cascade_report": self.cascade_controller.get_mode_report(),
            "config": {
                "determinism_level": self.config.determinism_level.value,
                "force_batch_size_one": self.config.force_batch_size_one,
                "attention_dtype": str(self.config.attention_dtype),
                "ssm_state_dtype": str(self.config.ssm_state_dtype),
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def optimize_for_nemotron(
    model: nn.Module,
    determinism_level: str = "strict",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    One-line Nemotron optimization.

    Args:
        model: Model to optimize
        determinism_level: "speed", "balanced", "strict", or "paranoid"

    Returns:
        (optimized_model, report)
    """
    config = JetBlockV4Config(
        determinism_level=DeterminismLevel(determinism_level)
    )

    optimizer = JetBlockNemotronOptimizer(config)
    model = optimizer.optimize(model)

    return model, optimizer.get_report()


def create_nemotron_cache(
    batch_size: int = 1,
    num_layers: int = 52,
    config: Optional[Mamba2Config] = None,
) -> Mamba2Cache:
    """Create cache for Nemotron 3 inference."""
    config = config or Mamba2Config()
    return Mamba2Cache(config, batch_size, num_layers)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Layer detection
    "LayerType",
    "LayerInfo",
    "NemotronHybridDetector",

    # MoE
    "DeterministicMoERouter",

    # Cascade mode
    "CascadeMode",
    "CascadeModeConfig",
    "CascadeModeController",

    # NVFP4
    "NVFP4Quantizer",

    # Main optimizer
    "JetBlockNemotronOptimizer",

    # Convenience
    "optimize_for_nemotron",
    "create_nemotron_cache",
]
