"""
ComfyUI JetBlock Optimizer v4.0
===============================

Full Nemotron 3 hybrid architecture support with:
- Mamba-2 deterministic execution
- MoE routing determinism
- Batch-invariant operators (ThinkingMachines)
- Cascade mode control (/think vs /no_think)
- NVFP4 quantization support (coming soon)

Optimized for RTX 4090 with 24GB VRAM

References:
- NVIDIA CES 2026 Context Memory Platform
- ThinkingMachines batch-invariant-ops research
- Nemotron 3 Technical Report
"""

__version__ = "4.0.0"

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', '__version__']

WEB_DIRECTORY = "./js"

# Startup message
print("=" * 60)
print(f"JetBlock Optimizer v{__version__} loaded")
print("=" * 60)
print("NEW in v4.0:")
print("  - Full Nemotron 3 hybrid architecture support")
print("  - Mamba-2 deterministic execution")
print("  - Batch-invariant operators")
print("  - Cascade mode control (/think vs /no_think)")
print("=" * 60)
print("Nodes: JetBlock/Nemotron, JetBlock/Deterministic")
print("=" * 60)