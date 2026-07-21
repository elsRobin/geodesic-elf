"""Geodesic Energy Ratio — Training-dynamics diagnostic for continuous diffusion language models.

Quantifies how the ODE denoising path's trajectory through LM Head output
probability space evolves during flow matching training.
"""

from experiments.geodesic_analysis.diagnostics import (
    TraceConfig,
    generate_with_intermediates,
    generate_multiple_traces,
    analyze_semantic_progression,
    print_trace,
    print_trace_comparison,
    print_analysis,
    trace_to_json,
)

__all__ = [
    "TraceConfig",
    "generate_with_intermediates",
    "generate_multiple_traces",
    "analyze_semantic_progression",
    "print_trace",
    "print_trace_comparison",
    "print_analysis",
    "trace_to_json",
]
