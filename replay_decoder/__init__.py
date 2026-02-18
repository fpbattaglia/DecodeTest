"""
Multivariate Decoding Analysis for Neural Replay

Python implementation of the algorithm from Liu et al. (2019) Cell:
"Human replay spontaneously reorganizes experience"

This package provides tools for detecting sequential replay of neural
representations during rest periods using MEG/EEG data.
"""

from .decoder import MultivariateReplayDecoder
from .visualization import (
    plot_sequenceness,
    plot_reactivations,
    plot_transition_matrix,
    plot_comparison
)
from .utils import (
    create_linear_transition_matrix,
    create_circular_transition_matrix,
    simulate_replay_data,
    find_replay_events
)

__version__ = "0.1.0"
__author__ = "Based on Liu et al. (2019)"
__all__ = [
    "MultivariateReplayDecoder",
    "plot_sequenceness",
    "plot_reactivations",
    "plot_transition_matrix",
    "plot_comparison",
    "create_linear_transition_matrix",
    "create_circular_transition_matrix",
    "simulate_replay_data",
    "find_replay_events",
]
