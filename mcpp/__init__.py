"""mcpp - Multi-robot Complete Coverage Path Planning on grids.

DARP + Spanning Tree Coverage, partition baselines, a min-max local search,
the Even et al. min-max rooted tree cover approximation and exact MIP models.
"""
from .darp import DARPConfig, darp
from .grid import CoarseMap, GridMap, build_coarse, load_movingai, sample_starts
from .planner import METHODS, plan

__all__ = ["DARPConfig", "darp", "CoarseMap", "GridMap", "build_coarse", "load_movingai",
           "sample_starts", "METHODS", "plan"]
__version__ = "1.0.0"
