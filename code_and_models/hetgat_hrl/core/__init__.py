"""Core formal interfaces and dataclasses."""

from .disaster_map_graph import DisasterMapGraph, MAP_COMPLEXITY_PRESETS, MapComplexitySpec

__all__ = [
    "DisasterMapGraph",
    "MAP_COMPLEXITY_PRESETS",
    "MapComplexitySpec",
]
