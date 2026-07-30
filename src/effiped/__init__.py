"""EffiPed: joint pedestrian detection, tracking, and cross-camera review."""

from .model import JDENet, build_jdenet_from_config

__all__ = ["JDENet", "build_jdenet_from_config"]
__version__ = "1.0.0"

