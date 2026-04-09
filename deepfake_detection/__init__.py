"""Deepfake detection utilities with Grad-CAM explainability support."""

from .gradcam import GradCAM, overlay_heatmap
from .preprocessing import make_default_preprocess
from .video_pipeline import DeepfakeVideoExplainer, ExplanationResult

__all__ = [
    "GradCAM",
    "overlay_heatmap",
    "make_default_preprocess",
    "DeepfakeVideoExplainer",
    "ExplanationResult",
]
