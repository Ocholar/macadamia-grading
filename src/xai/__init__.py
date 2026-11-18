"""
Explainable AI (XAI) Module for Macadamia Quality Grading.
Generates Grad-CAM heatmaps, visual defect attributions, and audit cards.
"""

from .gradcam import ExplainabilityEngine

__all__ = ["ExplainabilityEngine"]
