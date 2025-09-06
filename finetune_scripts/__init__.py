"""
Fine-tuning scripts for survival analysis.
Contains scripts and utilities for fine-tuning pretrained models for survival prediction.
"""

from .modality_dropout import apply_modality_dropout

__all__ = ["apply_modality_dropout"]
