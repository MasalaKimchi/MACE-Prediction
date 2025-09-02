"""
Architectures module for survival analysis models.
"""

from .resnet3d import build_network, print_resnet3d_param_counts

__all__ = ['build_network', 'print_resnet3d_param_counts']
