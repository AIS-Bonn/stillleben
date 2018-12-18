
"""
The stillleben package.
"""

import torch # important, otherwise we get undefined references during ._C import
from ._C import *

__all__ = [
    'init',
    'initCUDA',
    'Scene',
    'Mesh',
    'Object',
    'RenderPass',
    'RenderPassResult'
]
