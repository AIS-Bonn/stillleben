
"""
The stillleben package.
"""

import torch # important, otherwise we get undefined references during ._C import
import os
from ._C import init, init_cuda, Scene, Mesh, Object, RenderPass, RenderPassResult, _set_install_prefix

__all__ = [
    'init',
    'init_cuda',
    'Scene',
    'Mesh',
    'Object',
    'RenderPass',
    'RenderPassResult'
]


STILLLEBEN_PATH = os.path.dirname(os.path.abspath(__file__))
_set_install_prefix(os.path.join(STILLLEBEN_PATH))
