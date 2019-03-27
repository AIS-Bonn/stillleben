
"""
The stillleben package.
"""

import torch # important, otherwise we get undefined references during ._C import
import os
from . import camera_model
from . import diff
from ._C import *
from ._C import _set_install_prefix

__all__ = [
    'init',
    'init_cuda',
    'render_debug_image',
    'render_physics_debug_image',
    'Scene',
    'Mesh',
    'Object',
    'Range3D',
    'RenderPass',
    'RenderPassResult',
    'Texture',

    'camera_model',
    'diff',
]


STILLLEBEN_PATH = os.path.dirname(os.path.abspath(__file__))
_set_install_prefix(os.path.join(STILLLEBEN_PATH))
