
"""
The stillleben package.
"""

import torch # important, otherwise we get undefined references during ._C import
import os
from . import camera_model
from ._C import init, init_cuda, Scene, Mesh, Object, RenderPass, \
    RenderPassResult, Texture, _set_install_prefix, render_debug_image

__all__ = [
    'init',
    'init_cuda',
    'render_debug_image',
    'Scene',
    'Mesh',
    'Object',
    'Range3D',
    'RenderPass',
    'RenderPassResult',
    'Texture',

    'camera_model',
]


STILLLEBEN_PATH = os.path.dirname(os.path.abspath(__file__))
_set_install_prefix(os.path.join(STILLLEBEN_PATH))
