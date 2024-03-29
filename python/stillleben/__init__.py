
"""
The stillleben package.
"""

import torch # important, otherwise we get undefined references during ._C import
import os
from . import camera_model
from . import diff
from . import losses
from . import extension
from .lib.libstillleben_python import *
from .lib.libstillleben_python import _set_install_prefix

__all__ = [
    'init',
    'init_cuda',
    'render_debug_image',
    'Animator',
    'ImageLoader',
    'ImageSaver',
    'LightMap',
    'Mesh',
    'MeshCache',
    'Object',
    'Range3D',
    'RenderPass',
    'RenderPassResult',
    'Scene',
    'Texture',
    'Texture2D',
    'Viewer',
    'view',

    'camera_model',
    'diff',
    'extension',
    'losses',

    'quat_to_matrix',
    'matrix_to_quat',
]


STILLLEBEN_PATH = os.path.dirname(os.path.abspath(__file__))
_set_install_prefix(os.path.join(STILLLEBEN_PATH))
