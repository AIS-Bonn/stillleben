
import torch.utils.cpp_extension

import pathlib

SL_PATH = pathlib.Path(__file__).parent.absolute()

def load(**kwargs):
    includes = kwargs.get('extra_include_paths', [])
    kwargs['extra_include_paths'] = includes + [str(SL_PATH / 'include')]

    cflags = kwargs.get('extra_cflags', [])
    kwargs['extra_cflags'] = ['-DNDEBUG', '-O3', '-std=c++17'] + cflags

    ldflags = kwargs.get('extra_ldflags', [])
    kwargs['extra_ldflags'] = [f'-L{SL_PATH}/lib', '-lstillleben', '-lstillleben_python']

    return torch.utils.cpp_extension.load(**kwargs)
