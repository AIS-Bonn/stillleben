
from contextlib import redirect_stdout
import sys
from textwrap import dedent

import torch

# Someone in torch.utils.cpp_extension thinks it's fun to print warnings to stdout.
with redirect_stdout(sys.stderr):
    import torch.utils.cpp_extension

def generate_paths():
    use_cuda = torch.version.cuda is not None
    include_dirs = torch.utils.cpp_extension.include_paths(cuda=use_cuda)
    library_dirs = torch.utils.cpp_extension.library_paths(cuda=use_cuda)
    libraries = ['c10', 'torch', 'torch_python', '_C']

    return dedent(f"""
        set(TORCH_INCLUDE_DIRS "{';'.join(include_dirs)}")
        set(TORCH_LIBRARY_DIRS "{';'.join(library_dirs)}")
        set(TORCH_LIBRARIES "{';'.join(libraries)}")
        set(TORCH_USE_CUDA {"ON" if use_cuda else "OFF"})
        set(TORCH_VERSION "{torch.__version__}")
    """)

if __name__ == "__main__":
    print(generate_paths())
