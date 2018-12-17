
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='stillleben',
    ext_modules=[CppExtension('stillleben_c', ['stillleben/bridge.cpp'])],
    cmdclass={'build_ext': BuildExtension}
)
