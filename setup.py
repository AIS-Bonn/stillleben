
import setuptools
import setuptools.command.install
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
import subprocess
import os
import sys
import torch

BUILD_PATH = os.path.join(os.getcwd(), 'cpp_build')
INSTALL_PATH = os.path.join(os.getcwd(), 'stillleben')

def build_stillleben():
    os.makedirs(BUILD_PATH, exist_ok=True)

    cmd = [
        'cmake',
        '-DCMAKE_BUILD_TYPE=RelWithDebInfo',
        '-DCMAKE_INSTALL_PREFIX=' + INSTALL_PATH,
        '-GNinja',
        '..'
    ]

    if subprocess.call(cmd, cwd=BUILD_PATH) != 0:
        print('Failed to run "{}"'.format(' '.join(cmd)))

    make_cmd = [
        'ninja',
        'install',
    ]

    if subprocess.call(make_cmd, cwd=BUILD_PATH) != 0:
        raise RuntimeError('Failed to call ninja')

class PytorchCommand(setuptools.Command):
    """
    Base Pytorch command to avoid implementing initialize/finalize_options in
    every subclass
    """
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

# Build all dependent libraries
class build_deps(PytorchCommand):
    def run(self):
        print('setup.py::build_deps::run()')
        # Check if you remembered to check out submodules

        def check_file(f):
            if not os.path.exists(f):
                print("Could not find {}".format(f))
                print("Did you run 'git submodule update --init --recursive'?")
                sys.exit(1)

        check_file(os.path.join(os.getcwd(), 'contrib', 'corrade', 'CMakeLists.txt'))
        check_file(os.path.join(os.getcwd(), 'contrib', 'magnum', 'CMakeLists.txt'))
        check_file(os.path.join(os.getcwd(), 'contrib', 'magnum-plugins', 'CMakeLists.txt'))

        build_stillleben()

class install(setuptools.command.install.install):

    def run(self):
        print('setup.py::run()')
        if not self.skip_build:
            self.run_command('build_deps')

        setuptools.command.install.install.run(self)

def make_relative_rpath(path):
    return '-Wl,-rpath,$ORIGIN/' + path

cmdclass = {
    'build_deps': build_deps,
    'build_ext': BuildExtension,
    'install': install,
}

if torch.version.cuda is not None:
    print('CUDA detected!')
    ExtensionType = CUDAExtension
    extra_defs = ['-DHAVE_CUDA=1']
else:
    print('No CUDA found, interop disabled...')
    ExtensionType = CppExtension
    extra_defs = []

setuptools.setup(
    name='stillleben',
    cmdclass=cmdclass,
    packages=['stillleben'],
    package_data={
        'stillleben': [
            'lib/*.so*',
            'lib/magnum/importers/*',
        ]
    },
    ext_modules=[
        ExtensionType(
            name='stillleben._C',
            sources=['stillleben/bridge.cpp'],
            extra_compile_args=[
                '-I' + os.path.join(os.getcwd(), 'stillleben', 'include'),
                '-std=c++17',
            ] + extra_defs,
            extra_link_args=[
                os.path.join(os.getcwd(), 'stillleben', 'lib', 'libstillleben.so'),
                make_relative_rpath('lib')
            ],
            depends=[
                os.path.join(os.getcwd(), 'stillleben', 'lib', 'libstillleben.so'),
            ]
        )
    ],
)
