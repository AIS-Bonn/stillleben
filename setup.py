
import setuptools
import setuptools.command.install
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
import subprocess
import os
import sys
import torch
import glob

BUILD_PATH = os.path.join(os.getcwd(), 'python', 'cpp_build')
INSTALL_PATH = os.path.join(os.getcwd(), 'python', 'stillleben')

DEBUG = int(os.environ.get('DEBUG', '0')) == 1

def build_stillleben():
    os.makedirs(BUILD_PATH, exist_ok=True)

    env = os.environ.copy()
    env['CLICOLOR_FORCE'] = '1'

    if (not os.path.exists(os.path.join(BUILD_PATH, 'build.ninja'))) or os.environ.get('CMAKE', '0') == '1':
        cmd = [
            'cmake',
            '-DCMAKE_BUILD_TYPE={}'.format('Debug' if DEBUG else 'RelWithDebInfo'),
            '-DCMAKE_INSTALL_PREFIX=' + INSTALL_PATH,
            '-DUSE_RELATIVE_RPATH=ON',
            '-GNinja',
            '../..'
        ]

        # if CUDA_HOME is available, respect it to use non-default CUDA versions
        if 'CUDA_HOME' in env:
            cmd.append('-DCUDA_TOOLKIT_ROOT_DIR={}'.format(env['CUDA_HOME']))

        # are we installing inside anaconda?
        # if so, try to be helpful.
        if 'CONDA_PREFIX' in env:
            cmd.append(f'-DEXTRA_RPATH={env["CONDA_PREFIX"]}/lib')

            if 'CMAKE_PREFIX_PATH' in env:
                env['CMAKE_PREFIX_PATH'] = env['CONDA_PREFIX'] + ':' + env['CMAKE_PREFIX_PATH']
            else:
                env['CMAKE_PREFIX_PATH'] = env['CONDA_PREFIX']

        if subprocess.call(cmd, cwd=BUILD_PATH, env=env) != 0:
            print('Failed to run "{}"'.format(' '.join(cmd)))

    make_cmd = [
        'ninja',
        'install',
    ]

    if subprocess.call(make_cmd, cwd=BUILD_PATH, env=env) != 0:
        raise RuntimeError('Failed to call ninja')

# Build all dependent libraries
class build_deps(setuptools.Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
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
        if not self.skip_build:
            self.run_command('build_deps')

        setuptools.command.install.install.run(self)

def make_relative_rpath(path):
    return '-Wl,-rpath,$ORIGIN/' + path

cmdclass = {
    'build_deps': build_deps,
    'install': install,
}

MAGNUM_SUFFIX = '-d' if DEBUG else ''

setuptools.setup(
    name='stillleben',
    cmdclass=cmdclass,
    packages=['stillleben'],
    package_dir={'':'python'},
    include_package_data=True,
    ext_modules=[],
    options={
        'build': {
            'build_base': 'python/py_build'
        },
        'sdist': {
            'dist_dir': 'python/dist'
        },
    },
)
