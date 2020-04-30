
.. role:: sh(code)
    :language: sh

Installation
============

System requirements
-------------------

stillleben has been developed and tested on Ubuntu 18.04. Specifically, it
requires EGL, which did not work reliably in earlier Ubuntu versions.
Your mileage on other Linux distributions may vary. Other platforms have never
been tested and are unlikely to work.

If you want to use the Python API (you probably do), you will need PyTorch 1.5.
If you want to use CUDA, please install it first.
Then install PyTorch from source to avoid compiler ABI issues.
Here is a short guide:

.. code:: sh

    # Install Anaconda (skip this if you already have it)
    # you may adapt the link to a newer Anaconda version :)
    wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
    bash Anaconda3-2020.02-Linux-x86_64.sh

    # Make sure to enter the Anaconde environment here!
    # If you chose auto-activation during installation, just open a new shell.
    # Otherwise, source anaconda/bin/activate.

    # Install PyTorch requirements
    conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi
    conda install -c pytorch magma-cuda101 # adapt the 101 to your CUDA version!

    # Install PyTorch 1.5.0
    git clone --recursive --depth 1 -b v1.5.0 https://github.com/pytorch/pytorch.git
    cd pytorch
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
    python setup.py install

Compiling stillleben
--------------------

You can compile stillleben like any other Python package. If you are using
Anaconda (see above), make sure the Anaconda environment is loaded.

.. code:: sh

    cd stillleben
    python setup.py install
