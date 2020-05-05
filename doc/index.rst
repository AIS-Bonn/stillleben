
stillleben
==========

.. figure:: examples/ycb.jpeg
    :alt: YCB-Video example
    :width: 100%

What can it do for me?
----------------------

Stillleben generates realistic arrangements of rigid bodies and provides
various outputs that can be used to train deep learning models:

* A rendered RGB image of the scene. Depending on the rendering settings, this
  rendering can be of quite high quality, in some cases photorealistic.
  The image can be imbued with noise using the :ref:`stillleben.camera_model`
  module.
* A depth image.
* Class-wise and instance-wise semantic segmentation of the scene.
* Camera- and object-space coordinates, highly useful for training
  methods that exploit pixel-wise correspondence.
* Camera-space normals.

Stillleben is highly integrated with the PyTorch_
deep learning framework. It can be used to produce training data on-line to
save dataset generation time or to train methods that require interaction with
the scene.

This page documents the Python API of stillleben. The Python API is a wrapper
around the C++ core implementation.

Getting started
---------------

If you haven't already, go through the :ref:`installation instructions <std:doc:installation>`.

Here is a short API example:

.. include:: ../examples/viewer.py
    :code: python
    :start-line: 5


.. _PyTorch: https://pytorch.org
