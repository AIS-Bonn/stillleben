#!/usr/bin/env python

import os
SL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

import stillleben as sl
import torch

sl.init() # use sl.init_cuda() for CUDA interop

# Load a mesh
mesh = sl.Mesh(os.path.join(SL_PATH, 'tests/stanford_bunny/scene.gltf'))

# Meshes can come in strange dimensions - rescale to something reasonable
mesh.scale_to_bbox_diagonal(0.5)

# Create a scene with a few bunnies
scene = sl.Scene((1920,1080))

for i in range(5):
    obj = sl.Object(mesh)
    scene.add_object(obj)

# Let them fall in a heap
scene.simulate_tabletop_scene()

# Setup lighting
scene.choose_random_light_position()
scene.ambient_light = torch.tensor([0.3, 0.3, 0.3])

# Display a plane & set background color
scene.background_plane_size = torch.tensor([1.0, 1.0])
scene.background_plane_texture = sl.Texture2D(os.path.join(SL_PATH, 'tests/texture.jpg'))
scene.background_color = torch.tensor([0.1, 0.1, 0.1, 1.0])

# Render a frame
renderer = sl.RenderPass()
result = renderer.render(scene)
print('Resulting RGB frame:', result.rgb().shape)
print('Resulting segmentation frame:', result.instance_index().shape)

# Display interactive viewer
sl.view(scene)
