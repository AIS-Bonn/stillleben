#!/usr/bin/env python

import pathlib
SL_PATH = pathlib.Path(__file__).parent.parent.absolute()

import stillleben as sl
import torch
import sys
from PIL import Image

sl.init() # use sl.init_cuda() for CUDA interop

# Load a mesh
if len(sys.argv) > 1:
    mesh = sl.Mesh(sys.argv[1])
else:
    mesh = sl.Mesh(SL_PATH / 'tests' / 'stanford_bunny' / 'scene.gltf')

# Meshes can come in strange dimensions - rescale to something reasonable
mesh.center_bbox()
mesh.scale_to_bbox_diagonal(0.5)

# Dump some object statistics
obj = sl.Object(mesh)
print("Object properties:")
print(f" - mass: {obj.mass} kg")
print(f" - density: {obj.density} kg/m^3")
print(f" - volume: {obj.volume} m^3")
print(f" - inertial frame:\n{obj.inertial_frame}")
print(f" - inertia in inertial frame: {obj.inertia}")

# Create a scene with a few bunnies
scene = sl.Scene((1920,1080))

for i in range(10):
    obj = sl.Object(mesh)
    scene.add_object(obj)

# Let them fall in a heap
scene.simulate_tabletop_scene()

# Setup lighting
scene.choose_random_light_position()
scene.ambient_light = torch.tensor([0.3, 0.3, 0.3])

# Display a plane & set background color
scene.background_plane_size = torch.tensor([3.0, 3.0])
scene.background_color = torch.tensor([0.1, 0.1, 0.1, 1.0])

# Render a frame
renderer = sl.RenderPass()
result = renderer.render(scene)
print('Resulting RGB frame:', result.rgb().shape)
print('Resulting segmentation frame:', result.instance_index().shape)

# Save as JPEG
Image.fromarray(result.rgb()[:,:,:3].cpu().numpy()).save('rgb.jpeg')

# Display interactive viewer
sl.view(scene)
