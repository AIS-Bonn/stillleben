#!/usr/bin/env python

"""
This file demonstrates the usage of the stillleben library by generating
synthetic scenes that have roughly similar composition to the scenes of
the YCB Video Dataset.

For details on the dataset, see
Yu Xiang, et al. 2017:
PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes
https://arxiv.org/abs/1711.00199
"""

import stillleben as sl
import pathlib
import random
import torch
from PIL import Image

# Some boring details about the dataset
CLASSES = (
    '__background__',
    '002_master_chef_can', '003_cracker_box', '004_sugar_box',
    '005_tomato_soup_can', '006_mustard_bottle', '007_tuna_fish_can',
    '008_pudding_box', '009_gelatin_box', '010_potted_meat_can',
    '011_banana', '019_pitcher_base', '021_bleach_cleanser',
    '024_bowl', '025_mug', '035_power_drill', '036_wood_block',
    '037_scissors', '040_large_marker', '051_large_clamp',
    '052_extra_large_clamp', '061_foam_brick',
)
RESOLUTION = (640, 480)
INTRINSICS = (1066.778, 1067.487, 312.9869, 241.3109)

def run(ycb_path, ibl_path, plane_texture_path):
    mesh_path = pathlib.Path(ycb_path) / 'models'
    if ibl_path:
        ibl_path = pathlib.Path(ibl_path)

    sl.init() # use sl.init_cuda() for CUDA interop

    # Load all meshes
    meshes = sl.Mesh.load_threaded([ mesh_path / c / 'textured.obj' for c in CLASSES[1:]])

    # Setup class IDs
    for i, mesh in enumerate(meshes):
        mesh.class_index = i+1

    # Create a scene with matching intrinsics
    scene = sl.Scene(RESOLUTION)
    scene.set_camera_intrinsics(*INTRINSICS)

    for mesh in random.sample(meshes, 10):
        obj = sl.Object(mesh)

        # Override the metallic/roughness parameters so that it gets interesting
        obj.metallic = random.random()
        obj.roughness = random.random()
        scene.add_object(obj)

    # Let them fall in a heap
    scene.simulate_tabletop_scene()

    # Setup lighting
    if ibl_path:
        scene.light_map = sl.LightMap(ibl_path)
    else:
        scene.choose_random_light_position()

    # Display a plane & set background color
    scene.background_plane_size = torch.tensor([3.0, 3.0])
    scene.background_color = torch.tensor([0.1, 0.1, 0.1, 1.0])

    if plane_texture_path:
        scene.background_plane_texture = sl.Texture2D(plane_texture_path)

    # Display interactive viewer
    sl.view(scene)

    # Render a frame
    renderer = sl.RenderPass()
    result = renderer.render(scene)

    # Save as JPEG
    Image.fromarray(result.rgb()[:,:,:3].cpu().numpy()).save('rgb.jpeg')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', metavar='PATH', type=str,
        help='The path to the directory containing the "models" directory')
    parser.add_argument('--ibl', metavar='FILE.IBL', type=str,
        help='Use environment light map')
    parser.add_argument('--plane-texture', metavar='IMAGE', type=str,
        help='Use plane texture')

    args = parser.parse_args()

    run(ycb_path=args.dataset, ibl_path=args.ibl, plane_texture_path=args.plane_texture)
