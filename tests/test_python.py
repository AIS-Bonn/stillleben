
import unittest

import torch
import stillleben as sl

import os

import cv2

from PIL import Image

TESTS_PATH = os.path.dirname(os.path.abspath(__file__))

class PythonTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.is_available():
            sl.initCUDA(0)
        else:
            sl.init()

    def test_render(self):
        scene = sl.Scene((640,480))

        mesh = sl.Mesh(os.path.join(TESTS_PATH, 'stanford_bunny', 'scene.gltf'))
        mesh.center_bbox()
        mesh.scale_to_bbox_diagonal(0.5)
        object = sl.Object(mesh)

        scene.add_object(object)

        pose = torch.eye(4)
        pose[2,3] = -2.0
        object.set_pose(pose)

        renderer = sl.RenderPass()
        result = renderer.render(scene)

        rgb = result.rgb()

        print("First pixel:", rgb[0,0])
        print("min: {}, max: {}".format(rgb.min(), rgb.max()))

        rgb_np = rgb.cpu().numpy()

        img = Image.fromarray(rgb_np, mode='RGBA')
        img.save('/tmp/stillleben.png')

if __name__ == "__main__":
    unittest.main()
