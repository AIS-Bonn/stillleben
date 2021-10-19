
import unittest

import torch
import stillleben as sl
import numpy as np

from stillleben.camera_model import process_image
from stillleben.profiling import Timer

import os

from PIL import Image

TESTS_PATH = os.path.dirname(os.path.abspath(__file__))

class PythonTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.is_available():
            print('Running tests with CUDA')
            sl.init_cuda(0)
        else:
            print('Running tests without CUDA')
            sl.init()

    @Timer('test_render')
    def test_render(self):
        scene = sl.Scene((640,480))

        with Timer('Mesh load'):
            mesh = sl.Mesh(os.path.join(TESTS_PATH, 'stanford_bunny', 'scene.gltf'))
        mesh.center_bbox()
        mesh.scale_to_bbox_diagonal(0.5)
        object = sl.Object(mesh)

        print(f'Mesh points: {mesh.points.size()}')

        scene.add_object(object)

        pose = torch.eye(4)
        pose[2,3] = 0.5
        object.set_pose(pose)

        with Timer('render'):
            renderer = sl.RenderPass()
            result = renderer.render(scene)

        with Timer('retrieve'):
            rgb = result.rgb()

        print("First pixel:", rgb[0,0])
        print("min: {}, max: {}".format(rgb.min(), rgb.max()))

        rgb_np = rgb.cpu().numpy()

        img = Image.fromarray(rgb_np, mode='RGBA')
        img.save('/tmp/stillleben.png')

        dbg = sl.render_debug_image(scene)
        dbg_np = dbg.cpu().numpy()

        dbg_img = Image.fromarray(dbg_np, mode='RGBA')
        dbg_img.save('/tmp/stillleben_debug.png')

        rgb_noise = process_image(rgb.permute(2, 0, 1)[:3].float() / 255.0)
        rgb_noise_np = (rgb_noise * 255).byte().permute(1,2,0).contiguous().cpu().numpy()
        noise_img = Image.fromarray(rgb_noise_np, mode='RGB')
        noise_img.save('/tmp/stillleben_noise.png')

    def test_serialization(self):
        scene = sl.Scene((640,480))

        mesh = sl.Mesh(os.path.join(TESTS_PATH, 'stanford_bunny', 'scene.gltf'))
        mesh.center_bbox()
        mesh.scale_to_bbox_diagonal(0.5)
        object = sl.Object(mesh)

        scene.add_object(object)

        pose = torch.eye(4)
        pose[2,3] = 0.5
        object.set_pose(pose)

        ser = scene.serialize()

        scene2 = sl.Scene((640,480))
        scene2.deserialize(ser)

        self.assertLess((scene.objects[0].pose() - scene2.objects[0].pose()).norm(), 1e-9)
        self.assertLess((scene.objects[0].mesh.pretransform - scene2.objects[0].mesh.pretransform).norm(), 1e-5)

    def test_image_saver(self):
        with sl.ImageSaver() as saver:
            saver.save(torch.zeros(640,480,3, dtype=torch.uint8), '/tmp/test_color.png')
            saver.save(torch.zeros(640,480, dtype=torch.uint8), '/tmp/test_gray8.png')

            saver.save(torch.zeros(640,480, dtype=torch.int16), '/tmp/test_gray16.png')

        color = np.array(Image.open('/tmp/test_color.png'))
        self.assertEqual(color.shape, (640,480,3))
        self.assertEqual(color.dtype, np.uint8)

        gray8 = np.array(Image.open('/tmp/test_gray8.png'))
        self.assertEqual(gray8.shape, (640,480))
        self.assertEqual(gray8.dtype, np.uint8)

        gray16 = np.array(Image.open('/tmp/test_gray16.png'))
        self.assertEqual(gray16.shape, (640,480))
        self.assertEqual(gray16.dtype, np.int32)

    def test_physics(self):
        scene = sl.Scene((640,480))

        mesh = sl.Mesh(os.path.join(TESTS_PATH, 'stanford_bunny', 'scene.gltf'))
        mesh.center_bbox()
        mesh.scale_to_bbox_diagonal(0.5)
        object = sl.Object(mesh)

        object.linear_velocity = torch.tensor([100.0, 0.0, 0.0])

        scene.add_object(object)

        scene.simulate(0.002)

        # The horizontal velocity should not have changed
        self.assertAlmostEqual(object.linear_velocity[0], 100.0)
        self.assertAlmostEqual(object.linear_velocity[1], 0.0)

        # And we should have accelerated downwards (gravity)
        self.assertLess(object.linear_velocity[2], -0.0001)


if __name__ == "__main__":
    Timer.enabled = True
    unittest.main()
