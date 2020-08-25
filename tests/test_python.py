
import unittest

import torch
import stillleben as sl

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


if __name__ == "__main__":
    Timer.enabled = True
    unittest.main()
