'''
Unit testing for stillleben.diff.backpropagate_gradient_to_poses
http://schwarzm.pages.ais.uni-bonn.de/stillleben/stillleben.html
'''
import unittest

import torch as th
import torch.nn.functional as F
import stillleben as sl

from stillleben import camera_model

from PIL import Image
import numpy as np
import cv2
import os

VIS = False
TESTS_PATH = os.path.dirname(os.path.abspath(__file__))

def get_gaussian_pyramid_comparison(observed_img, rendered_img, mask_img):
    th_observed_img = observed_img.permute(2,0,1)
    th_rendered_img = rendered_img.permute(2,0,1)
    th_mask = mask_img

    th_rendered_img.requires_grad = True
    th_rendered_img =  th.nn.Parameter(th_rendered_img)

    o = np.tile(cv2.getGaussianKernel(3,3), (1, 3))
    kernel  = o * o.T
    th_gaussian_kernel = th.from_numpy(kernel)
    th_gaussian_kernel = th_gaussian_kernel.view(1,1,3,3).float()

    _th_rendered_img = F.conv2d(th_rendered_img.view(3, 1, 480, 640), th_gaussian_kernel, padding=1)
    th_observed_img = F.conv2d(th_observed_img.view(3, 1, 480, 640), th_gaussian_kernel, padding=1)

    th_combined_mask =  th_mask
    th_combined_mask = th_combined_mask.clamp(0, 1)

    _th_rendered_img = _th_rendered_img * th_combined_mask
    th_observed_img = th_observed_img * th_combined_mask
    th_difference = ((_th_rendered_img - th_observed_img).squeeze()) ** 2

    th_difference = th_difference.unsqueeze(0)

    in_1 = th_difference.view(3, 1, 480, 640)
    l1 = F.conv2d(in_1, th_gaussian_kernel, padding=1).view(1,-1)
    in_2 = F.interpolate(in_1, scale_factor = 0.5, mode='bilinear')
    l2 = F.conv2d(in_2, th_gaussian_kernel, padding=1).view(1,-1)
    in_3 = F.interpolate(in_2, scale_factor = 0.5, mode='bilinear')
    l3 = F.conv2d(in_3, th_gaussian_kernel, padding=1).view(1,-1)

    loss = th.cat([l1, l2, l3], dim=1)
    loss = loss.sum() / mask_img.sum()
    loss.backward()

    _grad = th_rendered_img.grad.clone()

    grad_wrt_img = _grad.view(3, 1, 480, 640)
    grad_wrt_img = grad_wrt_img.squeeze()

    return grad_wrt_img, float(loss.detach().numpy())

class TestGradients(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestGradients, self).__init__(*args, **kwargs)
        sl.init()
        self.scene = sl.Scene((640,480))

        # Taken from first YCB frame
        # NOTE: This is not constant throughout the real dataset!
        fx, fy, cx, cy = 1066.778, 1067.487, 312.9869, 241.3109
        self.scene.set_camera_intrinsics(fx, fy, cx, cy)

        mesh_files = [os.path.join(TESTS_PATH, 'stanford_bunny', 'scene.gltf')]
        self.meshes = []
        for mesh_file in mesh_files:
            mesh = sl.Mesh(mesh_file)

            print("Loaded mesh with bounding box:", mesh.bbox)
            print("center:", mesh.bbox.center, "size:", mesh.bbox.size)

            mesh.center_bbox()
            mesh.scale_to_bbox_diagonal(0.5, 'order_of_magnitude')

            print("normalized:", mesh.bbox)
            print("center:", mesh.bbox.center, "size:", mesh.bbox.size, "diagonal:", mesh.bbox.diagonal)
            self.meshes.append(mesh)

    def test_gradient(self):

        for mesh in self.meshes:
            object = sl.Object(mesh)
            self.scene.add_object(object)

            pose = th.tensor([[ 0.0596,  0.8315, -0.5523, -0.0651],
                    [ 0.4715,  0.4642,  0.7498, -0.06036],
                    [ 0.8798, -0.3051, -0.3644,  0.80551],
                    [ 0.0000,  0.0000,  0.0000,  1.0000]])
            U, S, V = th.svd(pose[:3, :3])
            pose[:3, :3] = th.matmul(U, V.t())
            object.set_pose(pose)

        renderer = sl.RenderPass()
        result = renderer.render(self.scene)
        gt_mask = (result.instance_index()!=0).float()

        rgb = result.rgb().detach()
        gt_rgb = rgb[:,:,:3]

        gt_rgb_np = gt_rgb.cpu().numpy()

        gt_pose = 0
        if VIS:
            import visdom
            from torchnet.logger import VisdomLogger
            env_name = 'stillleben'
            vis = visdom.Visdom(port=8097, env=env_name)
            vis.close(env=env_name)
            gt_logger = VisdomLogger('image', env=env_name, port=8097, opts=dict(title='gt'))
            img_logger = VisdomLogger('image', env=env_name, port=8097, opts=dict(title='rgb'))

        for param in range(6):
            for obj in self.scene.objects:
                new_pose = obj.pose().clone()
                gt_pose = obj.pose().clone()
                GT_DELTA = th.zeros(6)
                GT_DELTA[param] = 0.01

                new_pose = sl.diff.apply_pose_delta(gt_pose, GT_DELTA)
                obj.set_pose(new_pose)

                rendered_result = renderer.render(self.scene)

                rendered_rgb = rendered_result.rgb()
                rendered_rgb = rendered_rgb[:,:,:3]

                rgb_np = rendered_rgb.cpu().numpy()
                if VIS:
                    img_logger.log(rgb_np.transpose(2,0,1))
                    gt_logger.log(gt_rgb_np.transpose(2,0,1))

                gt_mask = th.ones_like(gt_mask)

                grad_wrt_img, l = get_gaussian_pyramid_comparison(gt_rgb.float() / 255.0, rendered_rgb.float() / 255.0, gt_mask.squeeze())

                delta = sl.diff.backpropagate_gradient_to_poses(self.scene, rendered_result, grad_wrt_img)

                print('GT delta', GT_DELTA)
                print('Delta', delta)
                self.assertGreater(delta[0][param], 0)

if __name__ == '__main__':
    unittest.main()
