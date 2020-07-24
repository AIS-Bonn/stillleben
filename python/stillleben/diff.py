
"""
Differentiation package for stillleben

.. note-warning:: CUDA required!

    At the moment, the :ref:`diff` module is only compiled when CUDA is
    available. Since CPU implementations are available, this restriction
    could be lifted in the future.

Author: Arul Periyasamy <arul.periyasamy@ais.uni-bonn.de>
"""

import torch
import torch.nn.functional as F
import os
import numpy as np
import types

from .profiling import Timer

from .lib.libstillleben_python import Scene, RenderPassResult

DIFF_AVAILABLE = False
try:
    from .lib.libstillleben_diff_python import generate_sobel_valid_mask
    from .lib.libstillleben_diff_python import dilate_object_mask
    DIFF_AVAILABLE = True
except ImportError as e:
    DIFF_ERROR = str(e)

def _check():
    if not DIFF_AVAILABLE:
        import sys
        raise NotImplementedError(
            'Could not import libstillleben_diff_python, maybe stillleben ' +
            'was built without CUDA support?\n' +
            'original exception follows:\n' + DIFF_ERROR)

__all__ = [
    'compute_image_space_gradients',
    'backpropagate_gradient_to_poses',
    'apply_pose_delta',
]

TH_GAUSSIAN_KERNEL = 0
KS = 0

def gaussian_kernel(l=5, sig=1.):
    """
    1D Gaussian kernel with side length l and sigma sig.
    Source: https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)

    kernel = np.exp(-0.5 * np.square(ax) / np.square(sig))

    return kernel / np.sum(kernel)

def _init_diff():
    global TH_GAUSSIAN_KERNEL
    global KS

    KS = 11
    SIGMA = 1
    o = np.tile(gaussian_kernel(KS, SIGMA).reshape(KS, 1), (1, KS))
    kernel  = o * o.T
    TH_GAUSSIAN_KERNEL = torch.from_numpy(kernel)
    TH_GAUSSIAN_KERNEL = TH_GAUSSIAN_KERNEL.view(1,1, KS, KS).float()
_init_diff()

def compute_image_space_gradients(scene : Scene, render_result : RenderPassResult):
    """
    Gradient of intensity w.r.t. 2D pixel positions.

    :param scene: stillleben scene
    :param render_result: Render result from :ref:`RenderPass`
    :return: (grad_x, grad_y, valid), where grad_x and grad_y are gradients
        w.r.t. X and Y pixel position for each pixel (shape HxWxC).

    Basically, this answers the question of how the image will change when we
    move pixels in 2D.
    """

    rgb = render_result.rgb()
    rgb = rgb[:,:,:3]

    device = rgb.device

    # convert to PyTorch CxHxW
    rgb_float = rgb.permute(2,0,1).float() / 255.0

    # grad_objective_wrt_poses = torch.zeros(len(scene.objects), 6)
    _c, _h, _w = rgb_float.shape

    # HXWxC
    with Timer('sobel'):
        coordinates = render_result.coordinates()

        sobel_x = torch.zeros(1, 1, 1, 3).to(device)
        sobel_x[:,:,:,0] = -1
        sobel_x[:,:,:,1] =  0
        sobel_x[:,:,:,2] =  1
        sobel_x /= 2 / _w * 2

        sobel_y = torch.zeros(1, 1, 3, 1).to(device)
        sobel_y[:,:,0] = -1
        sobel_y[:,:,1] =  0
        sobel_y[:,:,2] =  1
        sobel_y /= 2.0 / _h * 2

        rgb_float = rgb_float.view(3, 1, _h, _w)
        grad_x = -F.conv2d(rgb_float, sobel_x, padding=(0,1)).squeeze()
        grad_y = -F.conv2d(rgb_float, sobel_y, padding=(1,0)).squeeze()

    with Timer('sobel_valid_mask'):
        depth = render_result.depth() # HxW

        #import ipdb; ipdb.set_trace()
        sobel_valid_mask = generate_sobel_valid_mask(render_result.instance_index().squeeze(),  depth.squeeze())

        # disable gradients for occluded objects
        grad_x[:,~sobel_valid_mask] = 0
        grad_y[:,~sobel_valid_mask] = 0

    return grad_x, grad_y, sobel_valid_mask


def soft_forward(scene, render_result, obs_rgb, loss_fn):
    '''
    NOTE: render_result are expected to be in depth peeling order
    i.e. Oth element has the original renderered scene without any depth peeling
    '''

    _check()

    if not (type(render_result) is list or type(render_result) is tuple):
        raise ValueError("render_result should be a list or tuple")

    if len(obs_rgb.shape) != 3:
        raise ValueError("Observed RGB should have 3 dimension CxHxW")
    if obs_rgb.shape[0] != 3:
        raise ValueError("Observed RGB should of format CxHxW with C=3")
    if obs_rgb.dtype != torch.float32:
        raise ValueError("Observed RGB should be of type torch.float32")
    if obs_rgb.max().item() > 1:
        raise ValueError("Observed RGB should have range [0,1]")
    if not isinstance(loss_fn, types.FunctionType):
        raise ValueError("loss_fn should be a callable function")

    for rr in render_result:
        if not hasattr(rr, 'is_extracted'):
            raise ValueError("Entries in the render_results should be extracted")
        if rr.rgb().shape != render_result[0].rgb().shape:
            raise ValueError("render_results should correspond to the same scene")

    global TH_GAUSSIAN_KERNEL
    device = render_result[0].rgb().device
    rgb_list = []

    for rr in render_result:
        rgb = rr.rgb()
        rgb = rgb[:,:,:3].permute(2, 0, 1)
        rgb = (rgb.float() / 255.)

        rgb_list.append(rgb)
    rgbs = torch.stack(rgb_list).detach() # DBxCxHxW
    rgbs.requires_grad = True

    DB, C, H, W = rgbs.shape
    DB_WEIGHTS = torch.tensor([0.7, 0.3, 0.1, 0.1, 0.05]).float().to(device)

    soft_rgb = torch.zeros(3, H, W).to(device)

    for db in range(DB):
        soft_rgb[0] += rgbs[db][0] * DB_WEIGHTS[db]
        soft_rgb[1] += rgbs[db][1] * DB_WEIGHTS[db]
        soft_rgb[2] += rgbs[db][2] * DB_WEIGHTS[db]

    if TH_GAUSSIAN_KERNEL.device != device:
        TH_GAUSSIAN_KERNEL = TH_GAUSSIAN_KERNEL.to(device)
    soft_rgb = soft_rgb.unsqueeze(dim=1)

    TH_GAUSSIAN_KERNEL.requires_grad = True
    soft_rgb_gaussian = F.conv2d(soft_rgb, TH_GAUSSIAN_KERNEL, padding=KS//2)
    soft_rgb_gaussian = soft_rgb_gaussian.squeeze()

    _c, _h, _w = soft_rgb_gaussian.shape
    gt_mask = torch.ones(_h, _w).float().to(device)

    loss, loss_img = loss_fn(soft_rgb_gaussian.unsqueeze(0), obs_rgb.unsqueeze(0))
    loss.backward()

    grad_objective_wrt_rnd_imgs = rgbs.grad.clone() # DBxCxHxW

    vertex_index_2_bp = []
    grad_vertices_2_bp = []
    grad_colors_2_bp = []

    # backpropagate the gradients for each peel layer
    for ir, rr in enumerate(render_result):
        _vertex_index_2_bp, _grad_vertices_2_bp, _grad_color_2_bp = bp_to_vertices_and_colors(scene, rr, grad_objective_wrt_rnd_imgs[ir])

        vertex_index_2_bp += _vertex_index_2_bp
        grad_vertices_2_bp += _grad_vertices_2_bp
        grad_colors_2_bp += _grad_color_2_bp

    rgbs_return = []
    for rgb in rgbs:
        rgbs_return.append(rgb.clone().detach().squeeze())

    return soft_rgb.clone().detach().squeeze(), rgbs_return, loss_img, loss.item(), vertex_index_2_bp, grad_vertices_2_bp, grad_colors_2_bp

def bp_to_vertices_and_colors(scene, render_result, grad_objective_wrt_rnd_img, visualize_grad=False):
    r"""
    Performs backpropagation of a gradient on the output image back to the
    vertices of the meshes used to render the scene.
    Args:
        scene (stillleben.Scene): Scene
        render_result (stillleben.RenderPassResult): Rendered frame
        grad_objective_wrt_rnd_img (tensor): 3xHxW float tensor with gradient of the objective
            w.r.t. the image.
    Returns:
        TODO:
    """

    _check()

    rgb = render_result.rgb()
    rgb = rgb[:,:,:3]

    device = rgb.device

    # convert to PyTorch CxHxW
    rgb_float = rgb.permute(2,0,1).float() / 255.0

    grad_objective_wrt_poses = torch.zeros(len(scene.objects), 6)
    _c, _h, _w = rgb_float.shape

    coordinates = render_result.coordinates()

    # get screen_space_gradients and valid sobel mask.
    grad_x, grad_y, sobel_valid_mask = compute_image_space_gradients(scene, render_result)
    grad_wrt_xy = torch.stack([grad_x, grad_y], dim=0)

    grad_img_wrt_2D = torch.stack([grad_x, grad_y], dim=0) # [2x3xHxW]

    vertex_index_2_bp = []
    grad_vertices_2_bp = []
    grad_colors_2_bp = []
    for _idx, obj in enumerate(scene.objects):
        with Timer('object'):
            projection_matrix = scene.projection_matrix()
            object_mask = (render_result.instance_index() == obj.instance_index)
            object_mask_squeeze = object_mask.squeeze()

            _shape = object_mask_squeeze.shape

            object_mask_pixel_flat = object_mask.view(-1)

            bcfs= render_result.barycentric_coeffs()
            bcfs = bcfs.permute(2,0,1)[:3] 
            bcfs_pixel_flat = bcfs.permute(1,2,0).view(-1,3)

            vertex_index = render_result.vertex_indices().permute(2,0,1)[:3]
            vertex_index_pixel_flat = vertex_index.permute(1,2,0).view(-1,3)

            if Timer.enabled:
                # Take care that we don't count input synchronization in
                # the dilate block below
                with Timer('sync'):
                    torch.cuda.synchronize()

            with Timer('coordinates'):
                # HXWxC-> CxHXW
                obj_coordinates = coordinates.permute(2,0,1)

                # convert to homogeneous coordinates
                obj_coordinates = torch.cat([obj_coordinates,  torch.ones(1, _h, _w).to(device) ], dim=0)
                # CxHXW -> Cx(H*W)
                coordinates_col_vec = obj_coordinates.view(4, -1)

                obj_coordinates = coordinates_col_vec[:, object_mask_pixel_flat]
                if obj_coordinates.shape[1] == 0:
                    print ('instance_index image for the current object is empty')
                    print ('object not rendered as a part of the scene')
                    continue

                obj_pose = obj.pose().to(device)
                projection_matrix  = projection_matrix.to(device)

                #[3x4] <= [3x4] @ [4x4]
                P = projection_matrix @ obj_pose

                P_X3D_0 = (P[0:1] @ obj_coordinates)
                P_X3D_1 = (P[1:2] @ obj_coordinates)
                P_X3D_2 = (P[2:3] @ obj_coordinates)
                denominator = (P_X3D_2) ** 2

                _grad_x_wrt_X = (P_X3D_2 * P[0,0] - P_X3D_0 * P[2,0]) / denominator
                _grad_x_wrt_Y = (P_X3D_2 * P[0,1] - P_X3D_0 * P[2,1]) / denominator
                _grad_x_wrt_Z = (P_X3D_2 * P[0,2] - P_X3D_0 * P[2,2]) / denominator

                _grad_y_wrt_X = (P_X3D_2 * P[1,0] - P_X3D_1 * P[2,0]) / denominator
                _grad_y_wrt_Y = (P_X3D_2 * P[1,1] - P_X3D_1 * P[2,1]) / denominator
                _grad_y_wrt_Z = (P_X3D_2 * P[1,2] - P_X3D_1 * P[2,2]) / denominator

                grad_x_wrt_3D = torch.cat([_grad_x_wrt_X, _grad_x_wrt_Y, _grad_x_wrt_Z])
                grad_y_wrt_3D = torch.cat([_grad_y_wrt_X, _grad_y_wrt_Y, _grad_y_wrt_Z])

            with Timer('reshape'):
                grad_loss_wrt_img = grad_objective_wrt_rnd_img # [3xHxW]

                # P --> no.of pixels belonging to object
                grad_2D_wrt_3D = torch.stack([grad_x_wrt_3D, grad_y_wrt_3D], dim=0) # [2x3xP]

                grad_loss_wrt_img = grad_loss_wrt_img[:, object_mask_squeeze] # [3xHxW] -> [3xP]

                grad_loss_wrt_img = grad_loss_wrt_img.permute(1,0) # [3xP] -> [Px3]
                grad_loss_wrt_img = grad_loss_wrt_img.unsqueeze(dim=1) # [Px3] -> [Px1x3]

                _grad_img_wrt_2D_obj = grad_img_wrt_2D[:, :, object_mask_squeeze].view(2, 3, -1) # [2x3xHxW] -> [2x3xP]
                _grad_img_wrt_2D_obj = _grad_img_wrt_2D_obj.permute(2, 1, 0) # [2x3xP] -> [Px3x2]

                grad_2D_wrt_3D = grad_2D_wrt_3D.permute(2, 0, 1) # [2x3xP] -> [Px2x3]

            with Timer('bmm'):
                grad_img_wrt_3D = torch.bmm(_grad_img_wrt_2D_obj,  grad_2D_wrt_3D) # [Px3x3]
                grad_loss_wrt_3D = torch.bmm(grad_loss_wrt_img,  grad_img_wrt_3D) #  [Px1x3] @ [Px3x3] -> [Px1x3]

            with Timer('backpropagate'):
                bcfs_pixel_batch = bcfs_pixel_flat[object_mask_pixel_flat].unsqueeze(dim=-1) # [Px3] -> [Px3x1]
                grad_loss_wrt_3D_bcfs_weighted = torch.bmm(bcfs_pixel_batch, grad_loss_wrt_3D)
                _grad_loss_wrt_color = torch.bmm(bcfs_pixel_batch, grad_loss_wrt_img)

                _grad_vertices_2_bp = grad_loss_wrt_3D_bcfs_weighted
                _vertex_index_2_bp = vertex_index_pixel_flat[ object_mask_pixel_flat ]

                _grad_vertices_2_bp = _grad_vertices_2_bp.view(-1, 3)
                _vertex_index_2_bp =  _vertex_index_2_bp.view(-1)
                _grad_loss_wrt_color =  _grad_loss_wrt_color.view(-1, 3)

                # take small step in the opposite direction
                _grad_vertices_2_bp *= -1
                _grad_loss_wrt_color *= -1

                # obj.mesh.update_vertices(_vertex_index_2_bp.cpu(),  _grad_2_bp.squeeze().cpu())
                vertex_index_2_bp.append( _vertex_index_2_bp.detach().clone() )
                grad_vertices_2_bp.append( _grad_vertices_2_bp.detach().clone() )
                grad_colors_2_bp.append( _grad_loss_wrt_color.detach().clone() )
    return vertex_index_2_bp, grad_vertices_2_bp, grad_colors_2_bp


def backpropagate_gradient_to_poses(scene : Scene, render_result : RenderPassResult, grad_objective_wrt_rnd_img : torch.Tensor, visualize_grad=False):
    r"""
    Performs backpropagation of a gradient on the output image back to the
    object poses used to render the scene.

    :param scene: Scene
    :param render_result: Rendered frame
    :param grad_objective_wrt_rnd_img: 3xHxW float tensor with gradient of the
        objective w.r.t. the image.
    :param visualize_grad: Display grad visualization using visdom
    :return: Nx6 float gradient of the objective w.r.t. the N poses.

    Note that the orientation part of each pose is locally linearized, i.e.

    .. math::

        T(\alpha,\beta,\gamma,a,b,c) = T_0 \left( \begin{matrix}
            1 & -\gamma & \beta & a \\
            \gamma & 1 & -\alpha & b \\
            -\beta & \alpha & 1 & c \\
            0 & 0 & 0 & 1 \\
        \end{matrix} \right)
    """

    _check()

    rgb = render_result.rgb()
    rgb = rgb[:,:,:3]

    device = rgb.device

    # convert to PyTorch CxHxW
    rgb_float = rgb.permute(2,0,1).float() / 255.0

    grad_objective_wrt_poses = torch.zeros(len(scene.objects), 6)
    _c, _h, _w = rgb_float.shape

    coordinates = render_result.coordinates()

    # get screen_space_gradients and valid sobel mask.
    grad_x, grad_y, sobel_valid_mask = compute_image_space_gradients(scene, render_result)
    grad_wrt_xy = torch.stack([grad_x, grad_y], dim=0)

    for _idx, obj in enumerate(scene.objects):
        with Timer('object'):
            projection_matrix = scene.projection_matrix()
            object_mask = render_result.instance_index() == obj.instance_index

            if Timer.enabled:
                # Take care that we don't count input synchronization in
                # the dilate block below
                with Timer('sync'):
                    torch.cuda.synchronize()

            with Timer('dilate'):
                # dilate object mask
                object_mask, obj_coordinates = dilate_object_mask(object_mask.squeeze(), sobel_valid_mask, coordinates)
                if object_mask.sum() ==0:
                    print ('object_mask is empty')
                    print ('This could happen if the object is out of field of view')
                    continue

            with Timer('coordinates'):
                # HXWxC-> CxHXW
                obj_coordinates = obj_coordinates.permute(2,0,1)

                # convert to homogeneous coordinates
                obj_coordinates = torch.cat([obj_coordinates,  torch.ones(1, _h, _w).to(device) ], dim=0)
                # CxHXW -> Cx(H*W)
                coordinates_x = obj_coordinates.view(4, -1)

                obj_coordinates = coordinates_x[:, object_mask.view(-1)]
                if obj_coordinates.shape[1] == 0:
                    print ('instance_index image for the current object is empty')
                    print ('object not rendered as a part of the scene')
                    continue

                # apply object pose
                y = obj.pose().to(device) @ obj_coordinates

            with Timer('grad_wrt_coordinates'):
                grad_wrt_coordinates = torch.zeros(2, 3, _h, _w).to(device)
                P = projection_matrix.to(device)
                for _j in range(2):
                    for _i in range(3):
                        grad_1 = ( P[_j,_i] * (1/ (P[2:3] @ y)) )
                        grad_2 =  (  P[2,_i] * (-1 / ((P[2:3] @ y) **2)) ) * (P[_j:_j+1] @ y)
                        grad = grad_1 + grad_2
                        grad_wrt_coordinates[_j][_i].view(-1)[object_mask.view(-1)] = grad.view(-1)

            with Timer('diff_grad'):
                x = coordinates_x[:, object_mask.view(-1)]
                T_0 = obj.pose().to(device)

                diff_alpha = torch.zeros(4,4).to(device)
                diff_alpha[1,2] = -1
                diff_alpha[2,1] =  1
                diff_beta = torch.zeros(4,4).to(device)
                diff_beta[0,2] =  1
                diff_beta[2,0] = -1
                diff_gamma = torch.zeros(4,4).to(device)
                diff_gamma[0,1] = -1
                diff_gamma[1,0] =  1
                diff_a = torch.zeros(4,4).to(device)
                diff_a[0,3] = 1
                diff_b = torch.zeros(4,4).to(device)
                diff_b[1,3] = 1
                diff_c = torch.zeros(4,4).to(device)
                diff_c[2,3] = 1
                grad_alpha = T_0 @ diff_alpha @ x
                grad_beta  = T_0 @ diff_beta @ x
                grad_gamma = T_0 @ diff_gamma @ x
                grad_a     = T_0 @ diff_a @ x
                grad_b     = T_0 @ diff_b @ x
                grad_c     = T_0 @ diff_c @ x

                # _grad_*: 4 x N
                # we want: 6 x 3 x N

                # discard homogenous dimension
                grad_alpha = grad_alpha[:3]
                grad_beta = grad_beta[:3]
                grad_gamma = grad_gamma[:3]
                grad_a = grad_a[:3]
                grad_b = grad_b[:3]
                grad_c = grad_c[:3]

                # stack
                grad_wrt_pose = torch.stack([grad_alpha, grad_beta, grad_gamma, grad_a, grad_b, grad_c], dim=0)

            with Timer('grad_xy_pose'):
                # grad_xy_coordinates = grad_wrt_xy * grad_wrt_coordinates
                g_xy = grad_wrt_xy.view(2, 3, -1)[:,:,object_mask.view(-1)] # 2x3xN
                g_coord = grad_wrt_coordinates.view(2, 3, -1)[:,:,object_mask.view(-1)] # [2x3xN]
                g_pose = grad_wrt_pose # [6x3xN]

                g_xy = g_xy.permute(2,1,0) # [2x3xN] -> [Nx3x2]
                g_coord = g_coord.permute(2,0,1) # [2x3xN] -> [Nx2x3]
                g_pose  = g_pose.permute(2,1,0) # [6x3xN] -> [Nx3x6]

                grad_xy_pose = torch.bmm(torch.bmm(g_xy, g_coord), g_pose) # [ [Nx3x2]@[Nx2x3] ]  @ [Nx3x6] -> [Nx3x6]

            # this is input parameter
            grad_in = grad_objective_wrt_rnd_img.view(3,-1)[:,object_mask.view(-1)]

            _grad_xy_pose = grad_xy_pose # [Nx3x6]
            _grad_in =  grad_in.permute(1, 0) # [3xN] -> [Nx3]
            _grad_in = _grad_in.unsqueeze(dim=1) # [Nx3] -> [Nx1x3]

            with Timer('bmm'):
                grad = torch.bmm(_grad_in, _grad_xy_pose) # [N, 1, 6]

            if visualize_grad:
                grad_tmp = grad.view(grad_in.shape[1], grad_in.shape[2], 6).cpu()

                _grad = []
                for _x in range(6):
                    _t = grad_tmp[:,:,_x].unsqueeze(0)
                    _t = (_t - _t.min()) / (_t.max() - _t.min() )
                    _grad.append(_t)
                _grad_vis = torch.cat(_grad)
                # _grad_vis = (_grad_vis - _grad_vis.min()) /(_grad_vis.max() - _grad_vis.min())
                vis.images(_grad_vis.unsqueeze(1).numpy(), nrow=3)

            grad_bmm = grad.sum(dim=0)

            grad_objective_wrt_poses[_idx] = grad_bmm

    return grad_objective_wrt_poses

def apply_pose_delta(pose : torch.Tensor, delta : torch.Tensor, orthonormalize=True):
    r"""
    Applies a pose delta in the form of :math:`(\alpha,\beta,\gamma,a,b,c)` to
    a 4x4 pose matrix.

    :param pose: 4x4 pose matrix. May also be batched (Bx4x4).
    :param delta: 6-dim delta vector. May also be batched (Bx6).
    :param orthonormalize: If true (default), perform an SVD for
        orthonormalization of the rotation matrix after the update.
    :return: New 4x4 pose matrix. If the inputs are batched, this one is as
        well.

    See :ref:`backpropagate_gradient_to_poses` for the definition of :p:`delta`.
    """

    _check()

    if pose.dim() == 3:
        assert delta.dim() == 2
        batched = True
    else:
        assert delta.dim() == 1
        batched = False
        pose = pose.unsqueeze(0)
        delta = delta.unsqueeze(0)

    device = pose.device

    pose = pose.cpu()
    delta = delta.cpu()

    B = pose.size(0)

    delta_matrix = torch.zeros(B, 4, 4, device=pose.device)

    delta_matrix[:,0,0] = 1.0
    delta_matrix[:,0,1] = -delta[:,2]
    delta_matrix[:,0,2] = delta[:,1]

    delta_matrix[:,1,0] = delta[:,2]
    delta_matrix[:,1,1] = 1.0
    delta_matrix[:,1,2] = -delta[:,0]

    delta_matrix[:,2,0] = -delta[:,1]
    delta_matrix[:,2,1] = delta[:,0]
    delta_matrix[:,2,2] = 1.0

    delta_matrix[:,:3,3] = delta[:,3:]
    delta_matrix[:,3,3] = 1.0

    new_poses = torch.matmul(pose, delta_matrix)

    # and make sure it's orthonormal
    if orthonormalize:
        for b in range(B):
            U, S, V = torch.svd(new_poses[b, :3, :3])
            new_poses[b, :3, :3] = torch.matmul(U, V.t())

    # unbatch if required
    if not batched:
        new_poses = new_poses[0]

    # back to original device
    new_poses = new_poses.to(device)

    return new_poses
