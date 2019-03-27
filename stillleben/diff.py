
"""
Differentiation package for stillleben
"""

def backpropagate_gradient_to_poses(scene, render_result, grad):
    r"""
    Performs backpropagation of a gradient on the output image back to the
    object poses used to render the scene.

    Note that the orientation part of each pose is locally linearized, i.e.

    .. math::

        T(\alpha,\beta,\gamma,a,b,c) = \left( \begin{matrix}
            1 & -\gamma & \beta & a \\
            \gamma & 1 & -\alpha & b \\
            -\beta & \alpha & 1 & c \\
            0 & 0 & 0 & 1 \\
        \end{matrix} \right) T_0

    Args:
        scene (stillleben.Scene): Scene
        render_result (stillleben.RenderPassResult): Rendered frame
        grad (tensor): 3xHxW float tensor with gradient of the objective
            w.r.t. the image.
    Returns:
        tensor: Nx6 float gradient of the objective w.r.t. the N poses
        (see above).
    """
    pass

def apply_pose_delta(pose, delta, orthonormalize=True):
    r"""
    Applies a pose delta in the form of :math:`(\alpha,\beta,\gamma,a,b,c)` to
    a 4x4 pose matrix.

    See :func:`backpropagate_gradient_to_poses` for the definition of ``delta``.

    Args:
        pose (tensor): 4x4 pose matrix. May also be batched (Bx4x4).
        delta (tensor): 6-dim delta vector. May also be batched (Bx6).
        orthonormalize (bool): If true (default), perform an SVD for
            orthonormalization of the rotation matrix after the update.

    Returns:
        tensor: New 4x4 pose matrix. If the inputs are batched, this one is as
        well.
    """

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

    new_poses = torch.matmul(delta_matrix, pose)

    # and make sure it's orthonormal
    if orthonormalize:
        for b in range(B):
            U, S, V = torch.svd(new_poses[b, :3, :3])
            new_poses[b, :3, :3] = torch.matmul(U, V.t())

    # unbatch if required
    if not batch:
        new_poses = new_poses[0]

    # back to original device
    new_poses = new_poses.to(device)

    return new_poses
