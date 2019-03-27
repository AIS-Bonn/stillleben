
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
        render_result (stillleben.RenderResult): Rendered frame
        grad (tensor): 3xHxW float tensor with gradient of the objective
        w.r.t. the image.
    Returns:
        tensor: Nx6 float gradient of the objective w.r.t. the N poses
        (see above).
    """
