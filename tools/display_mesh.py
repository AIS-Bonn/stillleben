
import torch
import stillleben as sl

from PIL import Image

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='display a mesh')
    parser.add_argument('mesh', metavar='PATH', type=str,
                    help='Mesh file to display')

    args = parser.parse_args()

    sl.init()

    scene = sl.Scene((640,480))

    mesh = sl.Mesh(args.mesh)
    mesh.center_bbox()
    mesh.scale_to_bbox_diagonal(0.5)

    object = sl.Object(mesh)
    scene.add_object(object)

    pose = torch.eye(4)
    pose[2,3] = scene.min_dist_for_object_diameter(0.5)
    object.set_pose(pose)

    renderer = sl.RenderPass()
    result = renderer.render(scene)

    rgb = result.rgb()
    rgb_np = rgb.cpu().numpy()

    img = Image.fromarray(rgb_np, mode='RGBA')
    img.show()

