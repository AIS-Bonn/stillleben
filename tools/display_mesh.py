
import torch
import stillleben as sl

from stillleben import camera_model

from PIL import Image

import time

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='display a mesh')
    parser.add_argument('--background', metavar='PATH', type=str,
        help='Background image')
    parser.add_argument('--noise', action='store_true',
        help='Apply noise model')
    parser.add_argument('mesh', metavar='PATH', nargs='+', type=str,
        help='Mesh file to display')
    parser.add_argument('--debug', action='store_true',
        help='Render debug image with object poses')
    parser.add_argument('--physics-debug', action='store_true',
        help='Render physics debug image with collision wireframes')
    parser.add_argument('--normals', action='store_true',
        help='Display normals')
    parser.add_argument('--tabletop', action='store_true',
        help='Simulate a tabletop scene')
    parser.add_argument('--normalize', action='store_true',
        help='Normalize mesh scales')
    parser.add_argument('--shading', type=str, default='phong',
        choices=['phong', 'flat'],
        help='Shading type (phong/flat)')

    args = parser.parse_args()

    sl.init()

    scene = sl.Scene((640,480))

    if args.background:
        scene.background_image = sl.Texture(args.background)

    print('Loading meshes (this can take some time)...')
    meshes = sl.Mesh.load_threaded(args.mesh)
    for mesh in meshes:
        print("Loaded mesh with bounding box:", mesh.bbox)
        print("center:", mesh.bbox.center, "size:", mesh.bbox.size)

        if args.normalize:
            mesh.center_bbox()
            mesh.scale_to_bbox_diagonal(0.2, 'exact')

            print("normalized:", mesh.bbox)
            print("center:", mesh.bbox.center, "size:", mesh.bbox.size, "diagonal:", mesh.bbox.diagonal)

    print("Meshes loaded.")

    for mesh in meshes:
        object = sl.Object(mesh)
        scene.add_object(object)

        if not args.tabletop:
            if True:
                if not scene.find_noncolliding_pose(object, sampler='random', max_iterations=50, viewpoint=torch.tensor([1.0, 0.0, 0.0])):
                    print('WARNING: Could not find non-colliding pose')
            elif True:
                pose = scene.place_object_randomly(mesh.bbox.diagonal)
                object.set_pose(pose)
            else:
                pose = torch.eye(4)
                pose[2,3] = scene.min_dist_for_object_diameter(mesh.bbox.diagonal)
                object.set_pose(pose)

    renderer = sl.RenderPass(shading=args.shading)

    def vis_cb(iteration):
        result = renderer.render(scene)
        rgb = result.rgb()
        rgb = rgb[:,:,:3]
        rgb_np = rgb.cpu().numpy()

        img = Image.fromarray(rgb_np, mode='RGB')
        img.save('/tmp/iter{:03}.png'.format(iteration))

        #dbg_rgb = sl.render_physics_debug_image(scene)
        #dbg_img = Image.fromarray(dbg_rgb.cpu().numpy()[:,:,:3], mode='RGB')
        #dbg_img.save('/tmp/physics{:03}.png'.format(iteration))

    if args.tabletop:
        s1 = time.time()
        scene.simulate_tabletop_scene()
        s2 = time.time()
        print('Tabletop sim took {}s'.format(s2-s1))

    print('Resulting poses:')
    for obj in scene.objects:
        print(obj.pose())

    result = renderer.render(scene)

    rgb = result.rgb()
    rgb = rgb[:,:,:3]

    if args.noise:
        rgb_float = rgb.permute(2,0,1).float() / 255.0
        rgb_float = camera_model.process_image(rgb_float)
        rgb = (rgb_float * 255.0).byte().permute(1,2,0)

    print(rgb.size())
    rgb_np = rgb.cpu().numpy()

    img = Image.fromarray(rgb_np, mode='RGB')
    img.show()

    if args.debug:
        dbg_rgb = sl.render_debug_image(scene)
        dbg_img = Image.fromarray(dbg_rgb.cpu().numpy()[:,:,:3], mode='RGB')
        dbg_img.show()

    if args.physics_debug:
        dbg_rgb = sl.render_physics_debug_image(scene)
        dbg_img = Image.fromarray(dbg_rgb.cpu().numpy()[:,:,:3], mode='RGB')
        dbg_img.show()

    if args.normals:
        normal_img = result.normals()[:,:,2] * 128 + 128
        normal_img.clamp_(0,255)
        normal_img = Image.fromarray(normal_img.byte().cpu().numpy())
        normal_img.show()

        normal_img = result.normals()[:,:,3] * 255
        normal_img.clamp_(0,255)
        normal_img = Image.fromarray(normal_img.byte().cpu().numpy())
        normal_img.show()
