
import torch
import stillleben as sl

from stillleben import camera_model

from PIL import Image

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

    args = parser.parse_args()

    sl.init()

    scene = sl.Scene((640,480))

    if args.background:
        scene.background_image = sl.Texture(args.background)

    meshes = []
    for mesh_file in args.mesh:
        mesh = sl.Mesh(mesh_file, max_physics_triangles=6000)

        print("Loaded mesh with bounding box:", mesh.bbox)
        print("center:", mesh.bbox.center, "size:", mesh.bbox.size)

        #mesh.center_bbox()
        #mesh.scale_to_bbox_diagonal(0.5, 'order_of_magnitude')

        print("normalized:", mesh.bbox)
        print("center:", mesh.bbox.center, "size:", mesh.bbox.size, "diagonal:", mesh.bbox.diagonal)
        meshes.append(mesh)

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

    renderer = sl.RenderPass()

    def vis_cb(iteration):
        result = renderer.render(scene)
        rgb = result.rgb()
        rgb = rgb[:,:,:3]
        rgb_np = rgb.cpu().numpy()

        img = Image.fromarray(rgb_np, mode='RGB')
        img.save('/tmp/iter{:03}.png'.format(iteration))

        dbg_rgb = sl.render_physics_debug_image(scene)
        dbg_img = Image.fromarray(dbg_rgb.cpu().numpy()[:,:,:3], mode='RGB')
        dbg_img.save('/tmp/physics{:03}.png'.format(iteration))

    if args.tabletop:
        scene.simulate_tabletop_scene(vis_cb)

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
