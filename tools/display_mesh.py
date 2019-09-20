
import torch
import stillleben as sl

from stillleben import camera_model

from PIL import Image

import time

import math
from math import sin, cos

def rotx(theta):
    """
    Rotation about X-axis

    @type theta: number
    @param theta: the rotation angle
    @rtype: 3x3 orthonormal matrix
    @return: rotation about X-axis
    @see: L{roty}, L{rotz}, L{rotvec}
    """

    ct = cos(theta)
    st = sin(theta)
    return torch.tensor([[1,  0,    0],
            [0,  ct, -st],
            [0,  st,  ct]])

def roty(theta):
    """
    Rotation about Y-axis

    @type theta: number
    @param theta: the rotation angle
    @rtype: 3x3 orthonormal matrix
    @return: rotation about Y-axis
    @see: L{rotx}, L{rotz}, L{rotvec}
    """

    ct = cos(theta)
    st = sin(theta)

    return torch.tensor([[ct,   0,   st],
            [0,    1,    0],
            [-st,  0,   ct]])

def rotz(theta):
    """
    Rotation about Z-axis

    @type theta: number
    @param theta: the rotation angle
    @rtype: 3x3 orthonormal matrix
    @return: rotation about Z-axis
    @see: L{rotx}, L{roty}, L{rotvec}
    """

    ct = cos(theta)
    st = sin(theta)

    return torch.tensor([[ct,      -st,  0],
            [st,       ct,  0],
            [ 0, 0, 1]])

def rpy2r(roll, pitch=None,yaw=None):
    """
    Rotation from RPY angles.

    Two call forms:
        - R = rpy2r(S{theta}, S{phi}, S{psi})
        - R = rpy2r([S{theta}, S{phi}, S{psi}])
    These correspond to rotations about the Z, Y, X axes respectively.
    @type roll: number or list/array/matrix of angles
    @param roll: roll angle, or a list/array/matrix of angles
    @type pitch: number
    @param pitch: pitch angle
    @type yaw: number
    @param yaw: yaw angle
    @rtype: 4x4 homogenous matrix
    @return: R([S{theta} S{phi} S{psi}])
    @see:  L{tr2rpy}, L{rpy2r}, L{tr2eul}
    """
    r = rotz(roll) @ roty(pitch) @ rotx(yaw)
    return r

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
    parser.add_argument('--placement', type=str, choices=['center', 'random', 'tabletop', 'center_random'],
        default='center',
        help='Object placement')
    parser.add_argument('--tabletop', action='store_true',
        help='Simulate a tabletop scene')
    parser.add_argument('--normalize', action='store_true',
        help='Normalize mesh scales')
    parser.add_argument('--shading', type=str, default='phong',
        choices=['phong', 'flat'],
        help='Shading type (phong/flat)')
    parser.add_argument('--background-color', type=str, default='1,1,1,1',
        help='Background color (R,G,B,A)')
    parser.add_argument('--show-depth', action='store_true',
        help='Show depth image as well')
    parser.add_argument('--force-color', type=str)
    parser.add_argument('--serialize', action='store_true',
        help='Display serialized scene string')
    parser.add_argument('--ambient', metavar='COLOR', type=str,
        help='Set ambient scene color')

    parser.add_argument('--rpy', type=str,
        help='RPY rotation in degrees')

    parser.add_argument('--out', type=str,
        help='Save output in file')

    parser.add_argument('--size', type=str,
        help='Image size (WxH)', default="640x480")

    parser.add_argument('--shininess', type=float,
        help='Phong shininess parameter', default=80.0)
    parser.add_argument('--roughness', type=float,
        help='Roughness parameter')
    parser.add_argument('--metalness', type=float,
        help='Metalness parameter')
    parser.add_argument('--specular-color', type=str,
        help='Specular color (R,G,B)')

    parser.add_argument('--light-map', type=str, metavar='FILE.ibl',
        help='Use image-based lighting')

    parser.add_argument('--sticker-range', type=str, default='0,0,0,0',
        help='Sticker range')
    parser.add_argument('--sticker-texture', type=str,
        help='Sticker texture file')

    parser.add_argument('--background-plane-size', type=str, default='0,0',
        help='Background plane size')
    parser.add_argument('--background-plane-texture', type=str,
        help='Background plane texture')

    parser.add_argument('--tabletop-video', type=str,
        help='Write tabletop video to this location')

    args = parser.parse_args()

    sl.init()

    scene = sl.Scene([ int(d) for d in args.size.split('x')])

    if args.background:
        scene.background_image = sl.Texture(args.background)

    scene.background_color = torch.tensor([ float(a) for a in args.background_color.split(',') ])

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

    opts = {}

    if args.force_color:
        opts['color'] = torch.tensor([ float(a) for a in args.force_color.split(',') ])
        opts['force_color'] = True

    if args.ambient:
        scene.ambient_light = torch.tensor([ float(a) for a in args.ambient.split(',') ])

    if args.light_map:
        scene.light_map = sl.LightMap(args.light_map)

    scene.background_plane_size = torch.tensor([ float(a) for a in args.background_plane_size.split(',') ])
    if args.background_plane_texture:
        scene.background_plane_texture = sl.Texture2D(args.background_plane_texture)

    for mesh in meshes:
        object = sl.Object(mesh, options=opts)
        object.shininess = args.shininess

        if args.metalness:
            object.metalness = args.metalness
        if args.roughness:
            object.roughness = args.roughness

        if args.specular_color:
            object.specular_color = torch.tensor([ float(a) for a in args.specular_color.split(',') ])

        object.sticker_range = torch.tensor([ float(a) for a in args.sticker_range.split(',')])

        if args.sticker_texture:
            object.sticker_texture = sl.Texture(args.sticker_texture)

        scene.add_object(object)

        if args.placement == 'center' or args.placement == 'center_random':
            pose = torch.eye(4)

            if args.rpy:
                rpy = [ math.pi / 180.0 * float(a) for a in args.rpy.split(',') ]
                pose[:3,:3] = rpy2r(rpy[0], rpy[1], rpy[2])
                print(pose[:3,:3])

            if args.placement == 'center_random':
                q = torch.randn(4)
                q /= q.norm()

                pose[:3,:3] = sl.quat_to_matrix(q)

            pose[2,3] = scene.min_dist_for_object_diameter(mesh.bbox.diagonal)
            object.set_pose(pose)
        elif args.placement == 'random':
            if True:
                if not scene.find_noncolliding_pose(object, sampler='random', max_iterations=50, viewpoint=torch.tensor([1.0, 0.0, 0.0])):
                    print('WARNING: Could not find non-colliding pose')
            elif True:
                pose = scene.place_object_randomly(mesh.bbox.diagonal)
                object.set_pose(pose)

    renderer = sl.RenderPass(shading=args.shading)

    if args.placement == 'tabletop':

        if args.tabletop_video:
            import subprocess

            command = [ '/usr/bin/ffmpeg',
                '-y', # (optional) overwrite output file if it exists
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-s', '640x480', # size of one frame
                '-pix_fmt', 'rgb24',
                '-r', '25', # frames per second
                '-i', '-', # The imput comes from a pipe
                '-an', # Tells FFMPEG not to expect any audio
                '-c:v', 'libx264',
                '-vf', 'fps=25',
                '-pix_fmt', 'yuv420p',
                args.tabletop_video ]

            process = subprocess.Popen(command, stdin=subprocess.PIPE)

            def vis_cb(iteration):
                result = renderer.render(scene)
                rgb = result.rgb()
                rgb = rgb[:,:,:3]
                rgb_np = rgb.cpu().numpy()

                process.stdin.write(rgb_np.tobytes())
        else:
            vis_cb = None

        s1 = time.time()
        scene.simulate_tabletop_scene(vis_cb=vis_cb)
        s2 = time.time()
        print('Tabletop sim took {}s'.format(s2-s1))

        if args.tabletop_video:
            process.stdin.close()
            process.wait()

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

    print('Projection matrix:')
    print(scene.projection_matrix())

    img = Image.fromarray(rgb_np, mode='RGB')

    if args.out:
        img.save(args.out)
    else:
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

    if args.show_depth:
        depth = result.depth()
        depth = depth.clamp_(0,1.0) / 1.0 * 255.0
        depth = Image.fromarray(depth.byte().cpu().numpy())
        depth.show()

    if args.serialize:
        print('Serialized scene description:\n')
        print(scene.serialize())
