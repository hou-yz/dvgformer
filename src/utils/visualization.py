import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from transforms3d.quaternions import qinverse, qmult, quat2mat, mat2quat, qnorm, rotate_vector
from transforms3d.axangles import axangle2mat
from scipy.spatial.transform import Rotation
from src.utils.colmap_official_read_write_model import read_model


def visualize_points(points3d):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.array([point.xyz for point in points3d.values()]))
    pcd.colors = o3d.utility.Vector3dVector(
        np.array([point.rgb / 255 for point in points3d.values()]))
    return [pcd]


def visualize_cameras(tvecs, qvecs, size=0.5):
    # Visualize camera poses
    geometries = []
    for tvec, qvec in zip(tvecs, qvecs):
        R = quat2mat(qvec)
        # Custom camera frustum with extended forward axis and arrow
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=size, origin=[0, 0, 0])
        # Apply rotation
        mesh_frame.rotate(R, center=(0, 0, 0))
        # Apply translation
        mesh_frame.translate(tvec)

        geometries.append(mesh_frame)
    return geometries


def plot_camera_lines(tvecs, color=[0.8, 0.2, 0.8]):
    """
    Plots a camera path as a curve from a sequence of translation vectors.

    Parameters:
    - tvecs: A list or numpy array of translation vectors (Nx3).
    - color: The RGB color of the curve.
    - line_width: The width of the lines. Note that Open3D might not support line width in all environments.
    """
    # Ensure tvecs is a numpy array for easier manipulation
    tvecs = np.asarray(tvecs)

    # Create points and lines for the LineSet
    lines = [[i, i + 1]
             for i in range(len(tvecs) - 1)]  # Connect consecutive points

    # Create a LineSet object and set its properties
    camera_path = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(tvecs),
        lines=o3d.utility.Vector2iVector(lines),
    )
    camera_path.paint_uniform_color(color)  # Set the color of the curve

    return [camera_path]


def create_cylinder_line(p1, p2, radius=0.005, color=[0.8, 0.2, 0.8]):
    """
    Create a cylinder between two points p1 and p2.
    """
    # Create a mesh cylinder of the given radius
    mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=np.linalg.norm(p2 - p1))
    mesh_cylinder.paint_uniform_color(color)

    # Compute the transformation for the cylinder
    cyl_transform = np.eye(4)
    cyl_transform[0:3, 3] = (p1 + p2) / 2
    # Align the cylinder with the line p1 -> p2
    v = p2 - p1
    v /= np.linalg.norm(v)
    axis = np.array([0, 0, 1])  # Initial axis of the cylinder
    axis_x_v = np.cross(axis, v)
    angle = np.arctan2(np.linalg.norm(axis_x_v), np.dot(axis, v))

    cyl_transform[0:3, 0:3] = o3d.geometry.get_rotation_matrix_from_axis_angle(
        axis_x_v / np.linalg.norm(axis_x_v) * angle)

    # Apply the transformation
    mesh_cylinder.transform(cyl_transform)

    return mesh_cylinder


def plot_camera_cylinders(tvecs, radius=0.03, color=[0.8, 0.2, 0.8]):
    """
    Plots a camera path as a series of cylinders.
    """
    line_set = o3d.geometry.LineSet()
    cylinders = []

    for i in range(len(tvecs) - 1):
        p1 = tvecs[i]
        p2 = tvecs[i + 1]
        cylinders.append(create_cylinder_line(
            np.array(p1), np.array(p2), radius=radius, color=color))

    return cylinders


def main():
    # Example tvecs and qvecs
    # Replace these with your actual tvecs
    # tvecs = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    # # Replace these with your actual qvecs
    # qvecs = [np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0])]
    # cameras, images, points3D = read_model(
    #     'demo/8jT9ygmMvMg/scene00002_0/sparse/0')

    # calculation from tvec and qvec directly
    # transforms3d
    # tvecs = np.array([-qmult(qinverse(img.qvec),
    #                          qmult(np.insert(img.tvec, 0, 0), img.qvec))[1:] for img in images.values()])
    # qvecs = np.array([qinverse(img.qvec) for img in images.values()])
    # scipy
    # tvecs, qvecs = [], []
    # for img in images.values():
    #     rotation = Rotation.from_quat(img.qvec[[1, 2, 3, 0]]).inv()
    #     tvecs.append(-rotation.apply(img.tvec))
    #     qvecs.append(rotation.as_quat()[[3, 0, 1, 2]])
    # tvecs, qvecs = np.array(tvecs), np.array(qvecs)

    import time
    from transforms3d.euler import euler2quat, quat2euler, euler2mat, mat2euler
    from src.blender.blender_camera_env import R_colmap_from_blender, R_blender_cam_dir
    from src.utils.quaternion_operations import convert_to_global_frame, convert_to_local_frame, interpolate_eulers, interpolate_tvecs
    # Example usage
    # fpath = 'None/videos/fpv_rome_return10.00_crashNone_501_2025-01-25_15-45-46_config.txt'
    fpath = 'None/videos/debug_None_2025-01-26_01-13-14_config.txt'

    # blender coordinates
    loc_rot = np.loadtxt(fpath)
    loc_rot = loc_rot[::3]
    locations = loc_rot[:, :3]
    directions = loc_rot[:, 3:]
    # tvec and qvec from blender
    tvecs = locations
    qvecs = np.zeros((len(directions), 4))
    for i in range(len(directions)):
        rot = directions[i]
        # R_rot is for rotating the blender defualt camera direction in the blender world plane
        R_rot = euler2mat(*rot, axes='sxyz')
        # apply the global (blender world plane) rotation to the blender defualt camera direction
        # R_blender is for rotating the blender world plane to the actual camera direction
        R_blender = R_rot @ R_blender_cam_dir
        qvec = mat2quat(R_blender)
        qvecs[i] = qvec

    # Load the mesh or point cloud
    mesh = o3d.io.read_triangle_mesh(
        '/home/houyz/Data/blosm/himeji/scene.ply')
    # Create a point cloud from the vertices of the mesh
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = mesh.vertices

    # Downsample with a voxel size of (for example) 0.02
    voxel_size = 2.0
    downsampled_pcd = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    # add color to the downsampled point cloud according to their z value
    z = np.asarray(downsampled_pcd.points)[:, 2]
    z = (z - z.min()) / (z.max() - z.min())
    downsampled_pcd.colors = o3d.utility.Vector3dVector(
        # plt.cm.viridis(z)[:, :3]
        plt.cm.rainbow(z)[:, :3]
    )

    # visualize(tvecs, qvecs)
    geometries = []
    # geometries.extend(visualize_points(points3D))
    geometries.extend(visualize_cameras(tvecs, qvecs, 2))
    # geometries.extend(visualize_cameras(tvecs[:36], qvecs[:36], 1))
    # geometries.extend(visualize_cameras(tvecs[36:], qvecs[36:], 2))
    # geometries.extend(plot_camera_lines(tvecs))
    geometries.extend(plot_camera_cylinders(tvecs, radius=0.1))
    # geometries.extend(plot_camera_cylinders(
    #     tvecs[:36], radius=0.1, color=[0.6, 0.6, 0.6]))
    # geometries.extend(plot_camera_cylinders(tvecs[36:], radius=0.1))
    geometries.extend([downsampled_pcd])

    # Visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().line_width = 200
    # vis.get_render_option().point_size = 5

    # Add geometries to the visualizer
    for geometry in geometries:
        vis.add_geometry(geometry)

    # Get view control and adjust camera
    view_ctl = vis.get_view_control()

    # Set the front, lookat, up, and zoom parameters
    # These parameters adjust the camera's position and orientation in the visualization
    # Example: Setting a view that roughly corresponds to rotating the scene 180 degrees around the Y-axis
    # Note: You'll need to adjust these values based on your specific dataset and desired view
    # camera_params = {
    #     # Points in the direction the camera is looking
    #     # "front": np.array([0., 0., -1.]),
    #     # pitch the camera down a bit
    #     "front": np.array([1., -0.5, -1.]),
    #     # The point the camera is looking at
    #     "lookat": np.array([0., 0., 0.]),
    #     # The "up" direction in the camera coordinate system
    #     "up": np.array([0., -1., 0.]),
    #     "zoom": 0.1                        # Zoom level
    # }

    # in blender convention
    # x: right, y: forward, z: up
    camera_params = {
        # Points in the direction the camera is looking
        "front": np.array([-1., -0.5, 0.5]),
        # The point the camera is looking at
        "lookat": tvecs[0],
        # The "up" direction in the camera coordinate system
        "up": np.array([0., 0., 1.]),
        "zoom": 0.05                        # Zoom level
    }
    view_ctl.set_front(camera_params["front"])
    view_ctl.set_lookat(camera_params["lookat"])
    view_ctl.set_up(camera_params["up"])
    view_ctl.set_zoom(camera_params["zoom"])

    camera_params = view_ctl.convert_to_pinhole_camera_parameters()
    # Save camera location and the lookat point
    R, t = camera_params.extrinsic[:3, :3], camera_params.extrinsic[:3, 3]
    camera_location = -R.T @ t
    lookat_point = tvecs[0]
    # Save the camera parameters
    np.savetxt("camera_location.txt", camera_location, fmt='%f')
    np.savetxt("lookat_point.txt", lookat_point, fmt='%f')

    # save the 3d scene as .glb file with all the geometries
    # merges all points into one cloud
    combined_pcd = o3d.geometry.PointCloud()
    for g in geometries:
        if isinstance(g, o3d.geometry.PointCloud):
            combined_pcd += g
    o3d.io.write_point_cloud("combined_pointcloud.ply", combined_pcd)
    # merges vertices & triangles
    combined_mesh = o3d.geometry.TriangleMesh()
    for g in geometries:
        if isinstance(g, o3d.geometry.TriangleMesh):
            combined_mesh += g
    o3d.io.write_triangle_mesh("combined_mesh.ply", combined_mesh)

    # Run the visualizer
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()
