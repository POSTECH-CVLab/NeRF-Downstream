import MinkowskiEngine as ME
import numpy as np
import torch
from matplotlib import cm
from plyfile import PlyData


def load_ply(filename, load_label=False, load_instance=False):
    plydata = PlyData.read(filename)
    data = plydata.elements[0].data
    coords = np.array([data["x"], data["y"], data["z"]], dtype=np.float32).T
    feats = np.array([data["red"], data["green"], data["blue"]], dtype=np.float32).T
    return_args = [coords, feats]
    if load_label:
        labels = np.array(data["label"], dtype=np.int32)
        return_args.append(labels)
    if load_instance:
        instances = np.array(data["instance"], dtype=np.int32)
    else:
        instances = np.ones(coords.shape[0], dtype=np.int32)
    return_args.append(instances)
    return tuple(return_args)


def collate_mink(list_data):
    coordinates_batch, features_batch = ME.utils.sparse_collate(
        [d["coordinates"] for d in list_data],
        [d["features"] for d in list_data],
        dtype=torch.float32,
    )

    package = {
        "coordinates": coordinates_batch,
        "features": features_batch,
    }
    for key in list_data[0].keys():
        if "label" in key:
            package[key] = torch.from_numpy(np.concatenate([d[key] for d in list_data]))
        if "instance" in key:
            package[key] = torch.from_numpy(np.concatenate([d[key] for d in list_data]))
        if "dists" in key:
            package[key] = torch.from_numpy(np.concatenate([d[key] for d in list_data]))

    if "metadata" in list_data[0]:
        package["metadata"] = [d["metadata"] for d in list_data]
    if "dataset" in list_data[0]:
        package["dataset"] = [d["dataset"] for d in list_data]
    if "colors" in list_data[0]:
        package["colors"] = [d["colors"] for d in list_data]
    return package


def collate_pointnet(list_data):
    xyzs = [d["xyzs"] for d in list_data]
    features = [d["features"] for d in list_data]
    num_points_per_batch = np.array([xyz.shape[0] for xyz in xyzs])
    num_feats_per_batch = np.array([feature.shape[0] for feature in features])
    assert np.all(
        num_points_per_batch == num_points_per_batch[0]
    ), f"number of points per batch should be same when using 'collate_pointnet' function."
    assert np.all(
        num_feats_per_batch == num_points_per_batch
    ), f"number of features per batch should be same when using 'collate_pointnet' function."

    xyzs_batch = torch.from_numpy(np.stack(xyzs, axis=0)).float()
    features_batch = torch.from_numpy(np.stack(features, axis=0)).float()
    package = {"xyzs": xyzs_batch, "features": features_batch}
    for key in list_data[0].keys():
        if "label" in key:
            package[key] = torch.from_numpy(np.concatenate([d[key] for d in list_data]))
        if "instance" in key:
            package[key] = torch.from_numpy(np.concatenate([d[key] for d in list_data]))
    return package


def collate_pair(list_data):
    xyz0, xyz1, coords0, coords1, feats0, feats1, matching_inds, trans = list(
        zip(*list_data)
    )
    xyz_batch0, xyz_batch1 = [], []
    matching_inds_batch, trans_batch, len_batch = [], [], []

    batch_id = 0
    curr_start_inds = np.zeros((1, 2))

    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        else:
            raise ValueError(f"Can not convert to torch tensor, {x}")

    for batch_id, _ in enumerate(coords0):
        N0 = coords0[batch_id].shape[0]
        N1 = coords1[batch_id].shape[0]

        xyz_batch0.append(to_tensor(xyz0[batch_id]))
        xyz_batch1.append(to_tensor(xyz1[batch_id]))

        trans_batch.append(to_tensor(trans[batch_id]))

        matching_inds_batch.append(
            torch.from_numpy(np.array(matching_inds[batch_id]) + curr_start_inds)
        )
        len_batch.append([N0, N1])

        # Move the head
        curr_start_inds[0, 0] += N0
        curr_start_inds[0, 1] += N1

    coords_batch0, feats_batch0 = ME.utils.sparse_collate(coords0, feats0)
    coords_batch1, feats_batch1 = ME.utils.sparse_collate(coords1, feats1)

    # Concatenate all lists
    xyz_batch0 = torch.cat(xyz_batch0, 0).float()
    xyz_batch1 = torch.cat(xyz_batch1, 0).float()
    trans_batch = torch.cat(trans_batch, 0).float()
    matching_inds_batch = torch.cat(matching_inds_batch, 0).int()

    return {
        "pcd0": xyz_batch0,
        "pcd1": xyz_batch1,
        "sinput0_C": coords_batch0,
        "sinput0_F": feats_batch0.float(),
        "sinput1_C": coords_batch1,
        "sinput1_F": feats_batch1.float(),
        "correspondences": matching_inds_batch,
        "T_gt": trans_batch,
        "len_batch": len_batch,
    }


def create_o3d_pointcloud(coords, colors=None, colormap=None):
    r"""
    if colors given as a vector, use the color map.
    """
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()
    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(coords))
    if colors is not None:
        if isinstance(colors, torch.Tensor):
            colors = colors.cpu().numpy()
        if colors.ndim == 1 and colors.shape[0] == 3:
            colors = np.repeat(colors[None, :], len(coords), axis=0)
        elif colors.ndim == 1 and len(colors) == len(coords):
            if colormap is None:
                colormap = cm.get_cmap("jet")
            # Use the colormap for colors
            colors, heats = np.zeros((len(colors), 3)), colors
            for i, heat in enumerate(heats):
                colors[i] = colormap(heat)[:3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# https://github.com/theNded/Open3D-animation-workflow/blob/master/test.py
def custom_draw_geometry_with_camera_trajectory(mesh, traj):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.traj = traj
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer()

    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory
        if glb.index >= 0:
            pass
            # print("Capture image {:05d}".format(glb.index))
            # depth = vis.capture_depth_float_buffer(False)
            # image = vis.capture_screen_float_buffer(False)
            # plt.imsave("../../TestData/depth/{:05d}.png".format(glb.index),\
            #         np.asarray(depth), dpi = 1)
            # plt.imsave("../../TestData/image/{:05d}.png".format(glb.index),\
            #         np.asarray(image), dpi = 1)
            # vis.capture_depth_image("depth/{:05d}.png".format(glb.index), False)
            # vis.capture_screen_image("image/{:05d}.png".format(glb.index), False)
        glb.index = glb.index + 1
        if glb.index < len(glb.traj.parameters):
            ctr.convert_from_pinhole_camera_parameters(glb.traj.parameters[glb.index])
        else:
            custom_draw_geometry_with_camera_trajectory.vis.register_animation_callback(
                None
            )
        return False

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window()
    vis.add_geometry(mesh)
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()
