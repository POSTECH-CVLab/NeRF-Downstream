import copy

import numpy as np
import torch
from scipy.linalg import expm, norm


# Rotation matrix along axis with angle theta
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def sample_random_trans(pcd, randg, rotation_range=360):
    T = np.eye(4)
    R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
    T[:3, :3] = R
    T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
    return T


def make_open3d_point_cloud(xyz, color=None):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def get_matching_indices(source, target, trans, search_voxel_size, K=None):
    import open3d as o3d

    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds


def pdist(A, B, dist_type="L2"):
    if dist_type == "L2":
        D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
        return torch.sqrt(D2 + 1e-7)
    elif dist_type == "SquareL2":
        return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    else:
        raise NotImplementedError("Not implemented")


def corr_dist(est, gth, xyz0, xyz1, weight=None, max_dist=1):
    xyz0_est = xyz0 @ est[:3, :3].t() + est[:3, 3]
    xyz0_gth = xyz0 @ gth[:3, :3].t() + gth[:3, 3]
    dists = torch.clamp(torch.sqrt(((xyz0_est - xyz0_gth).pow(2)).sum(1)), max=max_dist)
    if weight is not None:
        dists = weight * dists
    return dists.mean()


def find_corr(xyz0, xyz1, F0, F1, subsample_size=-1, nn_max_n=500):
    subsample = len(F0) > subsample_size
    if subsample_size > 0 and subsample:
        N0 = min(len(F0), subsample_size)
        N1 = min(len(F1), subsample_size)
        inds0 = np.random.choice(len(F0), N0, replace=False)
        inds1 = np.random.choice(len(F1), N1, replace=False)
        F0, F1 = F0[inds0], F1[inds1]

    # Compute the nn
    nn_inds = find_nn_gpu(F0, F1, nn_max_n=nn_max_n)
    if subsample_size > 0 and subsample:
        return xyz0[inds0], xyz1[inds1[nn_inds]]
    else:
        return xyz0, xyz1[nn_inds]


def find_nn_gpu(F0, F1, nn_max_n=-1, return_distance=False, dist_type="SquareL2"):
    # Too much memory if F0 or F1 large. Divide the F0
    if nn_max_n > 1:
        N = len(F0)
        C = int(np.ceil(N / nn_max_n))
        stride = nn_max_n
        dists, inds = [], []
        for i in range(C):
            dist = pdist(F0[i * stride : (i + 1) * stride], F1, dist_type=dist_type)
            min_dist, ind = dist.min(dim=1)
            dists.append(min_dist.detach().unsqueeze(1).cpu())
            inds.append(ind.cpu())

        if C * stride < N:
            dist = pdist(F0[C * stride :], F1, dist_type=dist_type)
            min_dist, ind = dist.min(dim=1)
            dists.append(min_dist.detach().unsqueeze(1).cpu())
            inds.append(ind.cpu())

        dists = torch.cat(dists)
        inds = torch.cat(inds)
        assert len(inds) == N
    else:
        dist = pdist(F0, F1, dist_type=dist_type)
        min_dist, inds = dist.min(dim=1)
        dists = min_dist.detach().unsqueeze(1).cpu()
        inds = inds.cpu()
    if return_distance:
        return inds, dists
    else:
        return inds
