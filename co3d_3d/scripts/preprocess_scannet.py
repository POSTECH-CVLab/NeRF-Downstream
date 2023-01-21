import argparse
import json
import os
from multiprocessing.sharedctypes import Value

import numpy as np
import torch
from scipy import spatial

from co3d_3d.src.data.scannet import CLASS_LABELS, VALID_CLASS_IDS
from co3d_3d.src.data.utils import load_ply


def assign_label(
    orig_pcd,
    orig_labels,
    pcd,
    trans_info,
    K=1,
    ths=[0.05, 0.20],
    ignore_index=-255,
    filter_index=-500,
    return_pcd=False,
):
    # zero center
    orig_norm = orig_pcd - orig_pcd.mean(axis=0, keepdims=True)

    T_inv = np.linalg.inv(trans_info["T"][:3, :3])
    pcd_transformed = (T_inv @ pcd.T).T
    pcd_scaled = pcd_transformed / trans_info["scene_scale"]

    tree = spatial.KDTree(orig_norm)
    dists, indices = tree.query(pcd_scaled, k=K, workers=12)
    labels = orig_labels[indices]

    if isinstance(ths, list) and len(ths) > 1:
        # use first threshold for deciding ignore index
        in_use = dists < ths[0]
        labels[~in_use] = ignore_index

        # use second threshold for filter out index
        filter_out = dists > ths[1]
        labels[filter_out] = filter_index
    else:
        if isinstance(ths, list):
            ths = ths[0]
        in_use = dists < ths
        labels[~in_use] = ignore_index

    return_args = [labels]
    if return_pcd:
        return_args.append(orig_norm)
        return_args.append(pcd_scaled)
    return tuple(return_args)


def assign_label_2(orig_pcd, orig_labels, pcd, trans_info, K=1):
    # zero center
    orig_norm = orig_pcd - orig_pcd.mean(axis=0, keepdims=True)

    T_inv = np.linalg.inv(trans_info["T"][:3, :3])
    pcd_transformed = (T_inv @ pcd.T).T
    pcd_scaled = pcd_transformed / trans_info["scene_scale"]

    tree = spatial.KDTree(orig_norm)
    dists, indices = tree.query(pcd_scaled, k=K, workers=12)
    labels = orig_labels[indices]
    return labels, dists


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="./datasets/scannet")
    parser.add_argument("--scannet_dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--ignore_index", type=int, default=-255)
    parser.add_argument("--filter_index", type=int, default=-500)
    parser.add_argument("--ths", type=float, default=0.05)
    parser.add_argument("--filename", type=str, default="data.ckpt")
    parser.add_argument("--minimal", action="store_true")
    args = parser.parse_args()

    ignore_index = args.ignore_index
    scenes = os.listdir(args.datadir)
    print(f"searched {len(scenes)} scenes")
    scenes = list(sorted(scenes))

    with open("./datasets/split/scannetv2_train.txt", "r") as f:
        train_splits = f.readlines()
        train_splits = [l.strip() for l in train_splits]
    with open("./datasets/split/scannetv2_val.txt", "r") as f:
        val_splits = f.readlines()
        val_splits = [l.strip() for l in val_splits]

    failed = []
    for i, scene in enumerate(scenes):
        inst_id = scene.lstrip("plenoxel_torch_")
        ckpt_file = os.path.join(args.datadir, scene, args.filename)
        conf_file = os.path.join(args.datadir, scene, "args.txt")
        trans_file = os.path.join(args.datadir, scene, "trans.npz")
        orig_pcd_file = os.path.join(args.scannet_dir, f"{inst_id}.ply")

        try:
            if not os.path.exists(ckpt_file):
                print(f"{ckpt_file} not exists, skip.")
                raise FileNotFoundError()
            if not os.path.exists(conf_file):
                print(f"{conf_file} not exists, skip.")
                raise FileNotFoundError()
            if not os.path.exists(trans_file):
                print(f"{trans_file} not exists, skip.")
                raise FileNotFoundError()
            if not os.path.exists(orig_pcd_file):
                print(f"{orig_pcd_file} not exists, skip.")
                raise FileNotFoundError()

            out_scene = os.path.join(args.outdir, scene)
            os.makedirs(out_scene, exist_ok=True)
            out_file = os.path.join(out_scene, "data.npz")
            if os.path.exists(out_file):
                print(f"skip exists: {out_file}")
                continue

            ckpt = torch.load(ckpt_file, map_location="cpu")
            links_idx = ckpt["state_dict"]["model.links_idx"].numpy().astype(np.int32)
            density = (
                ckpt["state_dict"]["model.density_data"].numpy().astype(np.float32)
            )
            reso_idx = ckpt["reso_idx"]
            if reso_idx != 1:
                raise ValueError(f"reso idx should be 1 but got {reso_idx}")

            sh = ckpt["state_dict"]["model.sh_data"].numpy()
            sh_min = ckpt["model.sh_data_min"].numpy().astype(np.float32)
            sh_scale = ckpt["model.sh_data_scale"].numpy().astype(np.float32)

            with open(conf_file, "r") as f:
                lines = [l.strip() for l in f.readlines()]
                key, value = zip(*[l.split("=") for l in lines])
                key = [k.strip() for k in key]
                value = [v.strip() for v in value]
                conf = dict(zip(key, value))
            reso_list = json.loads(conf["reso"])
            resolution = reso_list[reso_idx]
            coordinates = np.stack(
                [
                    links_idx // (resolution[1] * resolution[2]),
                    links_idx % (resolution[1] * resolution[2]) // resolution[2],
                    links_idx % resolution[2],
                ],
                -1,
            )
            coordinates = coordinates / resolution * 2 - 1.0

            trans = np.load(trans_file)
            orig_pcd, _, orig_labels, _ = load_ply(orig_pcd_file, load_label=True)

            labels, dists = assign_label_2(orig_pcd, orig_labels, coordinates, trans)
            print(
                f"{scene} total={labels.shape[0]}, valid={(dists <= args.ths).sum()}, invalid={(dists > args.ths).sum()}, ratio={(dists <= args.ths).sum() / dists.shape[0]:.2f}"
            )
            # import pdb;pdb.set_trace()

            np.savez(
                out_file,
                links=links_idx,
                density=density,
                sh=sh,
                sh_min=sh_min,
                sh_scale=sh_scale,
                reso=resolution,
                labels=labels,
                dists=dists.astype(np.float16),
            )
            print(f"[{i}/{len(scenes)}] saved {out_file}")
        except Exception as e:
            print(e, i, scene)
            failed.append(inst_id)
            # import pdb;pdb.set_trace()

    new_train_splits, new_val_splits = [], []
    new_train_splits = list(filter(lambda x: x not in failed, train_splits))
    new_val_splits = list(filter(lambda x: x not in failed, val_splits))
    print(f"failed: {failed}")
    with open("scannet_256_train.txt", "w") as f:
        f.writelines([f"{l}\n" for l in new_train_splits])
    with open("scannet_256_val.txt", "w") as f:
        f.writelines([f"{l}\n" for l in new_val_splits])
