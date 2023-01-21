import argparse
import logging

import gin
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from co3d_3d.src.data.datasets import get_dataset
from train import get_model, setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ginc",
        action="append",
        help="gin config file",
    )
    parser.add_argument(
        "--ginb",
        action="append",
        help="gin bindings",
    )
    parser.add_argument(
        "--phase", type=str, default="train", choices=["train", "val", "test"]
    )
    parser.add_argument(
        "--p",
        type=float,
        default=-1.0,
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    exp_name = f"visualize_" + "-".join(args.ginc)
    setup_logger(exp_name, args.debug)

    ginbs = []
    if args.ginb:
        ginbs.extend(args.ginb)
    logging.info(f"Gin configuration files: {args.ginc}")
    logging.info(f"Gin bindings: {ginbs}")
    gin.parse_config_files_and_bindings(args.ginc, ginbs)

    dataset = get_dataset()(phase=args.phase)

    for i, d in enumerate(dataset):
        metadata = d["metadata"]
        name = f"[{i}/{len(dataset)}:{args.phase}]" + "-".join(metadata["file"])

        coordinates = d["coordinates"].numpy()
        density = d["features"].numpy().squeeze()
        density -= density.min()
        density /= density.max()

        if args.p > 0:
            thres = np.percentile(density, args.p)
            sel = density > thres
            print(len(sel), sel.sum())
            coordinates = coordinates[sel]
            density = density[sel]

        colors = plt.cm.plasma(density.squeeze())[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coordinates)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pcd], window_name=name)

    import pdb

    pdb.set_trace()
