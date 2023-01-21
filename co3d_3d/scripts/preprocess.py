import argparse
import json
import os

import numpy as np
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="./datasets/co3d")
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    scenes = os.listdir(args.datadir)
    print(f"searched {len(scenes)} scenes")

    failed = []
    for i, scene in enumerate(scenes):
        ckpt_file = os.path.join(args.datadir, scene, "last.ckpt")
        if not os.path.exists(ckpt_file):
            print(f"{scene} not exists, skip.")

        out_scene = os.path.join(args.outdir, scene)
        os.makedirs(out_scene, exist_ok=True)
        out_file = os.path.join(out_scene, "data.npz")
        if os.path.exists(out_file):
            print(f"skip exists: {out_file}")
            continue

        reso = [256, 256, 256]
        os.system(f"ls -lh {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        links = ckpt["state_dict"]["model.links_idx"].numpy().astype(np.int32)
        links_idx = np.stack([
            links // (reso[1] * reso[2]),
            links % (reso[1] * reso[2]) // reso[2],
            links % reso[2],
        ])
        sel = (links_idx % 2 == 0).all(axis=0)
        links_idx = links_idx[:, sel]
        links = (links_idx[0] * 128 * 128 + links_idx[1] * 128 + links_idx[2]) // 2
        density = (
            ckpt["state_dict"]["model.density_data"].numpy().astype(np.float32)
        )[sel]
        sh = ckpt["state_dict"]["model.sh_data"].numpy()[sel]
        sh_min = ckpt["model.sh_data_min"].numpy().astype(np.float32)
        sh_scale = ckpt["model.sh_data_scale"].numpy().astype(np.float32)

        np.savez(
            out_file,
            links=links,
            density=density,
            sh=sh,
            sh_min=sh_min,
            sh_scale=sh_scale,
            reso=[[128, 128, 128], [256, 256, 256]],
        )
        print(f"[{i}/{len(scenes)}] saved {out_file}")
    print(f"failed: {failed}")
