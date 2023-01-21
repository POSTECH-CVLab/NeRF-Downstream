import os

features = ["one", "density", "sh", "shdensity"]
models = ["14a", "18a", "34c"]
seed = [0, 100]

basedir = "./neurips2022_seed"

for m in models:
    for f in features:
        for s in seed:
            command = f"python eval.py --ginc configs/scannet_plenoxel.gin --ginc configs/resunet{m}.gin --ginc configs/scannet_feature_{f}.gin --load_path ./neurips2022_eval/b8x1-ResUNet{m.upper()}-feature_{f}_{s}/0/checkpoints/best.ckpt"
            print(command)
