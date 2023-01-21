import argparse
import os
from typing import OrderedDict

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./datasets/co3d")
    parser.add_argument("--out_dir", type=str, default="./datasets/split")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=-1.0)
    args = parser.parse_args()

    dataset_name = args.data_dir.split("/")[-1]

    with open(
        os.path.join(os.path.dirname(__file__), "../datasets/split/co3d_use.txt"), "r"
    ) as f:
        lines = [l.strip().split("\t")[-1] for l in f.readlines()]

    odict = OrderedDict()
    for l in lines:
        label, inst_id = l.split("/")
        odict[inst_id] = label

    classes = sorted(np.unique(list(odict.values())))
    data = os.listdir(args.data_dir)
    inst_ids = [d.strip("plenoxel_torch_") for d in data]
    for inst_id in inst_ids:
        assert inst_id in odict, f"{inst_id} not a valid co3d instance."

    labels = [odict[i.strip("plenoxel_torch_")] for i in data]
    label_id_map = dict()
    for l, i in zip(labels, inst_ids):
        if l in label_id_map:
            label_id_map[l].append(i)
        else:
            label_id_map[l] = []

    train_split_file = os.path.join(args.out_dir, f"{dataset_name}_train.txt")
    val_split_file = os.path.join(args.out_dir, f"{dataset_name}_val.txt")
    test_split_file = os.path.join(args.out_dir, f"{dataset_name}_test.txt")

    train_ratio = (
        1 - args.val_ratio
        if args.test_ratio < 0
        else 1 - (args.val_ratio + args.test_ratio)
    )
    train_split, val_split, test_split = dict(), dict(), dict()
    for label, ids in label_id_map.items():
        train_idx = int(train_ratio * len(ids))
        val_idx = (
            int(args.val_ratio * len(ids)) + train_idx
            if args.test_ratio > 0
            else len(ids)
        )
        test_idx = val_idx if args.test_ratio > 0 else train_idx
        train_split[label] = ids[:train_idx]
        val_split[label] = ids[train_idx:val_idx]
        test_split[label] = ids[test_idx:]

    labels = list(sorted(label_id_map.keys()))
    print(f"all classes ({len(labels)}): {labels}")
    for label in labels:
        print(
            f">> {label:<20}: total={len(label_id_map[label]):4d}, train={len(train_split[label]):4d}, val={len(val_split[label]):4d}, test={len(test_split[label])}"
        )

    def tolist(split):
        lines = []
        for label, ids in split.items():
            for inst_id in ids:
                line = f"{label}\t{inst_id}\n"
                lines.append(line)
        return lines

    with open(train_split_file, "w") as f:
        f.writelines(tolist(train_split))
    with open(val_split_file, "w") as f:
        f.writelines(tolist(val_split))
    with open(test_split_file, "w") as f:
        f.writelines(tolist(test_split))
