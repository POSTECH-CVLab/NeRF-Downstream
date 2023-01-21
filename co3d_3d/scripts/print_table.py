import json
import os

import numpy as np

models = ["14A", "18A", "34C"]
features = ["one", "density", "sh", "shdensity"]
seeds = [0, 100, 777]

for m in models:
    for f in features:
        miou_list = []
        macc_list = []
        for s in seeds:
            if s < 700:
                exp = f"neurips2022_seed/b8x1-Res16UNet{m}-feature_{f}_{s}/0/checkpoints/eval_results.json"
            else:
                basedir = f"neurips2022/b8x1-Res16UNet{m}-rot_y_segmentation-plenoxel-scannet/"
                logs = os.listdir(basedir)
                logs = sorted(logs)
                log = logs[-1]
                exp = os.path.join(basedir, log, "checkpoints", "eval_results.json")
            with open(exp, "r") as fd:
                data = json.load(fd)
            miou = np.array(data["iou"])
            macc = np.array(data["acc"])
            miou_list.append(miou)
            macc_list.append(macc)
            miou_list = np.stack(miou_list, axis=1)
            miou_list[-1, :] *= 100.0
            macc_list = np.stack(macc_list, axis=1)
            macc_list[-1, :] *= 100.0
            mean = miou_list.mean(axis=1)
            std = np.std(miou_list, axis=1)
            string = " & ".join(
                [
                    f"{_m:.1f}$\pm" + "{" + f"{_s:.1f}" + "}$"
                    for _m, _s in zip(mean, std)
                ]
            )
            if "sh" in f:
                string = "\checkmark & " + string
            else:
                string = " & " + string
            if "density" in f:
                string = "\checkmark & " + string
            else:
                string = "& " + string
            string = m + " & " + string
            string = string + " \\\\ "
            print(string)
