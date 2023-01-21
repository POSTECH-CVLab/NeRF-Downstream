import os
import numpy as np
import argparse

def generate_split(args):

    class_list = os.listdir(args.co3d_path)
    scene_num_class = {cls_name: 0 for cls_name in class_list}
    train_list, val_list, test_list = [], [], []
    for cls_name in sorted(class_list):
        cls_path = os.path.join(args.co3d_path, cls_name)
        scene_list = [
            dirname for dirname in sorted(os.listdir(cls_path))
            if os.path.isdir(os.path.join(cls_path, dirname)) 
        ]
        frame_num_list = [
            len(os.listdir(os.path.join(args.co3d_path, cls_name, scene_num, "images")))
            for scene_num in scene_list
        ]
        num_scenes = len(scene_list)
        num_train, num_val = int(0.8 * num_scenes), int(0.1 * num_scenes)
        num_test = num_scenes - num_train - num_val
        index_list = np.arange(num_scenes)
        np.random.shuffle(index_list)
        scene_num_class[cls_name] = len(scene_list)
        train_list += [(scene_list[idx], cls_name, frame_num_list[idx]) for idx in index_list[:num_train]]
        val_list += [(scene_list[idx], cls_name, frame_num_list[idx]) for idx in index_list[num_train:num_train + num_val]]
        test_list += [(scene_list[idx], cls_name, frame_num_list[idx]) for idx in index_list[num_train + num_val: num_train + num_val + num_test]]

    with open("configs/filelist/train.txt", "w") as fp:
        for scene_name, cls_name, frame_num in train_list:
            fp.write(f"{cls_name} {scene_name} {frame_num} \n")

    with open("configs/filelist/val.txt", "w") as fp:
        for scene_name, cls_name, frame_num in val_list:
            fp.write(f"{cls_name} {scene_name} {frame_num} \n")

    with open("configs/filelist/test.txt", "w") as fp:
        for scene_name, cls_name, frame_num in test_list:
            fp.write(f"{cls_name} {scene_name} {frame_num} \n")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--co3d_path",
        type=str,
        required=True,
        help="path to co3d dataset"
    )
    args = parser.parse_args()
    generate_split(args)