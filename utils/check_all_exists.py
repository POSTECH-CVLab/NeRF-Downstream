import os

if __name__ == "__main__":

    assert os.path.exists("co3d_2d/data/co3d")
    assert os.path.exists("co3d_2d/data/ours")

    assert os.path.exists("filelist/train.txt")
    assert os.path.exists("filelist/val.txt")
    assert os.path.exists("filelist/test.txt")

    co3d_files = []
    ours_files = []

    split = lambda x: x.rstrip("\n").split()

    with open("filelist/train.txt") as fp:
        train_lines = list(map(split, fp.readlines()))
        co3d_files += [f"co3d_2d/data/co3d/{cls_name}/{scene_number}" for (cls_name, scene_number, frame_num) in train_lines]
        ours_files += [f"co3d_2d/data/ours/{cls_name}/{scene_number}" for (cls_name, scene_number, frame_num) in train_lines]

    with open("filelist/val.txt") as fp:
        val_lines = list(map(split, fp.readlines()))
        co3d_files += [f"co3d_2d/data/co3d/{cls_name}/{scene_number}" for (cls_name, scene_number, frame_num) in val_lines]
        ours_files += [f"co3d_2d/data/ours/{cls_name}/{scene_number}" for (cls_name, scene_number, frame_num) in val_lines]

    with open("filelist/test.txt") as fp:
        test_lines = list(map(split, fp.readlines()))
        co3d_files += [f"co3d_2d/data/co3d/{cls_name}/{scene_number}" for (cls_name, scene_number, frame_num) in test_lines]
        ours_files += [f"co3d_2d/data/ours/{cls_name}/{scene_number}" for (cls_name, scene_number, frame_num) in test_lines]

    assert len(co3d_files) == 18619
    assert len(ours_files) == 18619

    for fpath in co3d_files:
        assert os.path.exists(fpath)
        for image_name in os.listdir(f"{fpath}/images"):
            assert image_name.endswith(".jpg")

    for fpath in ours_files:
        assert os.path.exists(fpath)
        assert len(os.listdir(f"{fpath}/fgbg")) == 50