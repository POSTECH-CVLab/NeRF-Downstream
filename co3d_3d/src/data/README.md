# Dataset Preparation


## Scannet

```
python2 download-scannet.py --type _vh_clean.ply -o scannet-ply; python2 download-scannet.py --type _vh_clean.segs.json -o scannet-ply

# Download splits as well

cd scannet-ply
wget https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/scannetv2_train.txt
wget https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/scannetv2_val.txt
wget https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/scannetv2_test.txt
```

This should generate the following directory structure

```
scannet-ply
├── scannetv2_train.txt
├── scannetv2_val.txt
├── scannetv2_test.txt
├── scans
│   ├── scene0000_00
│   │   ├── scene0000_00_vh_clean_2.labels.ply
│   │   └── scene0000_00_vh_clean_2.ply
│   ├── ...
│   └── scene0706_00
│       ├── scene0706_00_vh_clean_2.labels.ply
│       └── scene0706_00_vh_clean_2.ply
└── scans_test
    ├── scene0707_00
    │   ├── scene0707_00_vh_clean_2.labels.ply
    │   └── scene0707_00_vh_clean_2.ply
    ├── ...
    └── scene0806_00
        ├── scene0806_00_vh_clean_2.labels.ply
        └── scene0806_00_vh_clean_2.ply
```

## Semantic KITTI

```
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_calib.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip
wget http://www.semantic-kitti.org/assets/data_odometry_labels.zip

unzip data_odometry_velodyne.zip
unzip data_tracking_calib.zip
unzip data_odometry_labels.zip
unzip data_odometry_calib.zip
```
