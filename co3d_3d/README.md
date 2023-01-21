# NeRF Downstream

## Installation

```
conda create -n nerf_downstream python=3.8
conda activate nerf_downstream
conda install pytorch=1.8.0 torchvision=0.9.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -r requirements.txt

# Install MinkowskiEngine
sudo apt install libopenblas-dev
pip install -U MinkowskiEngine --install-option="--blas=openblas" -v --no-deps
```

## Data Prep

Please read `datasets/README.md` for more detail.

```
NeRF-Downstream
 |-...
 |-datasets
    |-co3d_10p
    |-co3d
    |-...
```
create symlinks in `./datasets`.

```
ln -n PATH/TO/CO3D_10% ./datasets/co3d_10p
ln -n PATH/TO/CO3D ./datasets/co3d
```

## Usage

Configuration files use `--ginc`, bindings use `--ginb`.
If configuration files contain the same parameters, the parameter will be overwritten by the last config file. e.g. `--ginc default.gin --ginc overwrite.gin` then `overwrite.gin` will overwrite any values.

Similarly, `--ginb` will overwrite any parameters. e.g. `--ginb train.max_steps=1000 --ginc default.ginc`, then `max_steps` parameter will follow the binding regardless of the position.

### Example Co3D 10p Classification Training

```
python train.py --ginc configs/co3d_cls.gin --ginc configs/resnet14.gin
```
