#!/bin/bash
sudo apt install libopenblas-dev -y
conda create -n nerf_downstream python=3.8 -y
conda install pytorch=1.8.0 torchvision=0.9.0 cudatoolkit=11.1 -c pytorch -c nvidia -y
pip install -r requirements.txt
pip install -U MinkowskiEngine --install-option="--blas=openblas" -v --no-deps
