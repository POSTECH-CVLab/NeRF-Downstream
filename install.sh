#!/bin/sh

SERVER=${2:-local} 

if [[ $SERVER = *local* ]]; then
    echo "[INFO] Running on Local: You should manually load modules..."
    conda init bash
    source ~/anaconda3/etc/profile.d/conda.sh # you may need to modify the conda path.
    export CUDA_HOME=/usr/local/cuda
else
    echo "[INFO] Running on Server..."
    conda init bash
    source ~/anaconda3/etc/profile.d/conda.sh

    module purge
    module load autotools 
    module load prun/1.3 
    module load gnu8/8.3.0 
    module load singularity
    
    module load cuDNN/cuda/11.1/8.0.4.30 
    module load cuda/11.1
    module load nccl/cuda/11.1/2.8.3

    echo "[INFO] Loaded all modules."
fi;

ENVS=$(conda env list | awk '{print $1}' )

if [[ $ENVS = *"$1"* ]]; then
    echo "[INFO] \"$1\" already exists. Pass the installation."
else
    echo "[INFO] Creating $1..."
    conda create -n $1 python=3.8 -y
    conda activate "$1"
    echo "[INFO] Done."

    echo "[INFO] Installing OpenBLAS and PyTorch..."
    conda install pytorch=1.10.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia -y
    conda install numpy -y
    conda install openblas-devel -c anaconda -y
    echo "[INFO] Done."

    echo "[INFO] Installing other dependencies..."
    conda install -c anaconda pandas scipy h5py scikit-learn -y
    conda install -c conda-forge plyfile pytorch-lightning wandb gin-config -y
    pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
    echo "[INFO] Done."

    echo "[INFO] Installing MinkowskiEngine..."
    cd MinkowskiEngine
    python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas --force_cuda
    cd ..
    echo "[INFO] Done."

    TORCH="$(python -c "import torch; print(torch.__version__)")"
    ME="$(python -c "import MinkowskiEngine as ME; print(ME.__version__)")"

    echo "[INFO] Finished the installation!"
    echo "[INFO] ========== Configurations =========="
    echo "[INFO] PyTorch version: $TORCH"
    echo "[INFO] MinkowskiEngine version: $ME"
    echo "[INFO] ===================================="
fi;