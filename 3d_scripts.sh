model=$1
feature=$2
seed=$3

python3 -m co3d_3d.train \
--ginc co3d_3d/configs/co3d_cls.gin \
--ginc co3d_3d/configs/co3d_aug3.gin \
--ginc co3d_3d/configs/feature_${feature}.gin \
--ginc co3d_3d/configs/${model}.gin \
--run_name ${model}_${feature}_${seed} \
--seed ${seed}