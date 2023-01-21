model=$1
option=$2
seed=$3

python3 -m co3d_2d.train --ginc co3d_2d/configs/${option}/${model}.gin --seed ${seed}