#!/bin/bash
#SBATCH  --output=log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source /scratch_net/bmicdl03/jonatank/conda/etc/profile.d/conda.sh shell.bash hook
conda activate pytorch9

python -u run_train_unet.py --model_name UNET_BRATS_1 --config conf/conf_unet.yaml --subjs 5

