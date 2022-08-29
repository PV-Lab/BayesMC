#!/bin/bash

#SBATCH -o batch.sh.log-%j
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20

# Loading the required module
source /etc/profile
module load anaconda/2021a

# Run the script
python run.py
