#!/bin/bash

#SBATCH --job-name="Kadara"
#SBATCH --error="Kadara.err"
#SBATCH --output="Kadara.output"
#SBATCH --reservation=yg9ca_43
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
module load anaconda
source activate base
module load cudnn
module load gcc
python main.py
