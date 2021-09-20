#!/bin/bash

#SBATCH --job-name="Kadara"
#SBATCH --error="slurm.err"
#SBATCH --output="slurm.out"
#SBATCH --mem=30GB
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu
module load anaconda
source activate base
module load cudnn
module load gcc
python main.py
