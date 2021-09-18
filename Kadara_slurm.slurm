#!/bin/bash

#SBATCH --job-name="Kadara"
#SBATCH --error="Kadara.err"
#SBATCH --output="Kadara.output"
conda activate base
module load cudnn
module load gcc
python main.py
