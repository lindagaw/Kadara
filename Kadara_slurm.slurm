(myenv) -bash-4.2$vi myjobs.25843213_4294967294.out
#!/bin/bash
#SBATCH --nodes=20 #total number of nodes for the job
#SBATCH --ntasks=1 #how many copies of code to run
#SBATCH --time=1-12:00:00 #amount of time for the whole job
#SBATCH --partition=gpu #the queue/partition to run on
#SBATCH --gres=gpu:rtx2080:8
#SBATCH --account=stressresearch #the account/allocation to use
#SBATCH -o myjobs.%A_%a.out
#SBATCH -e myjobs.%A_%a.err
module purge
module load anaconda
source activate myenv
module load cudnn
module load gcc #load modules that my job needs
module load singularity

# Assuming that the container has been copied to the user's home directory
# singularity run --nv ~/pytorch-1.4.0-py37.sif main.py
python main.py
