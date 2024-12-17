#!/bin/bash --login

#SBATCH --job-name=FlipBlock
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=v100:4
#SBATCH --cpus-per-gpu=4  
#SBATCH --mem=64G 
#SBATCH --partition=batch 
#SBATCH --mail-user=axel.aribowo@kaust.edu.sa #Your Email address assigned for your job
#SBATCH --mail-type=ALL #Receive an email for ALL Job Statuses
#SBATCH --output=results/%x/%j-slurm.out
#SBATCH --error=results/%x/%j-slurm.err





# activate the conda environment
conda init
conda activate RL_env_new


#################### Training ####################
python scripts/RL_train_scripts/train_td3.py --epoch 500
