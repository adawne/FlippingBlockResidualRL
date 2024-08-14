#!/bin/bash --login

#SBATCH --job-name=FlippingEnv-v0
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=v100:8
#SBATCH --cpus-per-gpu=4  
#SBATCH --mem=64G 
#SBATCH --partition=batch 
#SBATCH --mail-user=axel.aribowo@kaust.edu.sa #Your Email address assigned for your job
#SBATCH --mail-type=ALL #Receive an email for ALL Job Statuses
#SBATCH --output=results/%x/%j-slurm.out
#SBATCH --error=results/%x/%j-slurm.err



# activate the conda environment
conda init
conda activate RL_env


#################### Training ####################
python train_dqn.py --units-per-layer 128 \
                --hidden-layers 2 \
                --target-update-freq 10 \
                --buffer-size 20000 \
                --batch-size 128 \
                --lr 1e-3 \
                --eps-decay-steps 70000 \
                --step-per-collect 5 \
                --prioritized-replay True\

