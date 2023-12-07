#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH -n 4
#SBATCH --mem=60G
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH -J Train-MM-SA-ADNI
#SBATCH -o logs/train-MM-SA-%j.out
cd /gpfs/data/rsingh47/jgopal/DLGenomicsProject/training/
module load anaconda/3-5.2.0
module load cudnn/8.2.0
module load cuda/11.1.1
module load gcc/10.2
module load python/3.9.0

source activate CSCI2952G

python3 train_all_modalities_SA.py

